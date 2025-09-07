import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .cross_entropy_loss import CrossEntropyLoss
import random
from typing import List, Dict, Any


class MatryoshkaClassificationLoss(nn.Module):
    """
    Matryoshka loss adapted for classification tasks.
    This implements the core idea of training at multiple embedding dimensions.
    """
    def __init__(
        self,
        matryoshka_dims: List[int],
        matryoshka_weights: List[float] = None,
        n_dims_per_step: int = -1,
    ):
        super().__init__()
        if matryoshka_weights is None:
            matryoshka_weights = [1.0] * len(matryoshka_dims)
        
        # Sort dimensions and weights in descending order
        dims_weights = zip(matryoshka_dims, matryoshka_weights)
        self.matryoshka_dims, self.matryoshka_weights = zip(*sorted(dims_weights, key=lambda x: x[0], reverse=True))
        self.n_dims_per_step = n_dims_per_step
        
    def forward(self, logits_dict: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss across different matryoshka dimensions.
        
        Args:
            logits_dict: Dictionary with keys like 'logits_768', 'logits_512', etc.
            labels: Ground truth labels
        """
        loss_fct = nn.CrossEntropyLoss()
        
        # Determine which dimensions to use in this step
        dim_indices = range(len(self.matryoshka_dims))
        if self.n_dims_per_step > 0 and self.n_dims_per_step < len(dim_indices):
            dim_indices = random.sample(dim_indices, self.n_dims_per_step)
        
        total_loss = 0.0
        for idx in dim_indices:
            dim = self.matryoshka_dims[idx]
            weight = self.matryoshka_weights[idx]
            
            logits_key = f"logits_{dim}"
            if logits_key in logits_dict:
                loss = loss_fct(logits_dict[logits_key].view(-1, logits_dict[logits_key].size(-1)), labels.view(-1))
                total_loss += weight * loss
        
        # Normalize by number of dimensions used
        if len(list(dim_indices)) > 0:
            total_loss = total_loss / len(list(dim_indices))
        
        return total_loss


class AdaptiveLayerClassificationLoss(nn.Module):
    """
    Adaptive layer loss adapted for classification tasks.
    This trains intermediate layers to mimic the final layer's behavior.
    """
    def __init__(
        self,
        n_layers_per_step: int = 1,
        last_layer_weight: float = 1.0,
        prior_layers_weight: float = 1.0,
        kl_div_weight: float = 1.0,
        kl_temperature: float = 0.3,
    ):
        super().__init__()
        self.n_layers_per_step = n_layers_per_step
        self.last_layer_weight = last_layer_weight
        self.prior_layers_weight = prior_layers_weight
        self.kl_div_weight = kl_div_weight
        self.kl_temperature = kl_temperature
    
    def forward(
        self, 
        model,
        input_ids,
        attention_mask,
        labels,
        base_loss_fn,
        matryoshka_dims: List[int]
    ):
        """
        Compute loss across different layers of the model.
        """
        # Enable output of hidden states
        outputs = model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states  # Tuple of hidden states at each layer
        num_layers = len(hidden_states) - 1  # Exclude embedding layer
        
        total_loss = 0.0
        
        # Process final layer with full MRL
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            final_pooled = outputs.pooler_output
        else:
            final_pooled = hidden_states[-1][:, 0]  # [CLS] token from last layer
        
        # Get logits for all matryoshka dimensions at final layer
        final_logits_dict = {}
        for dim in matryoshka_dims:
            truncated_embedding = final_pooled[:, :dim]
            # Find the corresponding classifier
            for i, nesting_size in enumerate(model.mrl_classifier.nesting_list):
                if nesting_size == dim:
                    classifier = getattr(model.mrl_classifier, f"nesting_classifier_{i}")
                    final_logits_dict[f"logits_{dim}"] = classifier(truncated_embedding)
                    break
        
        # Compute loss for final layer
        final_loss = base_loss_fn(final_logits_dict, labels)
        total_loss += self.last_layer_weight * final_loss
        
        # Get final layer probabilities for KL divergence
        if self.kl_temperature > 0:
            # Use the largest dimension's logits for KL reference
            max_dim = max(matryoshka_dims)
            final_probs = F.softmax(final_logits_dict[f"logits_{max_dim}"] / self.kl_temperature, dim=-1)
        
        # Sample layers to train
        layer_indices = list(range(1, num_layers))  # Skip embedding layer and final layer
        if self.n_layers_per_step > 0 and self.n_layers_per_step < len(layer_indices):
            layer_indices = random.sample(layer_indices, self.n_layers_per_step)
        
        # Process sampled intermediate layers
        for layer_idx in layer_indices:
            layer_hidden = hidden_states[layer_idx][:, 0]  # [CLS] token
            
            # Project to match dimensions if necessary
            if layer_hidden.size(-1) != final_pooled.size(-1):
                # Simple linear projection - you might want to add this to the model
                projection = nn.Linear(layer_hidden.size(-1), final_pooled.size(-1)).to(layer_hidden.device)
                layer_pooled = projection(layer_hidden)
            else:
                layer_pooled = layer_hidden
            
            # Get logits for all matryoshka dimensions at this layer
            layer_logits_dict = {}
            for dim in matryoshka_dims:
                truncated_embedding = layer_pooled[:, :dim]
                for i, nesting_size in enumerate(model.mrl_classifier.nesting_list):
                    if nesting_size == dim:
                        classifier = getattr(model.mrl_classifier, f"nesting_classifier_{i}")
                        layer_logits_dict[f"logits_{dim}"] = classifier(truncated_embedding)
                        break
            
            # Compute loss for this layer
            layer_loss = base_loss_fn(layer_logits_dict, labels)
            # Weight by layer position (earlier layers get less weight)
            weight_factor = (layer_idx + 1) / num_layers
            total_loss += self.prior_layers_weight * weight_factor * layer_loss / len(layer_indices)
            
            # Add KL divergence loss
            if self.kl_temperature > 0 and f"logits_{max_dim}" in layer_logits_dict:
                layer_probs = F.log_softmax(layer_logits_dict[f"logits_{max_dim}"] / self.kl_temperature, dim=-1)
                kl_loss = F.kl_div(layer_probs, final_probs, reduction='batchmean')
                total_loss += self.kl_div_weight * kl_loss * self.kl_temperature
        
        return total_loss, final_logits_dict


class Epresso(CrossEntropyLoss):
    def __init__(self, args) -> None:
        super(Epresso, self).__init__()
        self.label_smoothing = args.label_smoothing
        
        # Matryoshka configuration
        self.matryoshka_dims = getattr(args, 'mrl_nesting_list', [768, 512, 256, 128, 64])
        self.matryoshka_weights = getattr(args, 'matryoshka_weights', None)
        self.n_dims_per_step = getattr(args, 'n_dims_per_step', -1)
        
        # Adaptive layer configuration
        self.use_adaptive_layers = getattr(args, 'use_adaptive_layers', True)
        self.n_layers_per_step = getattr(args, 'n_layers_per_step', 1)
        self.last_layer_weight = getattr(args, 'last_layer_weight', 1.0)
        self.prior_layers_weight = getattr(args, 'prior_layers_weight', 1.0)
        self.kl_div_weight = getattr(args, 'kl_div_weight', 1.0)
        self.kl_temperature = getattr(args, 'kl_temperature', 0.3)
        
        # Initialize loss components
        self.matryoshka_loss = MatryoshkaClassificationLoss(
            matryoshka_dims=self.matryoshka_dims,
            matryoshka_weights=self.matryoshka_weights,
            n_dims_per_step=self.n_dims_per_step
        )
        
        if self.use_adaptive_layers:
            self.adaptive_layer_loss = AdaptiveLayerClassificationLoss(
                n_layers_per_step=self.n_layers_per_step,
                last_layer_weight=self.last_layer_weight,
                prior_layers_weight=self.prior_layers_weight,
                kl_div_weight=self.kl_div_weight,
                kl_temperature=self.kl_temperature
            )
    
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        
        log = {}
        
        if self.use_adaptive_layers and hasattr(model, 'bert'):
            # Use adaptive layer loss (2D Matryoshka)
            loss, logits_dict = self.adaptive_layer_loss(
                model=model,
                input_ids=input_data["input_ids"],
                attention_mask=input_data["attention_mask"],
                labels=output_data["labels"],
                base_loss_fn=self.matryoshka_loss,
                matryoshka_dims=self.matryoshka_dims
            )
        else:
            # Standard forward pass with Matryoshka loss only
            outputs = model(
                input_data["input_ids"],
                attention_mask=input_data["attention_mask"],
                labels=output_data["labels"]
            )
            
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits_dict = outputs['logits']
            else:
                # Fallback if model doesn't return dict
                logits_dict = outputs
            
            # Apply Matryoshka loss
            loss = self.matryoshka_loss(logits_dict, output_data["labels"])
        
        log["loss"] = loss
        
        # Compute accuracy using the largest dimension
        max_dim = max(self.matryoshka_dims)
        if f"logits_{max_dim}" in logits_dict:
            predictions = torch.argmax(logits_dict[f"logits_{max_dim}"], dim=-1)
            accuracy = (predictions == output_data["labels"]).float().mean()
            log["accuracy"] = accuracy
        
        # Log individual dimension losses if needed
        if getattr(self.distiller.args, 'log_dimension_losses', False):
            with torch.no_grad():
                loss_fct = nn.CrossEntropyLoss()
                for dim in self.matryoshka_dims:
                    if f"logits_{dim}" in logits_dict:
                        dim_loss = loss_fct(
                            logits_dict[f"logits_{dim}"].view(-1, logits_dict[f"logits_{dim}"].size(-1)), 
                            output_data["labels"].view(-1)
                        )
                        log[f"loss_dim_{dim}"] = dim_loss
                        
                        dim_predictions = torch.argmax(logits_dict[f"logits_{dim}"], dim=-1)
                        dim_accuracy = (dim_predictions == output_data["labels"]).float().mean()
                        log[f"accuracy_dim_{dim}"] = dim_accuracy
        
        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        
        return loss, logging_output
