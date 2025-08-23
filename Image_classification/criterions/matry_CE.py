import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List

class Matryoshka_CE_Loss(nn.Module):
    def __init__(self, relative_importance: List[float]=None, **kwargs):
        super(Matryoshka_CE_Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(**kwargs)
        # relative importance shape: [G]
        self.relative_importance = relative_importance
    
    def forward(self, output, target):
        # output shape: [G granularities, N batch size, C number of classes]
        # target shape: [N batch size]
        # Calculate losses for each output and stack them. This is still O(N)
        losses = torch.stack([self.criterion(output_i, target) for output_i in output])
        
        # Set relative_importance to 1 if not specified
        rel_importance = torch.ones_like(losses) if self.relative_importance is None else torch.tensor(self.relative_importance, device=losses.device)
        
        # Apply relative importance weights
        weighted_losses = rel_importance * losses
        return weighted_losses.sum()

class Matry_CrossEntropyLoss(nn.Module):
    def __init__(self, args) -> None:
        super(Matry_CrossEntropyLoss, self).__init__()
        self.label_smoothing = args.label_smoothing
        
        # MRL specific parameters
        self.nesting_list = getattr(args, 'mrl_nesting_list', [128, 256, 512, 768])
        self.mrl_efficient = getattr(args, 'mrl_efficient', False)
        
        # Relative importance weights for different nesting dimensions
        # You can customize these weights based on your requirements
        self.relative_importance = getattr(args, 'mrl_relative_importance', None)
        if self.relative_importance is None:
            # Default: give more weight to larger dimensions
            if self.mrl_efficient:
                self.relative_importance = [1.0]  # Only one dimension in efficient mode
            else:
                # Progressive weighting: smaller dimensions get less weight
                num_dims = len(self.nesting_list)
                self.relative_importance = [(i + 1) / num_dims for i in range(num_dims)]
        
        # Initialize Matryoshka loss
        self.matryoshka_loss = Matryoshka_CE_Loss(
            relative_importance=self.relative_importance,
            label_smoothing=self.label_smoothing
        )
    
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        """
        Compute cross-entropy loss and accuracy for text classification with Matryoshka embeddings.
        - Expects logits dictionary with multiple nesting dimensions
        - batch_denom is typically the batch size.
        """
        self.distiller = distiller
        model = distiller.student_model
        target = output_data["labels"]

        # Get model outputs (returns dict with logits for each nesting dimension)
        model_outputs = model(
            input_ids=input_data['input_ids'],
            attention_mask=input_data['attention_mask']
        )
        
        # Extract logits dictionary
        if isinstance(model_outputs, dict) and 'logits' in model_outputs:
            logits_dict = model_outputs['logits']
        else:
            # Fallback for non-MRL models
            logits_dict = {'logits_full': model_outputs.logits if hasattr(model_outputs, 'logits') else model_outputs}

        # Compute loss and accuracy for MRL
        if len(logits_dict) > 1 or any('logits_' in key for key in logits_dict.keys()):
            # MRL case: multiple logits for different nesting dimensions
            loss, nll_loss = self.compute_matryoshka_cross_entropy_loss(logits_dict, target)
            correct = self.compute_matryoshka_accuracy(logits_dict, target)
        else:
            # Regular case: single logits
            logits = list(logits_dict.values())[0]
            loss, nll_loss = self.compute_cross_entropy_loss(logits, target)
            correct = self.compute_accuracy(logits, target)
        
        # Update logging output, return to main distillation
        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            {
                "loss": loss,
                "nll_loss": nll_loss,
                "correct": correct
            }
        )
        return loss, logging_output

    def compute_matryoshka_cross_entropy_loss(self, logits_dict, target):
        """
        Compute Matryoshka cross-entropy loss for multiple nesting dimensions.
        """
        # Convert logits dict to list ordered by nesting dimensions
        if self.mrl_efficient:
            # Efficient mode: only one logits tensor
            logits_list = [list(logits_dict.values())[0]]
        else:
            # Regular mode: order by nesting dimensions
            ordered_keys = []
            for nesting_size in self.nesting_list:
                key = f"logits_{nesting_size}"
                if key in logits_dict:
                    ordered_keys.append(key)
            
            if not ordered_keys:
                # Fallback: use all available logits
                ordered_keys = list(logits_dict.keys())
            
            logits_list = [logits_dict[key] for key in ordered_keys]
        
        # Stack logits: [G granularities, N batch size, C number of classes]
        stacked_logits = torch.stack(logits_list, dim=0)
        
        # Compute Matryoshka loss
        total_loss = self.matryoshka_loss(stacked_logits, target)
        
        # For logging, also compute NLL loss (using the largest dimension)
        largest_logits = logits_list[-1] if logits_list else torch.zeros_like(target, dtype=torch.float)
        _, nll_loss = self.compute_cross_entropy_loss(largest_logits, target)
        
        return total_loss, nll_loss

    def compute_matryoshka_accuracy(self, logits_dict, target):
        """
        Compute accuracy for Matryoshka embeddings (using the largest dimension).
        """
        if self.mrl_efficient:
            # Efficient mode: use the only available logits
            logits = list(logits_dict.values())[0]
        else:
            # Regular mode: use the largest nesting dimension for accuracy
            largest_key = f"logits_{max(self.nesting_list)}"
            if largest_key in logits_dict:
                logits = logits_dict[largest_key]
            else:
                # Fallback: use any available logits
                logits = list(logits_dict.values())[-1]
        
        return self.compute_accuracy(logits, target)

    def compute_cross_entropy_loss(self, logits, target):
        """
        Original cross-entropy loss computation for single logits tensor.
        """
        # Tính log softmax trên chiều lớp
        lprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        
        # Tính negative log likelihood loss
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1).mean()
        
        if self.label_smoothing > 0:
            # Tính mất mát mịn (smooth loss)
            smooth_loss = -lprobs.mean(dim=-1).mean()
            loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        else:
            loss = nll_loss
        
        return loss, nll_loss

    def compute_accuracy(self, logits, target):
        """
        Original accuracy computation for single logits tensor.
        """
        # Lấy chỉ số lớp có xác suất cao nhất
        pred = logits.argmax(dim=-1)
        
        # Tính số lượng mẫu dự đoán đúng
        correct = pred.eq(target).sum().float()
        accu = correct / target.size(0)
        return accu

    def record_logging_output(self, logging_output, batch_denom, content):
        """
        Record metrics like loss and accuracy for logging, handling distributed training.
        content = {
                "loss": loss,
                "nll_loss": nll_loss,
                "correct": correct
            }
        """
        
        for k, v in content.items():
            if k == "correct":
                # Sum the correct counts across processes
                record_v = v.clone()
                dist.all_reduce(record_v, dist.ReduceOp.SUM)
                record_v = record_v.item()
            else:
                # Normalize loss by batch_denom and average across processes
                record_v = v / batch_denom
                dist.all_reduce(record_v, dist.ReduceOp.SUM)
                record_v = record_v.item() / dist.get_world_size()
            if k in logging_output:
                logging_output[k].append(record_v)
            else:
                logging_output[k] = [record_v]
        return logging_output
