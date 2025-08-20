import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Dict, Optional, Tuple
from .matry_CE import Matry_CrossEntropyLoss
from .accumulate import MeanAccumulator
import torch.nn.functional as F

class OrthogonalProjection(nn.Module):
    def __init__(self, in_dim=768, out_dim=2048):
        super(OrthogonalProjection, self).__init__()
        # Create a regular linear layer first
        self.projector = nn.Linear(in_dim, out_dim, bias=False)
        # Initialize with orthogonal weights (in float32)
        with torch.no_grad():
            nn.init.orthogonal_(self.projector.weight)

    def forward(self, x):
        # Handle dtype conversion - ensure projector weights match input dtype
        if x.dtype != self.projector.weight.dtype:
            # Convert projector weights to match input dtype
            self.projector.weight.data = self.projector.weight.data.to(x.dtype)
        
        return self.projector(x)
        
    def orthogonal_regularization_loss(self):
        """
        Optional: Add this to your total loss to maintain orthogonality during training
        L_ortho = ||W^T W - I||_F^2
        """
        W = self.projector.weight  # [out_dim, in_dim]
        if W.shape[0] >= W.shape[1]:  # out_dim >= in_dim
            # W^T W should be identity
            WtW = torch.mm(W.t(), W)  # [in_dim, in_dim]
            I = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
        else:  # out_dim < in_dim  
            # W W^T should be identity
            WWt = torch.mm(W, W.t())  # [out_dim, out_dim]
            I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            WtW = WWt
        
        ortho_loss = torch.norm(WtW - I, p='fro') ** 2
        return ortho_loss

class NC2(Matry_CrossEntropyLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate
        self.nesting_list = getattr(args, 'mrl_nesting_list', [128, 256, 512, 768])
        self.mrl_efficient = getattr(args, 'mrl_efficient', False)
        
        # NC2 specific hyperparameters
        self.nc2_lambda = getattr(args, 'nc2_lambda', 1.0)  # Weight for NC2 loss
        self.ortho_lambda = getattr(args, 'ortho_lambda', 1)  # Weight for orthogonality loss
        self.ema_momentum = getattr(args, 'ema_momentum', 0.95)  # EMA momentum (beta in paper)
        self.nc2_alpha = getattr(args, 'nc2_alpha', 0.5)  # Interpolation between batch and EMA
        self.epsilon = 1e-8  # Small constant for numerical stability
        
        # Initialize projection matrices for each nesting dimension
        # Assuming teacher has hidden size 1024
        self.teacher_hidden_size = 1024
        self.projectors = nn.ModuleDict()
        
        for dim in self.nesting_list:
            self.projectors[f'proj_{dim}'] = OrthogonalProjection(
                in_dim=dim, 
                out_dim=self.teacher_hidden_size
            )
        
        # Initialize EMA class means for each dimension
        self.register_buffer('teacher_class_means', None)
        self.register_buffer('teacher_gram', None)
        self.ema_class_means = {}
        for dim in self.nesting_list:
            self.register_buffer(f'ema_means_{dim}', None)
        
        # Initialize accumulators for computing class means
        self.accumulators = {}
        for dim in self.nesting_list:
            self.accumulators[dim] = None
    
    def to(self, device, dtype=None):
        """Override to method to ensure projectors are moved to the correct device and dtype"""
        super().to(device, dtype=dtype)
        self.projectors.to(device, dtype=dtype)
        return self
    
    def cuda(self, device=None):
        """Override cuda method to ensure projectors are moved to CUDA"""
        super().cuda(device)
        self.projectors.cuda(device)
        return self
    
    def compute_teacher_targets(self, teacher_outputs, labels, device):
        """
        Pre-compute teacher class means and Gram matrix (done once after teacher training)
        This should ideally be called during initialization with the full training set
        """
        if self.teacher_class_means is None:
            # This is a simplified version - in practice, you'd compute this over the full dataset
            # For now, we'll compute batch statistics as a proxy
            with torch.no_grad():
                # Get teacher embeddings ([CLS] token from last hidden state)
                if hasattr(teacher_outputs, 'hidden_states') and teacher_outputs.hidden_states is not None:
                    teacher_embeddings = teacher_outputs.hidden_states[-1][:, 0]  # [batch_size, teacher_hidden_size]
                else:
                    # Fallback: use pooler output if available
                    teacher_embeddings = teacher_outputs.pooler_output
                
                # Initialize teacher class means (simplified - using batch stats)
                num_classes = self.num_labels
                self.teacher_class_means = torch.zeros(num_classes, self.teacher_hidden_size, 
                                                       device=device, dtype=teacher_embeddings.dtype)
                class_counts = torch.zeros(num_classes, device=device)
                
                # Accumulate class means
                for i in range(num_classes):
                    mask = (labels == i)
                    if mask.sum() > 0:
                        self.teacher_class_means[i] = teacher_embeddings[mask].mean(dim=0)
                        class_counts[i] = mask.sum()
                
                # Compute teacher Gram matrix and normalize
                self.teacher_gram = torch.mm(self.teacher_class_means, self.teacher_class_means.t())
                self.teacher_gram = self.teacher_gram / (torch.norm(self.teacher_gram, p='fro') + self.epsilon)
    
    def compute_student_class_means(self, embeddings, labels, dim, batch_size):
        """
        Compute student class means for a batch
        """
        device = embeddings.device
        dtype = embeddings.dtype
        num_classes = self.num_labels
        
        # Initialize batch class means
        batch_class_means = torch.zeros(num_classes, dim, device=device, dtype=dtype)
        class_counts = torch.zeros(num_classes, device=device)
        
        # Compute class means for classes present in batch
        for i in range(num_classes):
            mask = (labels == i)
            if mask.sum() > 0:
                batch_class_means[i] = embeddings[mask].mean(dim=0)
                class_counts[i] = mask.sum()
        
        return batch_class_means, class_counts
    
    def update_ema_means(self, batch_means, class_counts, dim):
        """
        Update EMA class means for a specific dimension
        """
        ema_buffer_name = f'ema_means_{dim}'
        ema_means = getattr(self, ema_buffer_name)
        
        if ema_means is None:
            # Initialize EMA means
            setattr(self, ema_buffer_name, batch_means.clone())
        else:
            # Update EMA for classes present in batch
            mask = (class_counts > 0)
            ema_means[mask] = self.ema_momentum * ema_means[mask] + (1 - self.ema_momentum) * batch_means[mask]
            setattr(self, ema_buffer_name, ema_means)
        
        return getattr(self, ema_buffer_name)
    
    def compute_gram_matrix(self, class_means, normalize=True):
        """
        Compute Gram matrix from class means
        """
        gram = torch.mm(class_means, class_means.t())
        if normalize:
            gram = gram / (torch.norm(gram, p='fro') + self.epsilon)
        return gram
    
    def compute_nc2_batch_loss(self, student_embeddings, labels, dim, teacher_gram_subset):
        """
        Compute mini-batch NC2 loss for a specific dimension
        """
        batch_size = student_embeddings.shape[0]
        device = student_embeddings.device
        
        # Project student embeddings
        projector = self.projectors[f'proj_{dim}']
        projected_embeddings = projector(student_embeddings)
        
        # Compute batch class means
        batch_means, class_counts = self.compute_student_class_means(
            projected_embeddings, labels, self.teacher_hidden_size, batch_size
        )
        
        # Get classes present in batch
        present_classes = (class_counts > 0).nonzero(as_tuple=True)[0]
        
        if len(present_classes) > 1:  # Need at least 2 classes for meaningful Gram matrix
            # Extract means for present classes
            batch_means_subset = batch_means[present_classes]
            
            # Compute student Gram matrix for batch
            student_gram = self.compute_gram_matrix(batch_means_subset, normalize=True)
            
            # Compute NC2 loss (Frobenius norm of difference)
            nc2_loss = torch.norm(student_gram - teacher_gram_subset, p='fro') ** 2
        else:
            nc2_loss = torch.tensor(0.0, device=device)
        
        return nc2_loss, batch_means, class_counts, projected_embeddings
    
    def compute_nc2_ema_loss(self, student_embeddings, labels, dim):
        """
        Compute EMA-stabilized NC2 loss for a specific dimension
        """
        device = student_embeddings.device
        
        # Project student embeddings
        projector = self.projectors[f'proj_{dim}']
        projected_embeddings = projector(student_embeddings)
        
        # Compute batch class means
        batch_means, class_counts = self.compute_student_class_means(
            projected_embeddings, labels, self.teacher_hidden_size, student_embeddings.shape[0]
        )
        
        # Update EMA means
        ema_means = self.update_ema_means(batch_means, class_counts, dim)
        
        # Project EMA means (they're already in teacher space)
        ema_means_proj = ema_means
        
        # Compute EMA-based student Gram
        student_gram_ema = self.compute_gram_matrix(ema_means_proj, normalize=True)
        
        # Compute NC2 loss against full teacher Gram
        nc2_loss = torch.norm(student_gram_ema - self.teacher_gram, p='fro') ** 2
        
        return nc2_loss
    
    def compute_nc2_loss(
        self, model_outputs, teacher_outputs, output_data, input_data, distiller, log
    ):
        """
        Compute NC2 distillation loss following the mathematical formulation in the paper
        """
        device = output_data["labels"].device
        labels = output_data["labels"]
        batch_size = labels.shape[0]
        
        # Ensure teacher targets are computed
        self.compute_teacher_targets(teacher_outputs, labels, device)
        
        # Get student embeddings for different dimensions
        student_embeddings_dict = self.get_student_embeddings_by_dimension(model_outputs, distiller)
        
        total_nc2_loss = 0.0
        total_ortho_loss = 0.0
        
        # Compute losses for each nesting dimension
        for dim in self.nesting_list:
            if dim not in student_embeddings_dict:
                continue
            
            student_embeddings = student_embeddings_dict[dim]
            
            # Get classes present in batch for teacher Gram subset
            unique_classes = labels.unique()
            if self.teacher_gram is not None and len(unique_classes) > 1:
                teacher_gram_subset = self.teacher_gram[unique_classes][:, unique_classes]
                teacher_gram_subset = teacher_gram_subset / (torch.norm(teacher_gram_subset, p='fro') + self.epsilon)
            else:
                teacher_gram_subset = None
            
            # Compute batch NC2 loss
            if teacher_gram_subset is not None:
                batch_nc2_loss, batch_means, class_counts, projected_embeddings = self.compute_nc2_batch_loss(
                    student_embeddings, labels, dim, teacher_gram_subset
                )
            else:
                batch_nc2_loss = torch.tensor(0.0, device=device)
            
            # Compute EMA NC2 loss
            ema_nc2_loss = self.compute_nc2_ema_loss(student_embeddings, labels, dim)
            
            # Combine batch and EMA losses
            nc2_loss_dim = self.nc2_alpha * batch_nc2_loss + (1 - self.nc2_alpha) * ema_nc2_loss
            
            # Add orthogonality regularization for this dimension's projector
            projector = self.projectors[f'proj_{dim}']
            ortho_loss_dim = projector.orthogonal_regularization_loss()
            
            # Accumulate losses
            total_nc2_loss += nc2_loss_dim
            total_ortho_loss += ortho_loss_dim
            
            # Log per-dimension losses
            log[f'nc2_loss_{dim}'] = nc2_loss_dim.item()
            log[f'ortho_loss_{dim}'] = ortho_loss_dim.item()
        
        # Combine all losses
        total_loss = self.nc2_lambda * total_nc2_loss + self.ortho_lambda * total_ortho_loss
        
        # Log total losses
        log['nc2_loss_total'] = total_nc2_loss.item()
        log['ortho_loss_total'] = total_ortho_loss.item()
        log['nc2_combined_loss'] = total_loss.item()
        
        return total_loss, log
    
    def get_student_embeddings_by_dimension(self, model_outputs, distiller):
        """
        Helper function to extract student embeddings for different nesting dimensions
        """
        embeddings_dict = {}
        
        # Try to get embeddings from model outputs
        if hasattr(model_outputs, 'hidden_states') and model_outputs.hidden_states is not None:
            # Use the last hidden state
            last_hidden_state = model_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            full_embeddings = last_hidden_state[:, 0]  # [CLS] token: [batch_size, hidden_size]
        elif isinstance(model_outputs, dict) and 'hidden_states' in model_outputs:
            # Dictionary output format
            last_hidden_state = model_outputs['hidden_states'][-1]
            full_embeddings = last_hidden_state[:, 0]
        else:
            # Alternative: try to get from the model directly if we have access to intermediate outputs
            # This might require modifying the forward pass to return hidden states
            raise ValueError("Cannot extract student embeddings. Ensure model returns hidden_states.")
        
        # Create truncated embeddings for each nesting dimension
        for dim in self.nesting_list:
            if dim <= full_embeddings.shape[1]:
                embeddings_dict[dim] = full_embeddings[:, :dim]
            else:
                # Handle case where requested dimension is larger than available
                # Pad with zeros
                embeddings_dict[dim] = torch.cat([
                    full_embeddings, 
                    torch.zeros(full_embeddings.shape[0], dim - full_embeddings.shape[1], 
                               device=full_embeddings.device, dtype=full_embeddings.dtype)
                ], dim=1)
        
        return embeddings_dict
    
    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        target = output_data["labels"]
        
        # Student forward pass with hidden states
        model_outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True  # Ensure we get hidden states
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

        # Teacher forward pass (no gradient)
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )
        
        # Initialize log dictionary
        log = {}
        
        # Compute NC2 distillation loss
        kd_loss, log = self.compute_nc2_loss(
            model_outputs, teacher_outputs, output_data, input_data, distiller, log
        )
        
        # Combine losses
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        # Compute accuracy from the first/main logits
        main_logits = list(logits_dict.values())[0]
        accuracy = self.compute_accuracy(
            main_logits, output_data["labels"]
        )
        log["accuracy"] = accuracy

        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss, logging_output
