import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Dict
from .matry_CE import Matry_CrossEntropyLoss

class OrthogonalProjection(nn.Module):
    def __init__(self, in_dim=768, out_dim=4096):
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

class NC1(Matry_CrossEntropyLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate
        self.nesting_list = getattr(args, 'mrl_nesting_list', [128, 256, 512, 768])
        self.mrl_efficient = getattr(args, 'mrl_efficient', False)
        
        # Initialize projection matrices for each nesting dimension
        # Assuming teacher has hidden size 4096 (from LLM2Vec-Mistral-7B)
        self.teacher_hidden_size = 4096
        self.projectors = nn.ModuleDict()
        
        for dim in self.nesting_list:
            self.projectors[f'proj_{dim}'] = OrthogonalProjection(
                in_dim=dim, 
                out_dim=self.teacher_hidden_size
            )
    
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
        
        # Student forward pass
        model_outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
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
        
        # Compute distillation loss
        kd_loss, log = self.compute_nc1_loss(
            model_outputs, teacher_outputs, output_data, input_data, distiller, log
        )
        print("nc1_loss:", kd_loss)
        
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

    def compute_nc1_loss(
        self, model_outputs, teacher_outputs, output_data, input_data, distiller, log
    ):
        """
        Compute NC1 distillation loss following the mathematical formulation:
        
        For each student subnet dimension d1:
        1. Get student embedding Z_S^{d1} âˆˆ R^{nÃ—d1}
        2. Project to teacher space: áº_S^{d1} = Z_S^{d1} P^{d1}
        3. Compute MSE loss: L_nc1-distill^{d1} = (1/n) Î£ ||áº'_{S,i}^{d1} - z_{T,i}||_2^2
        4. Sum over all dimensions: L_nc1-distill = Î£ L_nc1-distill^{d1}
        """
        
        # Get teacher embeddings (Z_T)
        # Extract teacher hidden states - Handle different model architectures
        if hasattr(teacher_outputs, 'hidden_states') and teacher_outputs.hidden_states is not None:
            # For LLM2Vec model loaded with AutoModelForSequenceClassification
            teacher_hidden = teacher_outputs.hidden_states[-1]  # Last layer hidden states
        elif isinstance(teacher_outputs, dict) and 'hidden_states' in teacher_outputs:
            # If teacher_outputs is a dict with hidden_states key
            teacher_hidden = teacher_outputs['hidden_states'][-1]
        else:
            raise ValueError("Cannot extract teacher hidden states")
        
        # Extract CLS token representation from teacher
        if teacher_hidden.dim() == 3:  # [batch_size, sequence_length, hidden_size]
            teacher_embeddings = teacher_hidden.mean(dim=1)  # Take mean across sequence length
        elif teacher_hidden.dim() == 2:  # [batch_size, hidden_size] - already CLS representation
            teacher_embeddings = teacher_hidden
        else:
            raise ValueError(f"Unexpected dimension for teacher_hidden: {teacher_hidden.shape}")

        
        
        # Get student embeddings for different nesting dimensions
        student_embeddings_dict = {}
        
        if hasattr(model_outputs, 'hidden_states') and model_outputs.hidden_states is not None:
            # Use the last hidden state and extract [CLS] token
            last_hidden_state = model_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            full_student_embeddings = last_hidden_state[:, 0]  # [batch_size, hidden_size]
        else:
            # Fallback: try to get embeddings from the model structure
            # For BertWithMRLWrapper, we need to access the BERT model's outputs
            if hasattr(distiller.student_model, 'bert'):
                # Re-run the BERT part to get hidden states
                bert_outputs = distiller.student_model.bert(
                    input_ids=input_data["input_ids"],
                    attention_mask=input_data['attention_mask'],
                    output_hidden_states=True
                )
                if hasattr(bert_outputs, 'pooler_output') and bert_outputs.pooler_output is not None:
                    full_student_embeddings = bert_outputs.pooler_output
                else:
                    full_student_embeddings = bert_outputs.last_hidden_state[:, 0]
            else:
                raise ValueError("Cannot extract student embeddings")
        
        # Ensure full_student_embeddings is on the correct device and dtype
        device = next(distiller.student_model.parameters()).device
        target_dtype = next(distiller.student_model.parameters()).dtype
        
        full_student_embeddings = full_student_embeddings.to(device=device, dtype=target_dtype)
        teacher_embeddings = teacher_embeddings.to(device=device, dtype=target_dtype)
        
        # Create student embeddings for each nesting dimension by truncating
        for dim in self.nesting_list:
            student_embeddings_dict[dim] = full_student_embeddings[:, :dim]  # [batch_size, dim]
        
        # Ensure projectors are on the correct device and dtype
        if not next(self.projectors.parameters()).device == device:
            self.projectors.to(device=device, dtype=target_dtype)
        
        # Compute NC1 distillation loss
        total_nc1_loss = 0.0
        nc1_losses_per_dim = {}
        
        for dim in self.nesting_list:
            # Get student embeddings for this dimension: Z_S^{d1}
            student_emb = student_embeddings_dict[dim]  # [batch_size, dim]
            
            # Ensure student embeddings are on correct device and dtype
            student_emb = student_emb.to(device=device, dtype=target_dtype)
            
            # Get projection matrix: P^{d1}
            projector = self.projectors[f'proj_{dim}']
            
            # Project student embeddings to teacher space: áº_S^{d1} = Z_S^{d1} P^{d1}
            projected_student_emb = projector(student_emb)  # [batch_size, teacher_hidden_size]
            
            # Compute MSE loss: L_nc1-distill^{d1} = (1/n) Î£ ||áº'_{S,i}^{d1} - z_{T,i}||_2^2
            mse_loss = nn.MSELoss()(projected_student_emb, teacher_embeddings)
            
            # Add to total loss
            total_nc1_loss += mse_loss
            nc1_losses_per_dim[f'nc1_loss_dim_{dim}'] = mse_loss.detach()  # Keep as tensor
            
            # Optional: Add orthogonal regularization
            if hasattr(self, 'use_ortho_reg') and self.use_ortho_reg:
                ortho_loss = projector.orthogonal_regularization_loss()
                total_nc1_loss += 1 * ortho_loss  # Small weight for regularization
        
        # Update log with individual dimension losses (keep as tensors)
        log.update(nc1_losses_per_dim)
        log['nc1_loss_total'] = total_nc1_loss.detach()  # Keep as tensor
        
        return total_nc1_loss, log

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
        else:
            # Alternative: access the BERT model directly
            if hasattr(distiller.student_model, 'bert'):
                # This might require storing input_ids and attention_mask somewhere accessible
                raise NotImplementedError("Need to store input data to re-run BERT forward pass")
            else:
                raise ValueError("Cannot extract student embeddings")
        
        # Create truncated embeddings for each nesting dimension
        for dim in self.nesting_list:
            if dim <= full_embeddings.shape[1]:
                embeddings_dict[dim] = full_embeddings[:, :dim]
            else:
                # Handle case where requested dimension is larger than available
                embeddings_dict[dim] = torch.cat([
                    full_embeddings, 
                    torch.zeros(full_embeddings.shape[0], dim - full_embeddings.shape[1], 
                               device=full_embeddings.device, dtype=full_embeddings.dtype)
                ], dim=1)
        
        return embeddings_dict
