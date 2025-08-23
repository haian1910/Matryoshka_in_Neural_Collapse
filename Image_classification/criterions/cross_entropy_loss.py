import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, args) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.label_smoothing = getattr(args, 'label_smoothing', 0.0)
    
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        """
        Compute cross-entropy loss and accuracy for image classification.
        - Expects images (batch_size, channels, height, width), target (batch_size,).
        - batch_denom is typically the batch size.
        """
        self.distiller = distiller
        model = distiller.student_model
        target = output_data["labels"]
        images = input_data["images"]

        # Forward pass through student model
        logits = model(images)

        # Compute loss and accuracy
        loss, nll_loss = self.compute_cross_entropy_loss(logits, target)
        correct = self.compute_accuracy(logits, target)
        
        # Update logging output
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

    def compute_cross_entropy_loss(self, logits, target):
        """Compute cross-entropy loss with optional label smoothing"""
        lprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1).mean()
        
        if self.label_smoothing > 0:
            smooth_loss = -lprobs.mean(dim=-1).mean()
            loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        else:
            loss = nll_loss
        
        return loss, nll_loss

    def compute_accuracy(self, logits, target):
        """Compute classification accuracy"""
        pred = logits.argmax(dim=-1)
        correct = pred.eq(target).sum().float()
        accu = correct / target.size(0)
        return accu

    def record_logging_output(self, logging_output, batch_denom, content):
        """
        Record metrics like loss and accuracy for logging (simplified for single GPU).
        content = {
                "loss": loss,
                "nll_loss": nll_loss,
                "correct": correct
            }
        """
        
        for k, v in content.items():
            if k == "correct":
                # Keep correct counts as is
                record_v = v.item()
            else:
                # Normalize loss by batch_denom
                record_v = (v / batch_denom).item()
            
            if k in logging_output:
                logging_output[k].append(record_v)
            else:
                logging_output[k] = [record_v]
        
        return logging_output


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for image classification
    Combines cross-entropy loss with KL divergence between teacher and student
    """
    def __init__(self, args) -> None:
        super(KnowledgeDistillationLoss, self).__init__()
        self.label_smoothing = getattr(args, 'label_smoothing', 0.0)
        self.temperature = getattr(args, 'kd_temperature', 4.0)
        self.alpha = getattr(args, 'kd_alpha', 0.7)  # Weight for KD loss
        self.beta = getattr(args, 'kd_beta', 0.3)    # Weight for CE loss
    
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        """
        Compute knowledge distillation loss combining CE and KL divergence
        """
        self.distiller = distiller
        student_model = distiller.student_model
        teacher_model = distiller.teacher_model
        target = output_data["labels"]
        images = input_data["images"]

        # Forward pass through student model
        student_logits = student_model(images)
        
        # Forward pass through teacher model (no gradients)
        with torch.no_grad():
            teacher_logits = teacher_model(images)

        # Compute cross-entropy loss (hard targets)
        ce_loss, nll_loss = self.compute_cross_entropy_loss(student_logits, target)
        
        # Compute KL divergence loss (soft targets)
        kd_loss = self.compute_kl_divergence(student_logits, teacher_logits)
        
        # Combined loss
        total_loss = self.beta * ce_loss + self.alpha * kd_loss
        
        # Compute accuracy
        correct = self.compute_accuracy(student_logits, target)
        
        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            {
                "loss": total_loss,
                "nll_loss": nll_loss,
                "kd_loss": kd_loss,
                "correct": correct
            }
        )
        return total_loss, logging_output
    
    def compute_cross_entropy_loss(self, logits, target):
        """Same as CrossEntropyLoss"""
        lprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1).mean()
        
        if self.label_smoothing > 0:
            smooth_loss = -lprobs.mean(dim=-1).mean()
            loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        else:
            loss = nll_loss
        
        return loss, nll_loss
    
    def compute_kl_divergence(self, student_logits, teacher_logits):
        """Compute KL divergence between teacher and student predictions"""
        # Apply temperature scaling
        student_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence
        kl_loss = nn.KLDivLoss(reduction='batchmean')(student_probs, teacher_probs)
        
        # Scale by temperature squared (standard practice)
        kl_loss = kl_loss * (self.temperature ** 2)
        
        return kl_loss
    
    def compute_accuracy(self, logits, target):
        """Same as CrossEntropyLoss"""
        pred = logits.argmax(dim=-1)
        correct = pred.eq(target).sum().float()
        accu = correct / target.size(0)
        return accu
    
    def record_logging_output(self, logging_output, batch_denom, content):
        """Same as CrossEntropyLoss but handles KD loss as well"""
        for k, v in content.items():
            if k == "correct":
                record_v = v.item()
            else:
                record_v = (v / batch_denom).item()
            
            if k in logging_output:
                logging_output[k].append(record_v)
            else:
                logging_output[k] = [record_v]
        
        return logging_output


class FeatureDistillationLoss(nn.Module):
    """
    Feature-based Knowledge Distillation Loss
    Matches intermediate features between teacher and student
    """
    def __init__(self, args) -> None:
        super(FeatureDistillationLoss, self).__init__()
        self.label_smoothing = getattr(args, 'label_smoothing', 0.0)
        self.temperature = getattr(args, 'kd_temperature', 4.0)
        self.alpha = getattr(args, 'kd_alpha', 0.7)
        self.beta = getattr(args, 'kd_beta', 0.3)
        self.gamma = getattr(args, 'feature_alpha', 0.5)  # Weight for feature loss
        
        # Feature matching layers (adapt dimensions if needed)
        # This assumes student and teacher have different feature dimensions
        # You may need to adjust based on your specific models
        self.feature_adapters = nn.ModuleList()
    
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        """
        Compute feature distillation loss
        """
        self.distiller = distiller
        target = output_data["labels"]
        images = input_data["images"]

        # Get features and logits from both models
        student_features, student_logits = distiller.get_features(images, 'student')
        teacher_features, teacher_logits = distiller.get_features(images, 'teacher')

        # Compute cross-entropy loss
        ce_loss, nll_loss = self.compute_cross_entropy_loss(student_logits, target)
        
        # Compute KL divergence loss  
        kd_loss = self.compute_kl_divergence(student_logits, teacher_logits)
        
        # Compute feature matching loss
        feature_loss = self.compute_feature_loss(student_features, teacher_features)
        
        # Combined loss
        total_loss = self.beta * ce_loss + self.alpha * kd_loss + self.gamma * feature_loss
        
        # Compute accuracy
        correct = self.compute_accuracy(student_logits, target)
        
        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            {
                "loss": total_loss,
                "nll_loss": nll_loss,
                "kd_loss": kd_loss,
                "feature_loss": feature_loss,
                "correct": correct
            }
        )
        return total_loss, logging_output
    
    def compute_cross_entropy_loss(self, logits, target):
        """Same as previous classes"""
        lprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1).mean()
        
        if self.label_smoothing > 0:
            smooth_loss = -lprobs.mean(dim=-1).mean()
            loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        else:
            loss = nll_loss
        
        return loss, nll_loss
    
    def compute_kl_divergence(self, student_logits, teacher_logits):
        """Same as KnowledgeDistillationLoss"""
        student_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=-1)
        
        kl_loss = nn.KLDivLoss(reduction='batchmean')(student_probs, teacher_probs)
        kl_loss = kl_loss * (self.temperature ** 2)
        
        return kl_loss
    
    def compute_feature_loss(self, student_features, teacher_features):
        """Compute MSE loss between student and teacher features"""
        feature_loss = 0
        
        # Match features from corresponding layers
        min_layers = min(len(student_features), len(teacher_features))
        
        for i in range(min_layers):
            s_feat = student_features[i]
            t_feat = teacher_features[i]
            
            # If dimensions don't match, you might need adaptation layers
            if s_feat.shape != t_feat.shape:
                # Simple adaptive pooling for spatial dimensions
                if len(s_feat.shape) == 4:  # Conv features [B, C, H, W]
                    t_feat = nn.AdaptiveAvgPool2d(s_feat.shape[2:])(t_feat)
                    if s_feat.shape[1] != t_feat.shape[1]:  # Channel mismatch
                        continue  # Skip this layer or implement channel adapter
            
            # MSE loss between features
            feature_loss += nn.MSELoss()(s_feat, t_feat)
        
        return feature_loss / min_layers if min_layers > 0 else torch.tensor(0.0, device=student_features[0].device)
    
    def compute_accuracy(self, logits, target):
        """Same as previous classes"""
        pred = logits.argmax(dim=-1)
        correct = pred.eq(target).sum().float()
        accu = correct / target.size(0)
        return accu
    
    def record_logging_output(self, logging_output, batch_denom, content):
        """Same as previous but handles feature loss"""
        for k, v in content.items():
            if k == "correct":
                record_v = v.item()
            else:
                record_v = (v / batch_denom).item()
            
            if k in logging_output:
                logging_output[k].append(record_v)
            else:
                logging_output[k] = [record_v]
        
        return logging_output