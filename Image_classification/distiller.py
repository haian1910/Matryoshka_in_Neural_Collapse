import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoTokenizer


class Distiller(nn.Module):
    def __init__(self, args, device):
        super(Distiller, self).__init__()
        self.args = args
        self.device = device
        self.student_model, self.student_tokenizer = self.load_student_model()
        
        if self.args.teacher_model_path is not None:
            self.teacher_model, self.teacher_tokenizers = self.load_teacher_model()
        else:
            self.teacher_model, self.teacher_tokenizers = None, {}

    
    def load_tokenizer(self, path):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return tokenizer
            
    def load_student_model(self):
        """
        Load student model based on configuration:
        - For KD: Load real ResNet-20 student
        - For SFT: Load fake student (ResNet-50) before distillation
        """
        num_classes = getattr(self.args, 'num_classes', 1000)
        
        if hasattr(self.args, 'distillation_mode') and self.args.distillation_mode == 'KD':
            # Case 1: Real student ResNet-18 for Knowledge Distillation
            print("Loading ResNet-18 student model for Knowledge Distillation")
            student_model = models.resnet18(pretrained=False, num_classes=num_classes)
            
            # Load pretrained weights if specified
            if hasattr(self.args, 'student_model_path') and self.args.student_model_path:
                print(f"Loading student weights from: {self.args.student_model_path}")
                checkpoint = torch.load(self.args.student_model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    student_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    student_model.load_state_dict(checkpoint)
        
        else:
            # Case 2: Fake student (ResNet-50) for SFT before distillation
            print("Loading ResNet-50 fake student model for SFT")
            student_model = models.resnet50(pretrained=False, num_classes=num_classes)
            
            # Initialize with random weights or load from checkpoint if specified
            if hasattr(self.args, 'student_model_path') and self.args.student_model_path:
                print(f"Loading fake student weights from: {self.args.student_model_path}")
                checkpoint = torch.load(self.args.student_model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    student_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    student_model.load_state_dict(checkpoint)
        
        student_model = student_model.to(self.device)
        
        # For image classification, we don't typically use tokenizers
        # But keeping it for compatibility with the original interface
        student_tokenizer = None
        
        return student_model, student_tokenizer

    def load_teacher_model(self):
        """
        Load teacher ResNet-50 from checkpoint after SFT
        """
        print("Loading ResNet-50 teacher model from checkpoint")
        
        num_classes = getattr(self.args, 'num_classes', 1000)
        
        # Create ResNet-50 architecture
        teacher_model = models.resnet50(pretrained=False, num_classes=num_classes)
        
        # Load teacher weights from checkpoint
        if os.path.exists(self.args.teacher_model_path):
            print(f"Loading teacher weights from: {self.args.teacher_model_path}")
            checkpoint = torch.load(self.args.teacher_model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                teacher_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                teacher_model.load_state_dict(checkpoint['state_dict'])
            else:
                teacher_model.load_state_dict(checkpoint)
                
            print("Teacher model loaded successfully")
        else:
            raise FileNotFoundError(f"Teacher model checkpoint not found at: {self.args.teacher_model_path}")
        
        teacher_model = teacher_model.to(self.device)
        teacher_model.eval()  # Set teacher to evaluation mode
        
        # Freeze teacher model parameters
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        # For image classification, we don't use tokenizers
        teacher_tokenizers = {}
        
        return teacher_model, teacher_tokenizers

    def forward(self, criterion, batch, logging_output, loss_denom):
        input_data = batch["input_batch"]
        output_data = batch["output_batch"]
        loss, logging_output = criterion(
            self,
            input_data, 
            output_data,
            logging_output,
            loss_denom,
        )
        return loss, logging_output
    
    def get_student_outputs(self, x):
        """Get student model outputs"""
        return self.student_model(x)
    
    def get_teacher_outputs(self, x):
        """Get teacher model outputs"""
        if self.teacher_model is None:
            return None
        
        with torch.no_grad():
            return self.teacher_model(x)
    
    def get_features(self, x, model_type='student'):
        """
        Extract intermediate features for feature-based distillation
        """
        if model_type == 'student':
            model = self.student_model
        elif model_type == 'teacher' and self.teacher_model is not None:
            model = self.teacher_model
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
        
        features = []
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        
        # Handle maxpool for ResNet-18/50 but not for custom ResNet-20
        if hasattr(model, 'maxpool'):
            x = model.maxpool(x)
        
        features.append(x)  # After initial conv
        
        x = model.layer1(x)
        features.append(x)  # After layer1
        
        x = model.layer2(x)
        features.append(x)  # After layer2
        
        x = model.layer3(x)
        features.append(x)  # After layer3
        
        if hasattr(model, 'layer4'):  # ResNet-18/50 has layer4
            x = model.layer4(x)
            features.append(x)  # After layer4
        
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        logits = model.fc(x)
        
        return features, logits