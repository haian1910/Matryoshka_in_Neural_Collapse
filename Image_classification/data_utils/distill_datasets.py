import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.distributed as dist
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import json
from typing import Dict, Optional

class ImageDistillDataset(Dataset):
    def __init__(
        self,
        args,
        split: str,
        transform=None,
    ):
        """
        Dataset for image classification distillation
        
        Args:
            args: Arguments containing data_dir and other config
            split: 'train', 'dev', or 'test'
            transform: Image transformations to apply
        """
        self.args = args
        self.split = split
        self.transform = transform
        
        self.dataset = self._load_and_process_data()

    def __len__(self):
        return len(self.dataset)
   
    def __getitem__(self, index):
        return self.dataset[index]
    
    def _load_and_process_data(self):
        dataset = []
        
        # Try CSV format first (image_path, label)
        csv_path = os.path.join(self.args.data_dir, f"{self.split}.csv")
        if os.path.exists(csv_path):
            return self._load_from_csv(csv_path)
        
        # Try ImageFolder format (data_dir/split/class/image.jpg)
        folder_path = os.path.join(self.args.data_dir, self.split)
        if os.path.exists(folder_path):
            return self._load_from_folder(folder_path)
        
        raise FileNotFoundError(f"No data found at {csv_path} or {folder_path}")
    
    def _load_from_csv(self, csv_path):
        """Load dataset from CSV file with columns: image_path, label"""
        dataset = []
        df = pd.read_csv(csv_path)
        
        required_cols = ['image_path', 'label']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV file {csv_path} must contain column '{col}'")
        
        print(f"Processing {len(df)} images from CSV for {self.split} split...")
        
        # Get unique labels and create mapping if labels are strings
        unique_labels = sorted(df['label'].unique())
        if isinstance(unique_labels[0], str):
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.classes = unique_labels
        else:
            self.classes = [str(label) for label in unique_labels]
            self.label_to_idx = {str(label): label for label in unique_labels}
        
        for _, row in tqdm(df.iterrows(), total=len(df), disable=(dist.get_rank() != 0)):
            image_path = os.path.join(self.args.data_dir, row['image_path'])
            
            # Convert label to integer if it's a string
            if isinstance(row['label'], str):
                label = self.label_to_idx[row['label']]
            else:
                label = int(row['label'])
            
            dataset.append({
                "image_path": image_path,
                "label": label
            })
        
        return dataset
    
    def _load_from_folder(self, folder_path):
        """Load dataset from ImageFolder structure"""
        print(f"Processing images from folder structure for {self.split} split...")
        
        # Create a temporary transform to get the dataset structure
        temp_transform = transforms.ToTensor()
        image_folder = ImageFolder(root=folder_path, transform=temp_transform)
        
        self.classes = image_folder.classes
        self.class_to_idx = image_folder.class_to_idx
        
        dataset = []
        for image_path, label in tqdm(image_folder.samples, disable=(dist.get_rank() != 0)):
            dataset.append({
                "image_path": image_path,
                "label": label
            })
        
        return dataset
    
    def _process_image(self, i, samp, model_data, output_data):
        """Process a single image sample"""
        # Load and transform image
        image = Image.open(samp["image_path"]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        model_data["images"][i] = image
        output_data["labels"][i] = torch.tensor(samp["label"], dtype=torch.long)

    def move_to_device(self, datazip, device):
        """Move data to specified device"""
        for data in datazip:
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].to(device)

    def collate(self, samples):
        """
        Collate function for DataLoader
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            model_data: Dict with 'images' key
            output_data: Dict with 'labels' key
        """
        bs = len(samples)
        
        # Determine image dimensions from first sample
        first_image_path = samples[0]["image_path"]
        temp_image = Image.open(first_image_path).convert('RGB')
        if self.transform:
            temp_image = self.transform(temp_image)
            if isinstance(temp_image, torch.Tensor):
                img_shape = temp_image.shape
            else:
                # Fallback if transform doesn't return tensor
                temp_image = transforms.ToTensor()(temp_image)
                img_shape = temp_image.shape
        else:
            temp_image = transforms.ToTensor()(temp_image)
            img_shape = temp_image.shape
        
        model_data = {
            "images": torch.zeros(bs, *img_shape)
        }
        
        output_data = {
            "labels": torch.zeros(bs, dtype=torch.long)
        }

        for i, samp in enumerate(samples):
            self._process_image(i, samp, model_data, output_data)
        
        return model_data, output_data
    
    def get_class_names(self):
        """Return list of class names"""
        return self.classes
    
    def get_num_classes(self):
        """Return number of classes"""
        return len(self.classes)