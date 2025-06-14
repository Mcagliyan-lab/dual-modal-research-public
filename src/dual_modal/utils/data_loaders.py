"""
Data Loading Utilities for Dual-Modal Analysis
==============================================

Standardized data loading and preprocessing for consistent experiments
across NN-EEG and NN-fMRI components.

STATUS: 🟡 BASIC STRUCTURE, EXTENDING AS NEEDED
CREATED: June 3, 2025
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from typing import Tuple, Optional, Dict, Any

class StandardDataLoader:
    """
    Standardized data loading for reproducible experiments
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
    def load_cifar10(self, batch_size: int = 32, 
                     train: bool = False) -> DataLoader:
        """Load CIFAR-10 dataset - our primary validation dataset"""
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Deterministic for reproducibility
            num_workers=2
        )
    
    def load_mnist(self, batch_size: int = 32, 
                   train: bool = False) -> DataLoader:
        """Load MNIST dataset - baseline validation"""
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        dataset = torchvision.datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    
    def create_synthetic_data(self, 
                            n_samples: int = 100,
                            input_shape: Tuple = (3, 32, 32),
                            n_classes: int = 10) -> DataLoader:
        """Create synthetic data for controlled testing"""
        
        # Generate random data with fixed seed
        torch.manual_seed(self.random_seed)
        
        data = torch.randn(n_samples, *input_shape)
        targets = torch.randint(0, n_classes, (n_samples,))
        
        dataset = TensorDataset(data, targets)
        return DataLoader(dataset, batch_size=32, shuffle=False)

# Global instances for easy access
cifar10_loader = StandardDataLoader().load_cifar10
mnist_loader = StandardDataLoader().load_mnist
synthetic_loader = StandardDataLoader().create_synthetic_data

if __name__ == "__main__":
    print("Data loading utilities ready")
    print("Available: cifar10_loader, mnist_loader, synthetic_loader")
