#!/usr/bin/env python3
"""
Real-world scenario tests for NN-Neuroimaging framework.
Includes medical diagnosis and autonomous vehicle test cases.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import time
from typing import Dict, Tuple, List
import cv2
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nn_neuroimaging.framework import NNNeuroimaging
from nn_neuroimaging.utils.metrics import calculate_metrics
from nn_neuroimaging.utils.visualization import visualize_attention

class MedicalDiagnosisTests:
    """Test suite for medical diagnosis scenarios."""
    
    @pytest.fixture(scope="class")
    def medical_model(self):
        """Load or create medical diagnosis model."""
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 2)  # Binary classification: benign vs malignant
        )
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def synthetic_medical_data(self):
        """Generate synthetic medical image data."""
        # Create synthetic melanoma-like images
        images = []
        labels = []
        
        for i in range(100):
            # Generate base skin texture
            img = np.random.normal(0.6, 0.1, (256, 256, 3))
            
            if i % 2 == 0:  # Create benign case
                # Add regular circular pattern
                cv2.circle(img, (128, 128), 40, (0.7, 0.5, 0.5), -1)
                labels.append(0)
            else:  # Create malignant case
                # Add irregular pattern with high contrast
                points = np.random.randint(80, 176, (10, 2))
                cv2.fillPoly(img, [points], (0.8, 0.3, 0.3))
                labels.append(1)
            
            images.append(img)
        
        return torch.tensor(np.array(images)).float().permute(0, 3, 1, 2), torch.tensor(labels)

    def test_melanoma_detection(self, medical_model, synthetic_medical_data):
        """Test melanoma detection scenario."""
        images, labels = synthetic_medical_data
        analyzer = NNNeuroimaging(medical_model)
        
        # Test normal cases
        normal_results = analyzer.analyze(images[labels == 0])
        assert normal_results["false_positive_rate"] < 0.05  # Less than 5% false positives
        
        # Test anomaly cases
        anomaly_results = analyzer.analyze(images[labels == 1])
        assert anomaly_results["detection_rate"] > 0.95  # Over 95% detection rate
        
        # Test processing time
        start_time = time.time()
        analyzer.analyze(images[0:1])
        processing_time = time.time() - start_time
        assert processing_time < 0.1  # Less than 100ms per image

    def test_vascular_pattern_analysis(self, medical_model, synthetic_medical_data):
        """Test vascular pattern analysis in Conv3_Block."""
        images, _ = synthetic_medical_data
        analyzer = NNNeuroimaging(medical_model)
        
        # Analyze specific convolutional layer
        layer_results = analyzer.analyze_layer("Conv3_Block", images[0:1])
        
        assert "activation_patterns" in layer_results
        assert "vascular_features" in layer_results
        assert layer_results["coherence_score"] > 0.8  # High spatial coherence

class AutonomousVehicleTests:
    """Test suite for autonomous vehicle scenarios."""
    
    @pytest.fixture(scope="class")
    def vehicle_model(self):
        """Load or create autonomous vehicle model."""
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 4)  # 4 classes: pedestrian, vehicle, sign, background
        )
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def synthetic_vehicle_data(self):
        """Generate synthetic autonomous vehicle scenario data."""
        images = []
        labels = []
        
        # Generate various scenarios
        for i in range(100):
            img = np.zeros((512, 512, 3))
            
            # Add background (road, buildings)
            img += np.random.normal(0.5, 0.1, (512, 512, 3))
            
            if i % 4 == 0:  # Pedestrian scenario
                # Add pedestrian-like shape
                cv2.rectangle(img, (200, 300), (230, 400), (0.7, 0.7, 0.7), -1)
                labels.append(0)
            elif i % 4 == 1:  # Vehicle scenario
                # Add vehicle-like shape
                cv2.rectangle(img, (150, 200), (350, 300), (0.6, 0.6, 0.8), -1)
                labels.append(1)
            elif i % 4 == 2:  # Traffic sign
                # Add sign-like shape
                cv2.circle(img, (256, 128), 30, (0.8, 0.2, 0.2), -1)
                labels.append(2)
            else:  # Background only
                labels.append(3)
            
            # Add random shadows
            if np.random.random() > 0.5:
                shadow = np.random.uniform(0.6, 0.9)
                img[256:, :] *= shadow
            
            images.append(img)
        
        return torch.tensor(np.array(images)).float().permute(0, 3, 1, 2), torch.tensor(labels)

    def test_pedestrian_detection_under_shadows(self, vehicle_model, synthetic_vehicle_data):
        """Test pedestrian detection under varying shadow conditions."""
        images, labels = synthetic_vehicle_data
        analyzer = NNNeuroimaging(vehicle_model)
        
        # Test detection under normal conditions
        normal_results = analyzer.analyze(images[labels == 0])
        assert normal_results["detection_rate"] > 0.95
        
        # Create shadow variations
        shadow_images = images[labels == 0].clone()
        shadow_images[:, :, 256:, :] *= 0.6  # Add strong shadows
        
        # Test detection under shadow conditions
        shadow_results = analyzer.analyze(shadow_images)
        assert shadow_results["detection_rate"] > 0.90  # Should still maintain high accuracy
        
        # Test early warning capability
        warning_time = analyzer.measure_warning_time(shadow_images[0:1])
        assert 200 <= warning_time <= 500  # 200-500ms as claimed

    def test_real_time_monitoring(self, vehicle_model, synthetic_vehicle_data):
        """Test real-time monitoring capabilities."""
        images, _ = synthetic_vehicle_data
        analyzer = NNNeuroimaging(vehicle_model)
        
        # Test continuous monitoring
        start_time = time.time()
        for i in range(10):  # Simulate 10 frames
            result = analyzer.analyze(images[i:i+1])
            frame_time = time.time() - start_time
            assert frame_time < 0.033  # Maintain 30+ FPS
            start_time = time.time()
        
        # Test anomaly detection latency
        anomaly_image = images[0].clone()
        anomaly_image[:, :, 200:300, 200:300] = 0  # Create sudden occlusion
        detection_time = analyzer.measure_detection_latency(anomaly_image)
        assert detection_time < 0.1  # Less than 100ms latency

class StressTests:
    """Long-running stress tests for stability verification."""
    
    @pytest.mark.slow
    def test_continuous_operation(self, vehicle_model, synthetic_vehicle_data):
        """Test system stability under continuous operation."""
        images, _ = synthetic_vehicle_data
        analyzer = NNNeuroimaging(vehicle_model)
        
        start_memory = self._get_memory_usage()
        start_time = time.time()
        
        # Run continuous analysis for 5 minutes
        while time.time() - start_time < 300:  # 5 minutes
            for i in range(len(images)):
                result = analyzer.analyze(images[i:i+1])
                
                # Check system health
                current_memory = self._get_memory_usage()
                assert current_memory - start_memory < 100  # Less than 100MB memory growth
                
                # Check processing time stability
                frame_time = time.time() - start_time
                assert frame_time < 0.033  # Maintain real-time performance
                
                start_time = time.time()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 