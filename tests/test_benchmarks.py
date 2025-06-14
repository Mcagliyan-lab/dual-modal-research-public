#!/usr/bin/env python3
"""
Benchmark test suite comparing NN-Neuroimaging with LIME and SHAP.
Tests performance, accuracy, and resource usage across different scenarios.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
import sys
import lime
import shap
from lime import lime_image
from typing import Dict, Tuple, Any
import psutil
import torch.utils.data as data_utils

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nn_neuroimaging.framework import NNNeuroimaging
from nn_neuroimaging.utils.metrics import calculate_metrics
from nn_neuroimaging.utils.config_utils import load_config

class BenchmarkTestSuite:
    """Comprehensive benchmark suite for comparing XAI methods."""
    
    @pytest.fixture(scope="class")
    def benchmark_model(self):
        """Create a standard CNN model for benchmarking."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def benchmark_data(self):
        """Create synthetic dataset for benchmarking."""
        data = torch.randn(1000, 3, 32, 32)
        labels = torch.randint(0, 10, (1000,))
        dataset = data_utils.TensorDataset(data, labels)
        return data_utils.DataLoader(dataset, batch_size=32, shuffle=False)

    def measure_performance(self, method: str, model: nn.Module, data: torch.Tensor) -> Dict[str, float]:
        """Measure performance metrics for a given XAI method."""
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        if method == "nn_neuroimaging":
            analyzer = NNNeuroimaging(model)
            result = analyzer.analyze(data)
        elif method == "lime":
            explainer = lime_image.LimeImageExplainer()
            result = explainer.explain_instance(
                data[0].numpy().transpose(1, 2, 0),
                model,
                top_labels=5,
                hide_color=0,
                num_samples=100
            )
        elif method == "shap":
            explainer = shap.DeepExplainer(model, data[:100])
            result = explainer.shap_values(data[0:1])

        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024

        return {
            "processing_time": end_time - start_time,
            "memory_usage": end_memory - start_memory,
            "result": result
        }

    def test_processing_time_comparison(self, benchmark_model, benchmark_data):
        """Compare processing time across methods."""
        batch = next(iter(benchmark_data))
        methods = ["nn_neuroimaging", "lime", "shap"]
        times = {}

        for method in methods:
            metrics = self.measure_performance(method, benchmark_model, batch[0])
            times[method] = metrics["processing_time"]

        # Assert NN-Neuroimaging is significantly faster
        assert times["nn_neuroimaging"] < times["lime"] / 50  # At least 50x faster
        assert times["nn_neuroimaging"] < times["shap"] / 50

        print(f"\nProcessing times (seconds):")
        for method, time_taken in times.items():
            print(f"{method}: {time_taken:.3f}")

    def test_memory_usage_comparison(self, benchmark_model, benchmark_data):
        """Compare memory usage across methods."""
        batch = next(iter(benchmark_data))
        methods = ["nn_neuroimaging", "lime", "shap"]
        memory_usage = {}

        for method in methods:
            metrics = self.measure_performance(method, benchmark_model, batch[0])
            memory_usage[method] = metrics["memory_usage"]

        # Assert NN-Neuroimaging uses less memory
        assert memory_usage["nn_neuroimaging"] < memory_usage["lime"] * 0.5
        assert memory_usage["nn_neuroimaging"] < memory_usage["shap"] * 0.5

        print(f"\nMemory usage (MB):")
        for method, usage in memory_usage.items():
            print(f"{method}: {usage:.2f}")

    def test_accuracy_comparison(self, benchmark_model, benchmark_data):
        """Compare accuracy and reliability across methods."""
        batch = next(iter(benchmark_data))
        methods = ["nn_neuroimaging", "lime", "shap"]
        accuracy = {}

        # Inject known anomalies for testing
        anomaly_data = batch[0].clone()
        anomaly_data[:, :, 16:24, 16:24] = 0  # Create dead zone

        for method in methods:
            # Analyze both normal and anomaly data
            normal_metrics = self.measure_performance(method, benchmark_model, batch[0])
            anomaly_metrics = self.measure_performance(method, benchmark_model, anomaly_data)
            
            # Calculate detection accuracy
            accuracy[method] = calculate_metrics(
                normal_metrics["result"],
                anomaly_metrics["result"]
            )

        # Assert NN-Neuroimaging achieves claimed accuracy
        assert accuracy["nn_neuroimaging"]["detection_rate"] > 0.97  # >97% as claimed
        assert accuracy["nn_neuroimaging"]["false_positive_rate"] < 0.01  # <1% as claimed

        print(f"\nDetection Accuracy:")
        for method, metrics in accuracy.items():
            print(f"{method}:")
            print(f"  Detection Rate: {metrics['detection_rate']:.3f}")
            print(f"  False Positive Rate: {metrics['false_positive_rate']:.3f}")

    def test_scalability_comparison(self, benchmark_model, benchmark_data):
        """Test scalability with increasing model and data size."""
        model_sizes = [(16, 32, 64), (32, 64, 128), (64, 128, 256)]
        batch_sizes = [32, 64, 128]
        
        for channels in model_sizes:
            model = self._create_model_with_size(channels)
            for batch_size in batch_sizes:
                data = self._create_batch_with_size(batch_size)
                
                metrics = self.measure_performance("nn_neuroimaging", model, data)
                
                # Assert processing time stays within reasonable bounds
                assert metrics["processing_time"] < 0.1 * batch_size  # Linear scaling
                assert metrics["memory_usage"] < 50 * batch_size  # Linear memory scaling

    def _create_model_with_size(self, channels: Tuple[int, int, int]) -> nn.Module:
        """Create model with specified channel sizes."""
        return nn.Sequential(
            nn.Conv2d(3, channels[0], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[0], channels[1], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[1], channels[2], 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[2], 10)
        )

    def _create_batch_with_size(self, batch_size: int) -> torch.Tensor:
        """Create synthetic batch with specified size."""
        return torch.randn(batch_size, 3, 32, 32)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 