"""
Metrics and Evaluation Utilities
===============================

Standardized metrics for evaluating dual-modal analysis results.

STATUS: 🟡 BASIC METRICS, EXTENDING AS NEEDED
CREATED: June 3, 2025
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json

class DualModalMetrics:
    """Comprehensive metrics for dual-modal analysis evaluation"""
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_frequency_analysis(self, frequency_results: Dict) -> Dict:
        """Evaluate NN-EEG frequency analysis quality"""
        
        metrics = {
            'n_layers_analyzed': len(frequency_results),
            'frequency_range': {'min': float('inf'), 'max': 0},
            'mean_dominant_frequency': 0,
            'std_dominant_frequency': 0,
            'spectral_entropy_stats': {},
            'analysis_quality': 'good'  # TODO: Define criteria
        }
        
        dominant_freqs = []
        entropies = []
        
        for layer_name, results in frequency_results.items():
            if 'dominant_frequency' in results:
                freq = results['dominant_frequency']
                dominant_freqs.append(freq)
                
                metrics['frequency_range']['min'] = min(metrics['frequency_range']['min'], freq)
                metrics['frequency_range']['max'] = max(metrics['frequency_range']['max'], freq)
            
            if 'spectral_entropy' in results.get('spectral_features', {}):
                entropies.append(results['spectral_features']['spectral_entropy'])
        
        if dominant_freqs:
            metrics['mean_dominant_frequency'] = np.mean(dominant_freqs)
            metrics['std_dominant_frequency'] = np.std(dominant_freqs)
        
        if entropies:
            metrics['spectral_entropy_stats'] = {
                'mean': np.mean(entropies),
                'std': np.std(entropies),
                'range': [min(entropies), max(entropies)]
            }
        
        return metrics
    
    def evaluate_spatial_analysis(self, spatial_results: Dict) -> Dict:
        """Evaluate NN-fMRI spatial analysis quality"""
        
        # TODO: Implement after NN-fMRI working
        return {
            'status': 'TODO - Implement after NN-fMRI complete',
            'planned_metrics': [
                'Grid coverage statistics',
                'Zeta-score distribution analysis', 
                'Critical region identification accuracy',
                'Spatial pattern coherence measures'
            ]
        }
    
    def evaluate_cross_modal_consistency(self, 
                                       nn_eeg_results: Dict,
                                       nn_fmri_results: Dict) -> Dict:
        """Evaluate consistency between temporal and spatial findings"""
        
        # TODO: Implement after both components working
        return {
            'status': 'TODO - Implement after dual-modal integration',
            'planned_metrics': [
                'Temporal-spatial correlation coefficient',
                'State agreement rate',
                'Cross-modal validation score',
                'Consistency confidence interval'
            ]
        }
    
    def compute_reproducibility_metrics(self, 
                                      results_list: List[Dict]) -> Dict:
        """Compute reproducibility statistics across multiple runs"""
        
        if len(results_list) < 2:
            return {'error': 'Need at least 2 runs for reproducibility analysis'}
        
        # Extract comparable metrics
        dominant_freqs_by_layer = {}
        
        for results in results_list:
            if 'layer_statistics' in results:
                for layer, stats in results['layer_statistics'].items():
                    if layer not in dominant_freqs_by_layer:
                        dominant_freqs_by_layer[layer] = []
                    dominant_freqs_by_layer[layer].append(stats['dominant_frequency'])
        
        reproducibility = {}
        for layer, freqs in dominant_freqs_by_layer.items():
            reproducibility[layer] = {
                'mean': np.mean(freqs),
                'std': np.std(freqs),
                'coefficient_of_variation': np.std(freqs) / np.mean(freqs) if np.mean(freqs) > 0 else 0,
                'n_runs': len(freqs)
            }
        
        # Overall reproducibility score
        cv_values = [metrics['coefficient_of_variation'] 
                    for metrics in reproducibility.values()]
        overall_reproducibility = 1 - np.mean(cv_values)  # High when CV is low
        
        return {
            'layer_reproducibility': reproducibility,
            'overall_score': overall_reproducibility,
            'interpretation': 'excellent' if overall_reproducibility > 0.95 else 
                           'good' if overall_reproducibility > 0.9 else 'needs_improvement'
        }

# Utility functions
def statistical_significance_test(group1: List[float], 
                                group2: List[float]) -> Dict:
    """Test statistical significance between two groups"""
    
    t_stat, p_value = stats.ttest_ind(group1, group2)
    effect_size = (np.mean(group1) - np.mean(group2)) / np.sqrt(
        (np.std(group1)**2 + np.std(group2)**2) / 2
    )
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'significant': p_value < 0.05,
        'interpretation': 'large_effect' if abs(effect_size) > 0.8 else
                        'medium_effect' if abs(effect_size) > 0.5 else 'small_effect'
    }

if __name__ == "__main__":
    print("Metrics utilities ready")
    print("Available: DualModalMetrics, statistical_significance_test")
