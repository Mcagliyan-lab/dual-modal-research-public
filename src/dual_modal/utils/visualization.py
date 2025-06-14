"""
Visualization Utilities for Dual-Modal Analysis
==============================================

Creates standardized visualizations for NN-EEG temporal and NN-fMRI spatial results.

STATUS: 🟡 BASIC PLOTTING, EXTENDING AS NEEDED  
CREATED: June 3, 2025
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import json

def plot_frequency_analysis(frequency_results: Dict, 
                          save_path: Optional[str] = None):
    """Plot NN-EEG frequency analysis results"""
    
    n_layers = len(frequency_results)
    fig, axes = plt.subplots(1, min(3, n_layers), figsize=(15, 5))
    
    if n_layers == 1:
        axes = [axes]
    
    for idx, (layer_name, results) in enumerate(list(frequency_results.items())[:3]):
        ax = axes[idx] if n_layers > 1 else axes[0]
        
        if 'frequencies' in results and 'power_spectral_density' in results:
            freqs = np.array(results['frequencies'])
            psd = np.array(results['power_spectral_density']) 
            
            ax.semilogy(freqs, psd, 'b-', linewidth=2)
            ax.set_title(f'{layer_name}\nDominant: {results.get("dominant_frequency", 0):.3f} Hz')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power Spectral Density')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Frequency analysis plot saved: {save_path}")
    
    plt.show()

def plot_spatial_grids(spatial_results: Dict,
                      save_path: Optional[str] = None):
    """Plot NN-fMRI spatial grid results"""
    
    print("TODO: Implement spatial grid visualization")
    print("Will create:")
    print("- Grid-based heatmaps")
    print("- Critical region highlighting")
    print("- Zeta-score distributions")
    
    # PLACEHOLDER
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.text(0.5, 0.5, 'Spatial Grid Visualization\n(TODO: Implement)', 
            ha='center', va='center', fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def create_dual_modal_dashboard(nn_eeg_results: Dict, 
                               nn_fmri_results: Dict,
                               save_path: Optional[str] = None):
    """Create comprehensive dual-modal visualization dashboard"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # NN-EEG temporal plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("NN-EEG Frequency Analysis")
    ax1.text(0.5, 0.5, 'Temporal\nFrequency\nAnalysis', ha='center', va='center')
    
    ax2 = fig.add_subplot(gs[0, 1]) 
    ax2.set_title("State Classification")
    ax2.text(0.5, 0.5, 'Operational\nState\nDetection', ha='center', va='center')
    
    # NN-fMRI spatial plots
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("NN-fMRI Spatial Grids") 
    ax3.text(0.5, 0.5, 'Spatial\nGrid\nAnalysis', ha='center', va='center')
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title("Zeta-Score Distribution")
    ax4.text(0.5, 0.5, 'Impact\nAssessment\nScores', ha='center', va='center')
    
    # Integration results
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.set_title("Cross-Modal Integration")
    ax5.text(0.5, 0.5, 'Temporal-Spatial Correlation\nCross-Modal Validation', 
             ha='center', va='center')
    
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard saved: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Visualization utilities ready")
    print("Available: plot_frequency_analysis, plot_spatial_grids, create_dual_modal_dashboard")
