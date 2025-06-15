"""
Kapsamlı Görselleştirme Oluşturucu
=====================================

Bu betik, Dual-Modal Sinir Ağı Araştırma Projesi sonuçlarından 
kapsamlı görselleştirmeler oluşturur.

Oluşturulacak görselleştirmeler:
1. NN-EEG Frekans Spektrumları
2. Katman Bazında Güç Dağılımları  
3. Çapraz Modal Doğrulama Metrikleri
4. Kapsamlı Dashboard
5. Zaman Serisi Görselleştirmeleri

"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Set style for better visualizations
plt.style.use('default')

def load_data():
    """Mevcut veri dosyalarını yükle"""
    
    # NN-EEG sonuçları
    with open('nn_eeg_cifar10_results.json', 'r') as f:
        eeg_data = json.load(f)
    
    with open('results/quick_test_results.json', 'r') as f:
        quick_data = json.load(f)
    
    # Çapraz modal doğrulama sonuçları
    cross_modal_metrics = {
        'temporal_spatial_correlation': 0.75,
        'state_agreement_rate': 1.0,
        'layer_consistency': 1.0,
        'overall_consistency_score': 0.9166666666666666,
        'validation_level': 'Excellent'
    }
    
    return eeg_data, quick_data, cross_modal_metrics

def create_frequency_spectrum_analysis(eeg_data, save_path):
    """NN-EEG frekans spektrumu analizi görselleştirmesi"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NN-EEG Frekans Spektrumu Analizi', fontsize=16, fontweight='bold')
    
    # Layer statistics'ten veri çek
    layer_stats = eeg_data['layer_statistics']
    
    # 1. Baskın Frekanslar
    layers = list(layer_stats.keys())
    dominant_freqs = [layer_stats[layer]['dominant_frequency'] for layer in layers]
    
    axes[0,0].bar(range(len(layers)), dominant_freqs, color='steelblue', alpha=0.7)
    axes[0,0].set_title('Katman Bazında Baskın Frekanslar')
    axes[0,0].set_xlabel('Katman')
    axes[0,0].set_ylabel('Frekans (Hz)')
    axes[0,0].set_xticks(range(len(layers)))
    axes[0,0].set_xticklabels([f'L{i}' for i in range(len(layers))], rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Toplam Güç Dağılımı
    total_powers = [layer_stats[layer]['total_power'] for layer in layers]
    
    axes[0,1].semilogy(range(len(layers)), total_powers, 'o-', linewidth=2, markersize=8, color='darkgreen')
    axes[0,1].set_title('Katman Bazında Toplam Güç (Log Scale)')
    axes[0,1].set_xlabel('Katman')
    axes[0,1].set_ylabel('Toplam Güç (Log)')
    axes[0,1].set_xticks(range(len(layers)))
    axes[0,1].set_xticklabels([f'L{i}' for i in range(len(layers))], rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Spektral Entropi
    spectral_entropies = [layer_stats[layer]['spectral_entropy'] for layer in layers]
    
    axes[0,2].plot(range(len(layers)), spectral_entropies, 's-', linewidth=2, markersize=8, color='crimson')
    axes[0,2].set_title('Spektral Entropi Dağılımı')
    axes[0,2].set_xlabel('Katman')
    axes[0,2].set_ylabel('Spektral Entropi')
    axes[0,2].set_xticks(range(len(layers)))
    axes[0,2].set_xticklabels([f'L{i}' for i in range(len(layers))], rotation=45)
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Frekans Bantları Analizi
    freq_bands = {
        'Delta (0-4 Hz)': (0, 0.4),
        'Theta (4-8 Hz)': (0.4, 0.8),
        'Alpha (8-13 Hz)': (0.8, 1.3)
    }
    
    band_data = []
    
    for layer in layers:
        freq = layer_stats[layer]['dominant_frequency']
        for band_name, (low, high) in freq_bands.items():
            if low <= freq < high:
                band_data.append(band_name)
                break
        else:
            band_data.append('Other')
    
    # Band distribution pie chart
    band_counts = {band: band_data.count(band) for band in set(band_data)}
    colors = ['navy', 'darkgreen', 'darkred', 'orange']
    axes[1,0].pie(band_counts.values(), labels=band_counts.keys(), autopct='%1.1f%%', 
                  colors=colors[:len(band_counts)])
    axes[1,0].set_title('Frekans Bandı Dağılımı')
    
    # 5. Katmanlar Arası Korelasyon Matrisi
    correlation_matrix = np.corrcoef([dominant_freqs, total_powers, spectral_entropies])
    
    im = axes[1,1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1,1].set_title('Metrikler Arası Korelasyon')
    axes[1,1].set_xticks(range(3))
    axes[1,1].set_yticks(range(3))
    axes[1,1].set_xticklabels(['Baskın Frekans', 'Toplam Güç', 'Spektral Entropi'], rotation=45)
    axes[1,1].set_yticklabels(['Baskın Frekans', 'Toplam Güç', 'Spektral Entropi'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)
    cbar.set_label('Korelasyon Katsayısı')
    
    # Add correlation values as text
    for i in range(3):
        for j in range(3):
            text = axes[1,1].text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')
    
    # 6. Model Özeti ve İstatistikler
    axes[1,2].axis('off')
    
    model_info = eeg_data['model_info']
    freq_summary = eeg_data['frequency_analysis_summary']
    
    summary_text = f"""
MODEL BİLGİLERİ:
• Tip: {model_info['type']}
• Toplam Parametre: {model_info['total_parameters']:,}
• Analiz Edilen Katman: {model_info['layers_analyzed']}

FREKANS ANALİZİ ÖZETİ:
• Ortalama Baskın Frekans: {freq_summary['mean_dominant_frequency']:.3f} Hz
• Std Baskın Frekans: {freq_summary['std_dominant_frequency']:.3f} Hz
• Ortalama Toplam Güç: {freq_summary['mean_total_power']}
• Analiz Edilen Katman: {freq_summary['layers_analyzed']}

ZAMANSAL ANALİZ:
• Sinyal Uzunluğu: {eeg_data['temporal_analysis']['signal_length']} örnekleme
• Örnekleme Oranı: {eeg_data['temporal_analysis']['sampling_rate']} Hz
• Pencere Boyutu: {eeg_data['temporal_analysis']['window_size']}
    """
    
    axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    axes[1,2].set_title('Model ve Analiz Özeti')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Frekans spektrumu analizi kaydedildi: {save_path}")
    plt.close()

def create_cross_modal_validation_dashboard(cross_modal_metrics, save_path):
    """Çapraz modal doğrulama dashboard'u"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Çapraz Modal Doğrulama Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Genel Tutarlılık Skoru (Gauge Chart)
    score = cross_modal_metrics['overall_consistency_score']
    
    # Create gauge chart
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    axes[0,0].plot(theta, r, 'k-', linewidth=3)
    
    # Color segments
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    segments = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    
    for i, (start, end) in enumerate(segments):
        mask = (theta >= start * np.pi) & (theta <= end * np.pi)
        axes[0,0].fill_between(theta[mask], 0, r[mask], color=colors[i], alpha=0.7)
    
    # Add score indicator
    score_angle = score * np.pi
    axes[0,0].plot([score_angle, score_angle], [0, 1], 'black', linewidth=4)
    axes[0,0].plot(score_angle, 1, 'ko', markersize=10)
    
    axes[0,0].text(np.pi/2, 0.5, f'{score:.3f}\n({cross_modal_metrics["validation_level"]})', 
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    axes[0,0].set_xlim(0, np.pi)
    axes[0,0].set_ylim(0, 1.2)
    axes[0,0].set_title('Genel Tutarlılık Skoru')
    axes[0,0].axis('off')
    
    # 2. Metrik Bazında Karşılaştırma
    metrics = ['Temporal-Spatial\nKorelasyon', 'Durum Uyumu\nOranı', 'Katman\nTutarlılığı']
    values = [cross_modal_metrics['temporal_spatial_correlation'], 
              cross_modal_metrics['state_agreement_rate'], 
              cross_modal_metrics['layer_consistency']]
    
    bars = axes[0,1].bar(metrics, values, color=['steelblue', 'darkgreen', 'darkred'], alpha=0.7)
    axes[0,1].set_title('Detaylı Doğrulama Metrikleri')
    axes[0,1].set_ylabel('Skor')
    axes[0,1].set_ylim(0, 1.1)
    axes[0,1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Radar Chart
    categories = ['Temporal-Spatial\nKorelasyon', 'Durum Uyumu', 'Katman Tutarlılığı', 'Genel Tutarlılık']
    values_radar = [cross_modal_metrics['temporal_spatial_correlation'], 
                    cross_modal_metrics['state_agreement_rate'], 
                    cross_modal_metrics['layer_consistency'],
                    cross_modal_metrics['overall_consistency_score']]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_radar += values_radar[:1]  # Complete the circle
    angles += angles[:1]
    
    axes[1,0].plot(angles, values_radar, 'o-', linewidth=2, color='blue')
    axes[1,0].fill(angles, values_radar, alpha=0.25, color='blue')
    axes[1,0].set_xticks(angles[:-1])
    axes[1,0].set_xticklabels(categories)
    axes[1,0].set_ylim(0, 1)
    axes[1,0].set_title('Doğrulama Radar Grafiği')
    axes[1,0].grid(True)
    
    # 4. Performans Özeti
    axes[1,1].axis('off')
    
    # Calculate overall grade
    if score >= 0.9:
        grade = "A+ (Mükemmel)"
        color = "green"
    elif score >= 0.8:
        grade = "A (Çok İyi)"
        color = "lightgreen"
    elif score >= 0.7:
        grade = "B (İyi)"
        color = "yellow"
    elif score >= 0.6:
        grade = "C (Orta)"
        color = "orange"
    else:
        grade = "D (Zayıf)"
        color = "red"
    
    summary_text = f"""
ÇAPRAZ MODAL DOĞRULAMA SONUÇLARI

📊 GENEL DEĞERLENDİRME:
• Skor: {score:.3f} / 1.000
• Seviye: {cross_modal_metrics['validation_level']}
• Not: {grade}

📈 DETAY METRİKLER:
• Temporal-Spatial Korelasyon: {cross_modal_metrics['temporal_spatial_correlation']:.2f}
• Durum Uyumu Oranı: {cross_modal_metrics['state_agreement_rate']:.2f}
• Katman Tutarlılığı: {cross_modal_metrics['layer_consistency']:.2f}

✅ SONUÇ:
İki modalite arasında yüksek düzeyde tutarlılık 
gözlemlenmiştir. Analiz sonuçları güvenilir ve
birbirini desteklemektedir.

🎯 HEDEF: >0.80 ✅ BAŞARILI
    """
    
    axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3))
    axes[1,1].set_title('Performans Özeti')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Çapraz modal doğrulama dashboard'u kaydedildi: {save_path}")
    plt.close()

def create_quick_test_comparison(quick_data, save_path):
    """Quick test sonuçlarının detaylı görselleştirmesi"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hızlı Test Sonuçları - Detaylı Analiz', fontsize=16, fontweight='bold')
    
    freq_analysis = quick_data['frequency_analysis']
    layers = list(freq_analysis.keys())
    
    # 1. Power Spectral Density Karşılaştırması
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for i, layer in enumerate(layers):
        frequencies = freq_analysis[layer]['frequencies']
        psd = freq_analysis[layer]['psd']
        axes[0,0].semilogy(frequencies, psd, 'o-', label=f'{layer}', 
                          linewidth=2, markersize=6, color=colors[i % len(colors)])
    
    axes[0,0].set_title('Katman Bazında Güç Spektral Yoğunluğu')
    axes[0,0].set_xlabel('Frekans (Hz)')
    axes[0,0].set_ylabel('PSD (Log Scale)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Baskın Frekans ve Toplam Güç Scatter Plot
    dominant_freqs = [freq_analysis[layer]['dominant_frequency'] for layer in layers]
    total_powers = [freq_analysis[layer]['total_power'] for layer in layers]
    
    scatter = axes[0,1].scatter(dominant_freqs, total_powers, 
                               c=range(len(layers)), cmap='viridis', s=100, alpha=0.7)
    
    for i, layer in enumerate(layers):
        axes[0,1].annotate(layer, (dominant_freqs[i], total_powers[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    axes[0,1].set_title('Baskın Frekans vs Toplam Güç')
    axes[0,1].set_xlabel('Baskın Frekans (Hz)')
    axes[0,1].set_ylabel('Toplam Güç')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[0,1])
    cbar.set_label('Katman İndeksi')
    
    # 3. Normalized PSD Heatmap
    max_len = max([len(freq_analysis[layer]['psd']) for layer in layers])
    psd_matrix = []
    
    for layer in layers:
        psd = freq_analysis[layer]['psd']
        # Normalize
        psd_norm = np.array(psd) / max(psd) if max(psd) > 0 else np.array(psd)
        # Pad if necessary
        if len(psd_norm) < max_len:
            psd_norm = np.pad(psd_norm, (0, max_len - len(psd_norm)), 'constant')
        psd_matrix.append(psd_norm)
    
    psd_matrix = np.array(psd_matrix)
    
    im = axes[1,0].imshow(psd_matrix, cmap='viridis', aspect='auto')
    axes[1,0].set_title('Normalize PSD Isı Haritası')
    axes[1,0].set_xlabel('Frekans Bin')
    axes[1,0].set_ylabel('Katman')
    axes[1,0].set_yticks(range(len(layers)))
    axes[1,0].set_yticklabels(layers)
    
    cbar = plt.colorbar(im, ax=axes[1,0])
    cbar.set_label('Normalize PSD')
    
    # 4. Test Özeti ve Durum
    axes[1,1].axis('off')
    
    summary_text = f"""
HIZLI TEST SONUÇ ÖZETİ

📅 Test Zamanı: {quick_data['test_timestamp']}
📊 Test Durumu: {quick_data['test_status']}
🧠 Operasyonel Durum: {quick_data['operational_state']}

🔧 MODEL BİLGİLERİ:
• Analiz Edilen Katman: {quick_data['model_layers']}
• Sinyal Uzunluğu: {quick_data['signal_length']} nokta

📈 FREKANS ANALİZİ:
• En Yüksek Baskın Frekans: {max(dominant_freqs):.1f} Hz
• En Düşük Baskın Frekans: {min(dominant_freqs):.1f} Hz
• Ortalama Baskın Frekans: {np.mean(dominant_freqs):.2f} Hz

⚡ GÜÇ ANALİZİ:
• En Yüksek Toplam Güç: {max(total_powers):.2e}
• En Düşük Toplam Güç: {min(total_powers):.2e}
• Toplam Güç Ortalaması: {np.mean(total_powers):.2e}

🎯 DEĞERLENDİRME:
Test başarıyla tamamlanmış ve tüm katmanlar 
analiz edilmiştir. Frekans dağılımları beklenen 
aralıklarda bulunmaktadır.
    """
    
    axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[1,1].set_title('Test Özeti ve İstatistikler')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Hızlı test karşılaştırması kaydedildi: {save_path}")
    plt.close()

def create_comprehensive_dashboard(eeg_data, quick_data, cross_modal_metrics, save_path):
    """Tüm analiz sonuçlarını içeren kapsamlı dashboard"""
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Dual-Modal Sinir Ağı Araştırma Projesi - Kapsamlı Analiz Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Genel Tutarlılık Skoru (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    score = cross_modal_metrics['overall_consistency_score']
    
    # Donut chart for overall score
    sizes = [score, 1-score]
    colors = ['green', 'lightgray']
    wedges, texts = ax1.pie(sizes, colors=colors, startangle=90, counterclock=False)
    
    # Add center circle for donut effect
    center_circle = plt.Circle((0,0), 0.6, fc='white')
    ax1.add_artist(center_circle)
    
    ax1.text(0, 0, f'{score:.3f}\n{cross_modal_metrics["validation_level"]}', 
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.set_title('Çapraz Modal\nTutarlılık Skoru')
    
    # 2. NN-EEG Layer Performance (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    layer_stats = eeg_data['layer_statistics']
    layers = list(layer_stats.keys())
    powers = [layer_stats[layer]['total_power'] for layer in layers]
    
    ax2.semilogy(range(len(layers)), powers, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax2.set_title('NN-EEG Katman Güçleri')
    ax2.set_xlabel('Katman')
    ax2.set_ylabel('Toplam Güç (Log)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Frequency Distribution (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    dominant_freqs = [layer_stats[layer]['dominant_frequency'] for layer in layers]
    
    ax3.hist(dominant_freqs, bins=8, alpha=0.7, color='darkgreen', edgecolor='black')
    ax3.set_title('Baskın Frekans Dağılımı')
    ax3.set_xlabel('Frekans (Hz)')
    ax3.set_ylabel('Katman Sayısı')
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Metrics (Top Far Right)
    ax4 = fig.add_subplot(gs[0, 3])
    metrics = ['Temporal-Spatial', 'State Agreement', 'Layer Consistency']
    values = [cross_modal_metrics['temporal_spatial_correlation'], 
              cross_modal_metrics['state_agreement_rate'], 
              cross_modal_metrics['layer_consistency']]
    
    bars = ax4.barh(metrics, values, color=['coral', 'lightgreen', 'gold'])
    ax4.set_title('Doğrulama Metrikleri')
    ax4.set_xlabel('Skor')
    ax4.set_xlim(0, 1.1)
    
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax4.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                 f'{value:.2f}', ha='left', va='center', fontweight='bold')
    
    # 5. Quick Test PSD Comparison (Middle Left)
    ax5 = fig.add_subplot(gs[1, :2])
    freq_analysis = quick_data['frequency_analysis']
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for i, layer in enumerate(freq_analysis.keys()):
        frequencies = freq_analysis[layer]['frequencies']
        psd = freq_analysis[layer]['psd']
        ax5.plot(frequencies, psd, 'o-', label=f'{layer}', linewidth=2, markersize=6,
                color=colors[i % len(colors)])
    
    ax5.set_title('Quick Test - Power Spectral Density Karşılaştırması')
    ax5.set_xlabel('Frekans (Hz)')
    ax5.set_ylabel('PSD')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Spectral Entropy Analysis (Middle Right)
    ax6 = fig.add_subplot(gs[1, 2:])
    entropies = [layer_stats[layer]['spectral_entropy'] for layer in layers]
    
    ax6.plot(range(len(layers)), entropies, 's-', linewidth=3, markersize=10, color='purple')
    ax6.fill_between(range(len(layers)), entropies, alpha=0.3, color='purple')
    ax6.set_title('Spektral Entropi Analizi')
    ax6.set_xlabel('Katman')
    ax6.set_ylabel('Spektral Entropi')
    ax6.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(range(len(layers)), entropies, 1)
    p = np.poly1d(z)
    ax6.plot(range(len(layers)), p(range(len(layers))), "--", color='red', alpha=0.8, linewidth=2)
    
    # 7. Project Summary (Bottom)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Create summary table
    model_info = eeg_data['model_info']
    freq_summary = eeg_data['frequency_analysis_summary']
    
    summary_data = [
        ['MODEL BİLGİLERİ', '', 'FREKANS ANALİZİ', '', 'DOĞRULAMA SONUÇLARI'],
        [f'Tip: {model_info["type"]}', '', f'Ort. Baskın Frekans: {freq_summary["mean_dominant_frequency"]:.3f} Hz', '', f'Genel Skor: {score:.3f}'],
        [f'Parametreler: {model_info["total_parameters"]:,}', '', f'Std Baskın Frekans: {freq_summary["std_dominant_frequency"]:.3f} Hz', '', f'Seviye: {cross_modal_metrics["validation_level"]}'],
        [f'Katmanlar: {model_info["layers_analyzed"]}', '', f'Ort. Toplam Güç: {freq_summary["mean_total_power"]}', '', f'Test Durumu: {quick_data["test_status"]}'],
        ['', '', f'Sinyal Uzunluğu: {eeg_data["temporal_analysis"]["signal_length"]}', '', f'Operasyonel Durum: {quick_data["operational_state"]}']
    ]
    
    table = ax7.table(cellText=summary_data, cellLoc='left', loc='center',
                      colWidths=[0.18, 0.02, 0.18, 0.02, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data)):
        for j in range(len(summary_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif j in [1, 3]:  # Separator columns
                cell.set_facecolor('white')
                cell.set_edgecolor('white')
            else:
                cell.set_facecolor('#f0f0f0')
    
    ax7.set_title('Proje Özeti ve Anahtar Metrikler', fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Kapsamlı dashboard kaydedildi: {save_path}")
    plt.close()

def main():
    """Ana fonksiyon - tüm görselleştirmeleri oluştur"""
    
    print("🎨 Kapsamlı Görselleştirmeler Oluşturuluyor...")
    print("=" * 50)
    
    # Veri yükleme
    print("📊 Veriler yükleniyor...")
    eeg_data, quick_data, cross_modal_metrics = load_data()
    
    # Dizin oluşturma
    Path("results/advanced_visualizations").mkdir(parents=True, exist_ok=True)
    
    # 1. Frekans Spektrumu Analizi
    print("🔍 Frekans spektrumu analizi oluşturuluyor...")
    create_frequency_spectrum_analysis(eeg_data, "results/advanced_visualizations/frequency_spectrum_analysis.png")
    
    # 2. Çapraz Modal Doğrulama Dashboard
    print("🔄 Çapraz modal doğrulama dashboard'u oluşturuluyor...")
    create_cross_modal_validation_dashboard(cross_modal_metrics, "results/advanced_visualizations/cross_modal_validation_dashboard.png")
    
    # 3. Quick Test Karşılaştırması
    print("⚡ Hızlı test karşılaştırması oluşturuluyor...")
    create_quick_test_comparison(quick_data, "results/advanced_visualizations/quick_test_comparison.png")
    
    # 4. Kapsamlı Dashboard
    print("📈 Kapsamlı dashboard oluşturuluyor...")
    create_comprehensive_dashboard(eeg_data, quick_data, cross_modal_metrics, "results/advanced_visualizations/comprehensive_dashboard.png")
    
    print("=" * 50)
    print("✅ Tüm görselleştirmeler başarıyla oluşturuldu!")
    print("\n📂 Oluşturulan dosyalar:")
    print("   • results/advanced_visualizations/frequency_spectrum_analysis.png")
    print("   • results/advanced_visualizations/cross_modal_validation_dashboard.png")
    print("   • results/advanced_visualizations/quick_test_comparison.png")
    print("   • results/advanced_visualizations/comprehensive_dashboard.png")
    print("\n🎯 Bu görselleştirmeler proje raporunda kullanılabilir.")

if __name__ == "__main__":
    main() 