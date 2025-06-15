# 🚀 Gelişmiş Kullanım Örnekleri

## Gerçek Zamanlı İzleme
```python
from src.nn_eeg import LiveMonitor
from src.nn_fmri import SpatialVisualizer

monitor = LiveMonitor(model=my_model)
visualizer = SpatialVisualizer()

for batch in data_stream:
    activations = monitor.process(batch)
    
    # NN-EEG Analizi
    freq_analysis = monitor.analyze_frequencies()
    
    # NN-fMRI Analizi
    spatial_map = visualizer.generate_map(activations)
    
    # Entegre Dashboard
    dashboard.update(
        temporal=freq_analysis,
        spatial=spatial_map
    )
```

## Özel Analiz Pipeline'ı
```python
class CustomAnalyzer:
    def __init__(self, model):
        self.eeg_analyzer = EEGAnalyzer(model)
        self.fmri_analyzer = fMRIAnalyzer(model)
        
    def full_analysis(self, inputs):
        # 1. Temporal analiz
        eeg_results = self.eeg_analyzer.process(inputs)
        
        # 2. Spatial analiz
        fmri_results = self.fmri_analyzer.process(inputs)
        
        # 3. Çapraz doğrulama
        consistency = self.check_consistency(eeg_results, fmri_results)
        
        return {
            'temporal': eeg_results,
            'spatial': fmri_results,
            'consistency': consistency
        }
``` 