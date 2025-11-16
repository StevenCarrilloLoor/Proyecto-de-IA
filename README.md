# Sistema de Detección y Análisis de Jugadas en Baloncesto

## Descripción
Sistema de análisis automático de partidos de baloncesto mediante técnicas de visión por computador e inteligencia artificial. El sistema detecta jugadores, sigue el balón, identifica jugadas clave y genera estadísticas relevantes a partir de videos.

## Características Principales
- 🏀 Detección de jugadores y balón usando YOLOv8
- 🎯 Tracking multi-objeto con DeepSORT
- 📊 Clasificación automática de jugadas (tiros libres, triples, contraataques)
- 📈 Generación de mapas de calor y estadísticas
- 🎥 Procesamiento de videos desde cámaras convencionales

## Estructura del Proyecto
```
Proyecto-de-IA/
├── src/
│   ├── detection/          # Módulo de detección YOLOv8
│   ├── tracking/           # Módulo de tracking
│   ├── classification/     # Clasificación de jugadas
│   ├── visualization/      # Visualización y estadísticas
│   └── utils/             # Utilidades generales
├── data/
│   ├── raw/               # Videos sin procesar
│   ├── processed/         # Datos procesados
│   └── models/            # Modelos entrenados
├── notebooks/             # Jupyter notebooks para experimentación
├── tests/                 # Tests unitarios
├── config/                # Archivos de configuración
└── requirements.txt       # Dependencias del proyecto
```

## Instalación

### Requisitos Previos
- Python 3.9 o superior
- CUDA 11.x (opcional, para GPU)
- pip

### Instalación de Dependencias
```bash
pip install -r requirements.txt
```

## Uso Rápido

### 1. Procesar un Video
```python
from src.main import BasketballAnalyzer

analyzer = BasketballAnalyzer()
results = analyzer.process_video("path/to/video.mp4")
```

### 2. Visualizar Resultados
```python
from src.visualization.visualizer import Visualizer

viz = Visualizer()
viz.generate_heatmap(results)
viz.plot_statistics(results)
```

### 3. Usando Scripts de Línea de Comandos
```bash
# Procesar video completo
python scripts/analyze_video.py --input data/raw/game.mp4 --output data/processed/

# Entrenar modelo de clasificación
python scripts/train_classifier.py --config config/classifier_config.yaml

# Evaluar modelo
python scripts/evaluate.py --model data/models/classifier.pth --test data/test/
```

## Módulos Principales

### Detección (YOLOv8)
Detecta jugadores y el balón en cada frame del video con alta precisión.

### Tracking (DeepSORT)
Mantiene la identidad de cada jugador a través del tiempo, manejando oclusiones.

### Clasificación de Jugadas
Identifica automáticamente:
- Tiros libres
- Triples
- Contraataques
- Pick and roll

### Visualización
Genera:
- Mapas de calor de posiciones
- Trayectorias de jugadores
- Estadísticas de rendimiento
- Dashboards interactivos

## Configuración

Edita el archivo `config/config.yaml` para personalizar:
- Modelos a utilizar
- Umbrales de detección
- Parámetros de tracking
- Opciones de visualización

## Entrenamiento de Modelos

Para entrenar el clasificador de jugadas:
```bash
python scripts/train_classifier.py --data data/training/ --epochs 50 --batch-size 32
```

## Evaluación

Métricas implementadas:
- Precisión de detección (mAP, IoU)
- Accuracy de clasificación (F1-score, matrices de confusión)
- FPS de procesamiento

## Ejemplos

Ver el directorio `notebooks/` para ejemplos detallados:
- `01_detection_demo.ipynb`: Demostración de detección
- `02_tracking_demo.ipynb`: Demostración de tracking
- `03_classification_demo.ipynb`: Clasificación de jugadas
- `04_full_pipeline.ipynb`: Pipeline completo

## Resultados Esperados

Según los objetivos del proyecto:
- Precisión de detección: >85%
- Precisión de tracking: >90%
- Accuracy de clasificación: >75%
- Velocidad de procesamiento: >10 FPS

## Contribución

Este proyecto fue desarrollado como parte del curso de Inteligencia Artificial 1 en UDLA.

**Integrante:** Steven Carrillo
**Profesor:** Enrique Vinicio Carrera
**Fecha:** Octubre 2025

## Licencia

Proyecto académico - UDLA 2025

## Referencias

- YOLOv8: https://github.com/ultralytics/ultralytics
- DeepSORT: https://github.com/nwojke/deep_sort
- OpenCV: https://opencv.org/
