# Guía de Inicio Rápido

## Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/StevenCarrilloLoor/Proyecto-de-IA.git
cd Proyecto-de-IA
```

### 2. Crear Entorno Virtual

```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# o
venv\Scripts\activate  # En Windows
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

## Uso Básico

### Opción 1: Análisis de Video Completo

```bash
python scripts/analyze_video.py \
    --input data/raw/tu_video.mp4 \
    --output data/processed/resultado.mp4 \
    --save-results results/
```

### Opción 2: Usar el Sistema en Python

```python
from src.main import BasketballAnalyzer

# Inicializar
analyzer = BasketballAnalyzer()

# Procesar video
results = analyzer.process_video(
    video_path="data/raw/partido.mp4",
    output_path="data/processed/analizado.mp4"
)

# Ver estadísticas
print(analyzer.stats_calc.format_report(results['statistics']))

# Guardar resultados
analyzer.save_results(results, "results/mi_analisis/")
```

### Opción 3: Jupyter Notebook

```bash
jupyter notebook notebooks/01_quickstart.ipynb
```

## Configuración

Edita `config/config.yaml` para personalizar:

```yaml
detection:
  model: "yolov8n.pt"  # Cambiar a yolov8m.pt para mejor precisión
  confidence_threshold: 0.5
  device: "cuda"  # O "cpu" si no tienes GPU

tracking:
  max_age: 30
  min_hits: 3

classification:
  sequence_length: 30
```

## Ejemplos

### Detectar y Visualizar

```python
from src.detection import BasketballDetector
import cv2

detector = BasketballDetector()
frame = cv2.imread("frame.jpg")

detections, annotated = detector.detect_frame(frame, return_annotated=True)

cv2.imshow("Detecciones", annotated)
cv2.waitKey(0)
```

### Generar Heatmap

```python
from src.visualization import HeatmapGenerator

heatmap_gen = HeatmapGenerator()

# Procesar video primero
results = analyzer.process_video("video.mp4")

# Generar heatmap
heatmap = heatmap_gen.generate_player_heatmap(results['tracking'])

# Visualizar
heatmap_gen.visualize_heatmap(
    heatmap,
    title="Actividad de Jugadores",
    save_path="heatmap.png"
)
```

### Estadísticas Personalizadas

```python
from src.visualization import StatisticsCalculator

stats_calc = StatisticsCalculator()

# Calcular estadísticas de un jugador
player_stats = stats_calc.calculate_player_statistics(
    tracking_history=results['tracking'],
    player_id=1
)

print(f"Distancia recorrida: {player_stats['total_distance']:.2f} píxeles")
print(f"Tiempo con balón: {player_stats['time_with_ball']} frames")
```

## Solución de Problemas

### Error: "No se pudo abrir el video"

- Verifica que la ruta al video sea correcta
- Asegúrate de que el formato sea compatible (MP4, AVI, MOV)
- Instala codecs adicionales: `pip install opencv-contrib-python`

### Error: "CUDA out of memory"

- Reduce el tamaño del batch
- Usa modelo más pequeño (yolov8n.pt en vez de yolov8x.pt)
- Cambia device a "cpu" en config.yaml

### Detección lenta

- Usa GPU si está disponible
- Reduce la resolución del video
- Usa modelo más pequeño (yolov8n.pt)
- Procesa solo cada N frames

## Siguiente Paso

- Lee la [documentación completa](README.md)
- Explora los [notebooks de ejemplo](notebooks/)
- Revisa la [guía de contribución](CONTRIBUTING.md)

## Soporte

Este es un proyecto académico. Para preguntas:
- Contacta al profesor: Enrique Vinicio Carrera
- Revisa la documentación en el repositorio
