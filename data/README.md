# Directorio de Datos

Este directorio contiene los datos necesarios para el proyecto.

## Estructura

```
data/
├── raw/              # Videos sin procesar
├── processed/        # Datos procesados y resultados
└── models/          # Modelos entrenados
```

## Cómo Obtener Datos

### Videos de Prueba

Puedes obtener videos de baloncesto de:

1. **YouTube**: Descargar videos de partidos públicos
   ```bash
   # Usar youtube-dl o yt-dlp
   yt-dlp "URL_DEL_VIDEO" -o "data/raw/game.mp4"
   ```

2. **Datasets Públicos**:
   - [SportVU Dataset](http://www.stats.com/sportvu-basketball/)
   - [NBA Stats](https://www.nba.com/stats/)
   - [Kaggle Basketball Datasets](https://www.kaggle.com/search?q=basketball+video)

3. **Videos Propios**: Grabar partidos locales con cámara convencional

### Modelos Pre-entrenados

Los modelos YOLOv8 se descargan automáticamente:
- `yolov8n.pt` - Nano (más rápido)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (más preciso)

### Modelo de Clasificación

El modelo de clasificación de jugadas debe ser entrenado:

```bash
python scripts/train_classifier.py --data data/training/ --output data/models/play_classifier.pth
```

## Requisitos de Formato

### Videos de Entrada

- Formato: MP4, AVI, MOV
- Resolución recomendada: 1920x1080 (Full HD)
- FPS: 25-30
- Vista: Preferiblemente vista lateral de la cancha completa

### Datos de Entrenamiento

Para entrenar el clasificador, necesitas:

1. **Secuencias de tracking anotadas**
   - Formato: PKL (pickle) o JSON
   - Estructura: Lista de frames con detecciones y tracks

2. **Etiquetas de jugadas**
   - Tipos: tiro_libre, triple, contraataque, pick_and_roll, normal
   - Formato: CSV o JSON

## Ejemplo de Uso

```python
from src.main import BasketballAnalyzer

# Procesar video
analyzer = BasketballAnalyzer()
results = analyzer.process_video(
    video_path="data/raw/game.mp4",
    output_path="data/processed/analyzed_game.mp4"
)

# Guardar resultados
analyzer.save_results(results, "data/processed/")
```

## Notas

- Los archivos de video no se incluyen en el repositorio por su tamaño
- Agregar videos a `.gitignore` para evitar subirlos a Git
- Mantener los modelos entrenados en `data/models/`
