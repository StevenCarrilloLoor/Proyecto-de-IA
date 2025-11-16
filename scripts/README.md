# Scripts

Scripts de línea de comandos para análisis y entrenamiento.

## Scripts Disponibles

### 1. analyze_video.py

Analiza un video de baloncesto completo.

```bash
python scripts/analyze_video.py \
    --input data/raw/game.mp4 \
    --output data/processed/analyzed.mp4 \
    --save-results results/
```

**Opciones:**
- `--input`: Video de entrada (requerido)
- `--output`: Video de salida con visualización
- `--config`: Archivo de configuración (default: config/config.yaml)
- `--save-results`: Directorio para guardar resultados
- `--no-heatmap`: No generar heatmaps
- `--no-classification`: No clasificar jugadas
- `--generate-highlights`: Generar video de highlights

### 2. train_classifier.py

Entrena el clasificador de jugadas.

```bash
python scripts/train_classifier.py \
    --data data/training/ \
    --output data/models/classifier.pth \
    --epochs 50 \
    --batch-size 32
```

**Opciones:**
- `--data`: Directorio con datos de entrenamiento (requerido)
- `--output`: Ruta para guardar modelo
- `--epochs`: Número de épocas
- `--batch-size`: Tamaño de batch
- `--lr`: Learning rate
- `--model-type`: Tipo de modelo (lstm o mlp)
- `--device`: cuda o cpu

### 3. evaluate.py

Evalúa el sistema en un video de prueba.

```bash
python scripts/evaluate.py \
    --video data/test/game.mp4 \
    --output results/evaluation/
```

**Opciones:**
- `--video`: Video de prueba (requerido)
- `--config`: Archivo de configuración
- `--output`: Directorio de salida

## Ejemplos de Uso

### Análisis Rápido

```bash
# Análisis básico sin guardar video
python scripts/analyze_video.py \
    --input data/raw/game.mp4 \
    --save-results results/quick_analysis/
```

### Análisis Completo

```bash
# Análisis completo con todo activado
python scripts/analyze_video.py \
    --input data/raw/game.mp4 \
    --output data/processed/full_analysis.mp4 \
    --save-results results/full_analysis/ \
    --generate-highlights
```

### Entrenamiento Personalizado

```bash
# Entrenar con parámetros personalizados
python scripts/train_classifier.py \
    --data data/training/ \
    --output data/models/custom_classifier.pth \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0001 \
    --model-type lstm
```

## Notas

- Los scripts requieren que las dependencias estén instaladas
- Verificar que config/config.yaml esté configurado correctamente
- Para GPU, asegurar que CUDA esté instalado
