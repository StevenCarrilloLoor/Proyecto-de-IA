"""
EJEMPLO DE USO COMPLETO DEL SISTEMA DE ANÁLISIS DE BALONCESTO
================================================================

Este script demuestra cómo usar el sistema paso a paso.
"""

import sys
import os

# Asegurarse de que src esté en el path
sys.path.insert(0, 'src')

print("="*70)
print("SISTEMA DE ANÁLISIS DE BALONCESTO - EJEMPLO DE USO")
print("="*70)
print()

# ============================================================================
# EJEMPLO 1: USO BÁSICO - Analizar un frame individual
# ============================================================================
print("EJEMPLO 1: Detectar jugadores y balón en una imagen")
print("-"*70)

from detection import BasketballDetector
import cv2
import numpy as np

# Crear detector
print("1. Inicializando detector YOLOv8...")
detector = BasketballDetector(
    model_path="yolov8n.pt",  # Modelo más liviano para demo
    conf_threshold=0.5,
    device="cpu"  # Cambia a "cuda" si tienes GPU
)
print("   ✓ Detector inicializado")

# Crear una imagen de prueba (negro con algunos rectángulos simulando jugadores)
print("\n2. Creando imagen de prueba...")
test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
# Dibujar algunos "jugadores" simulados
cv2.rectangle(test_frame, (300, 400), (400, 700), (0, 255, 0), -1)
cv2.rectangle(test_frame, (800, 450), (900, 750), (0, 255, 0), -1)
cv2.circle(test_frame, (600, 300), 30, (255, 0, 0), -1)  # "balón"
print("   ✓ Imagen de prueba creada (1920x1080)")

# Detectar (en imagen real detectaría automáticamente)
print("\n3. Ejecutando detección...")
detections, annotated_frame = detector.detect_frame(test_frame, return_annotated=True)
print(f"   ✓ Detectadas {len(detections)} objetos")

for i, det in enumerate(detections):
    print(f"   - Objeto {i+1}: {det['type']} (confianza: {det['confidence']:.2f})")

print("\n" + "="*70)

# ============================================================================
# EJEMPLO 2: TRACKING - Seguir jugadores en múltiples frames
# ============================================================================
print("\nEJEMPLO 2: Tracking de jugadores a través del tiempo")
print("-"*70)

from tracking import BasketballTracker

# Crear tracker
print("1. Inicializando tracker...")
tracker = BasketballTracker(
    max_age=30,      # Frames sin detección antes de eliminar
    min_hits=3,      # Detecciones mínimas para confirmar
    iou_threshold=0.3
)
print("   ✓ Tracker inicializado")

# Simular varios frames
print("\n2. Procesando 10 frames simulados...")
for frame_num in range(10):
    # Crear detecciones simuladas (en uso real vienen del detector)
    simulated_detections = [
        {
            'bbox': [300 + frame_num*10, 400, 400 + frame_num*10, 700],
            'confidence': 0.9,
            'type': 'player',
            'center': (350 + frame_num*10, 550)
        },
        {
            'bbox': [800, 450, 900, 750],
            'confidence': 0.85,
            'type': 'player',
            'center': (850, 600)
        }
    ]

    # Actualizar tracker
    tracking_data = tracker.update(simulated_detections, frame_num)

    players = tracking_data.get('players', [])
    print(f"   Frame {frame_num}: {len(players)} jugadores trackeados")

# Mostrar estadísticas de tracking
stats = tracker.get_statistics()
print(f"\n3. Estadísticas de tracking:")
print(f"   - Total frames procesados: {stats['total_frames']}")
print(f"   - Jugadores únicos: {stats['unique_players']}")

print("\n" + "="*70)

# ============================================================================
# EJEMPLO 3: CLASIFICACIÓN - Identificar tipos de jugadas
# ============================================================================
print("\nEJEMPLO 3: Clasificación de jugadas")
print("-"*70)

from classification import PlayClassifier
import torch

print("1. Inicializando clasificador LSTM...")
classifier = PlayClassifier(
    model_type="lstm",
    sequence_length=30,
    device="cpu"
)
print("   ✓ Clasificador inicializado")
print(f"   - Tipo de modelo: LSTM")
print(f"   - Tamaño de entrada: {classifier.input_size} features")

# Crear secuencia simulada de tracking
print("\n2. Creando secuencia de 30 frames para análisis...")
tracking_sequence = []
for i in range(30):
    frame_data = {
        'players': np.array([
            [300 + i*5, 400, 400 + i*5, 700, 1],  # Jugador 1 moviéndose
            [800, 450, 900, 750, 2],              # Jugador 2 estático
        ]),
        'ball': np.array([[500 + i*10, 300, 530 + i*10, 330, 1]])  # Balón moviéndose rápido
    }
    tracking_sequence.append(frame_data)

print("   ✓ Secuencia creada con 30 frames")

# Clasificar jugada
print("\n3. Clasificando tipo de jugada...")
play_type, confidence = classifier.classify(tracking_sequence)
print(f"   ✓ Jugada detectada: {play_type.name}")
print(f"   - Confianza: {confidence:.2%}")
print(f"\n   NOTA: El modelo no está entrenado, por lo que la predicción es aleatoria")
print(f"   Para obtener predicciones reales, entrena el modelo con:")
print(f"   → python scripts/train_classifier.py --data tu_dataset/")

print("\n" + "="*70)

# ============================================================================
# EJEMPLO 4: VISUALIZACIÓN - Generar heatmaps y estadísticas
# ============================================================================
print("\nEJEMPLO 4: Visualización y estadísticas")
print("-"*70)

from visualization import HeatmapGenerator, StatisticsCalculator

print("1. Generando heatmap de actividad...")
heatmap_gen = HeatmapGenerator(
    court_width=1920,
    court_height=1080
)

# Usar el historial de tracking del ejemplo anterior
heatmap = heatmap_gen.generate_player_heatmap(tracking_sequence)
print(f"   ✓ Heatmap generado (tamaño: {heatmap.shape})")

print("\n2. Calculando estadísticas del juego...")
stats_calc = StatisticsCalculator()

# Actualizar con cada frame
for i, frame_data in enumerate(tracking_sequence):
    stats_calc.update(frame_data, i)

# Generar reporte
report = stats_calc.generate_report(tracking_sequence)
print(f"   ✓ Reporte generado")
print(f"\n{stats_calc.format_report(report)}")

print("\n" + "="*70)

# ============================================================================
# EJEMPLO 5: SISTEMA COMPLETO - Pipeline integrado
# ============================================================================
print("\nEJEMPLO 5: Sistema completo (cómo usarías en tu video real)")
print("-"*70)

print("""
Para analizar un video real de baloncesto, usarías:

from src.main import BasketballAnalyzer

# 1. Inicializar el sistema completo
analyzer = BasketballAnalyzer(config_path='config/config.yaml')

# 2. Procesar tu video
results = analyzer.process_video(
    video_path='data/raw/mi_partido.mp4',
    output_path='data/processed/analizado.mp4',
    generate_heatmap=True,
    classify_plays=True,
    show_progress=True
)

# 3. Ver estadísticas
print(analyzer.stats_calc.format_report(results['statistics']))

# 4. Guardar resultados completos
analyzer.save_results(results, 'results/mi_analisis/')

El sistema automáticamente:
✓ Detecta jugadores y balón en cada frame
✓ Hace tracking de identidades
✓ Clasifica las jugadas que detecta
✓ Genera mapas de calor de actividad
✓ Calcula estadísticas del partido
✓ Exporta video anotado + reportes
""")

print("\n" + "="*70)

# ============================================================================
# EJEMPLO 6: USO CON LÍNEA DE COMANDOS
# ============================================================================
print("\nEJEMPLO 6: Uso desde la línea de comandos (la forma más fácil)")
print("-"*70)

print("""
Si solo quieres analizar un video rápidamente, usa los scripts:

# Análisis completo de un video
$ python scripts/analyze_video.py \\
    --input data/raw/partido.mp4 \\
    --output data/processed/resultado.mp4 \\
    --save-results results/mi_analisis/

# Entrenar el clasificador con tus datos
$ python scripts/train_classifier.py \\
    --data data/training/ \\
    --epochs 50 \\
    --batch-size 32

# Evaluar el sistema
$ python scripts/evaluate.py \\
    --video data/test/partido_test.mp4 \\
    --output results/evaluation/
""")

print("\n" + "="*70)
print("\n✓ EJEMPLOS COMPLETADOS")
print("\nPara ejecutar este script completo: python ejemplo_uso.py")
print("="*70)
