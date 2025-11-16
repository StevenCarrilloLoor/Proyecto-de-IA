#!/usr/bin/env python3
"""
Script para evaluar el sistema de análisis
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from main import BasketballAnalyzer
from utils import setup_logger


def evaluate_detection(results: dict) -> dict:
    """
    Evalúa la calidad de las detecciones.

    Args:
        results: Resultados del análisis

    Returns:
        Métricas de detección
    """
    all_detections = results['detections']

    total_frames = len(all_detections)
    frames_with_players = 0
    frames_with_ball = 0
    total_player_detections = 0
    total_ball_detections = 0

    for frame_dets in all_detections:
        has_players = False
        has_ball = False

        for det in frame_dets:
            if det['type'] == 'player':
                total_player_detections += 1
                has_players = True
            elif det['type'] == 'ball':
                total_ball_detections += 1
                has_ball = True

        if has_players:
            frames_with_players += 1
        if has_ball:
            frames_with_ball += 1

    metrics = {
        'total_frames': total_frames,
        'frames_with_players': frames_with_players,
        'frames_with_ball': frames_with_ball,
        'player_detection_rate': frames_with_players / max(total_frames, 1),
        'ball_detection_rate': frames_with_ball / max(total_frames, 1),
        'avg_players_per_frame': total_player_detections / max(total_frames, 1),
        'total_player_detections': total_player_detections,
        'total_ball_detections': total_ball_detections
    }

    return metrics


def evaluate_tracking(results: dict) -> dict:
    """
    Evalúa la calidad del tracking.

    Args:
        results: Resultados del análisis

    Returns:
        Métricas de tracking
    """
    all_tracking = results['tracking']

    unique_player_ids = set()
    unique_ball_ids = set()
    track_lengths = {}

    for frame_data in all_tracking:
        players = frame_data.get('players', np.empty((0, 5)))
        ball = frame_data.get('ball', np.empty((0, 5)))

        for track in players:
            player_id = int(track[4])
            unique_player_ids.add(player_id)

            if player_id not in track_lengths:
                track_lengths[player_id] = 0
            track_lengths[player_id] += 1

        for track in ball:
            ball_id = int(track[4])
            unique_ball_ids.add(ball_id)

    avg_track_length = np.mean(list(track_lengths.values())) if track_lengths else 0

    metrics = {
        'unique_players_tracked': len(unique_player_ids),
        'unique_balls_tracked': len(unique_ball_ids),
        'average_track_length': avg_track_length,
        'max_track_length': max(track_lengths.values()) if track_lengths else 0,
        'min_track_length': min(track_lengths.values()) if track_lengths else 0
    }

    return metrics


def evaluate_classification(results: dict, output_dir: str = None):
    """
    Evalúa la clasificación de jugadas.

    Args:
        results: Resultados del análisis
        output_dir: Directorio para guardar visualizaciones
    """
    play_classifications = results.get('play_classifications', [])

    if not play_classifications:
        print("No hay clasificaciones de jugadas para evaluar")
        return {}

    # Distribución de tipos de jugadas
    play_types = [p['type'] for p in play_classifications]
    confidences = [p['confidence'] for p in play_classifications]

    from collections import Counter
    type_distribution = Counter(play_types)

    metrics = {
        'total_plays_classified': len(play_classifications),
        'play_type_distribution': dict(type_distribution),
        'average_confidence': np.mean(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
        'high_confidence_plays': sum(1 for c in confidences if c > 0.7)
    }

    # Visualizar distribución
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Gráfico de distribución
        plt.figure(figsize=(10, 6))
        plt.bar(type_distribution.keys(), type_distribution.values())
        plt.xlabel('Tipo de Jugada')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Tipos de Jugadas')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'play_distribution.png', dpi=300)
        plt.close()

        # Histograma de confianzas
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, edgecolor='black')
        plt.xlabel('Confianza')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Confianzas en Clasificación')
        plt.tight_layout()
        plt.savefig(output_path / 'confidence_distribution.png', dpi=300)
        plt.close()

    return metrics


def generate_evaluation_report(
    detection_metrics: dict,
    tracking_metrics: dict,
    classification_metrics: dict
) -> str:
    """Genera un reporte de evaluación."""
    lines = []
    lines.append("=" * 60)
    lines.append("REPORTE DE EVALUACIÓN DEL SISTEMA")
    lines.append("=" * 60)
    lines.append("")

    # Detección
    lines.append("MÉTRICAS DE DETECCIÓN:")
    lines.append("-" * 60)
    lines.append(f"Total de frames: {detection_metrics['total_frames']}")
    lines.append(f"Frames con jugadores: {detection_metrics['frames_with_players']}")
    lines.append(f"Frames con balón: {detection_metrics['frames_with_ball']}")
    lines.append(f"Tasa de detección de jugadores: {detection_metrics['player_detection_rate']:.2%}")
    lines.append(f"Tasa de detección de balón: {detection_metrics['ball_detection_rate']:.2%}")
    lines.append(f"Promedio de jugadores por frame: {detection_metrics['avg_players_per_frame']:.2f}")
    lines.append("")

    # Tracking
    lines.append("MÉTRICAS DE TRACKING:")
    lines.append("-" * 60)
    lines.append(f"Jugadores únicos rastreados: {tracking_metrics['unique_players_tracked']}")
    lines.append(f"Longitud promedio de tracks: {tracking_metrics['average_track_length']:.2f} frames")
    lines.append(f"Longitud máxima de track: {tracking_metrics['max_track_length']} frames")
    lines.append("")

    # Clasificación
    if classification_metrics:
        lines.append("MÉTRICAS DE CLASIFICACIÓN:")
        lines.append("-" * 60)
        lines.append(f"Total de jugadas clasificadas: {classification_metrics['total_plays_classified']}")
        lines.append(f"Confianza promedio: {classification_metrics['average_confidence']:.2%}")
        lines.append(f"Jugadas de alta confianza: {classification_metrics['high_confidence_plays']}")
        lines.append("")
        lines.append("Distribución de tipos de jugadas:")
        for play_type, count in classification_metrics['play_type_distribution'].items():
            percentage = count / classification_metrics['total_plays_classified'] * 100
            lines.append(f"  {play_type}: {count} ({percentage:.1f}%)")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluar sistema de análisis de baloncesto"
    )

    parser.add_argument(
        '--video',
        required=True,
        help='Video para evaluar'
    )

    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Archivo de configuración'
    )

    parser.add_argument(
        '--output',
        default='results/evaluation',
        help='Directorio de salida'
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logger(level='INFO')

    # Procesar video
    logger.info("Procesando video para evaluación...")
    analyzer = BasketballAnalyzer(config_path=args.config)

    results = analyzer.process_video(
        video_path=args.video,
        generate_heatmap=True,
        classify_plays=True,
        show_progress=True
    )

    # Evaluar componentes
    logger.info("Evaluando detección...")
    detection_metrics = evaluate_detection(results)

    logger.info("Evaluando tracking...")
    tracking_metrics = evaluate_tracking(results)

    logger.info("Evaluando clasificación...")
    classification_metrics = evaluate_classification(results, args.output)

    # Generar reporte
    report = generate_evaluation_report(
        detection_metrics,
        tracking_metrics,
        classification_metrics
    )

    print("\n" + report)

    # Guardar reporte
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / 'evaluation_report.txt', 'w') as f:
        f.write(report)

    logger.info(f"Reporte guardado en {output_path / 'evaluation_report.txt'}")


if __name__ == "__main__":
    main()
