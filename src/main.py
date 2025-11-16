"""
Sistema principal de análisis de baloncesto
Integra todos los módulos: detección, tracking, clasificación y visualización
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import logging
from tqdm import tqdm

from detection import BasketballDetector
from tracking import BasketballTracker
from classification import PlayClassifier
from visualization import Visualizer, HeatmapGenerator, StatisticsCalculator
from utils import load_config, setup_logger, VideoProcessor, VideoWriter


class BasketballAnalyzer:
    """
    Sistema completo de análisis de baloncesto que integra todos los componentes.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicializa el analizador.

        Args:
            config_path: Ruta al archivo de configuración
        """
        # Cargar configuración
        self.config = load_config(config_path)

        # Configurar logging
        log_config = self.config.get('logging', {})
        self.logger = setup_logger(
            level=log_config.get('level', 'INFO'),
            log_file=log_config.get('log_file') if log_config.get('save_logs') else None
        )

        self.logger.info("Inicializando Basketball Analyzer...")

        # Inicializar componentes
        self._init_detector()
        self._init_tracker()
        self._init_classifier()
        self._init_visualizer()

        self.logger.info("Basketball Analyzer inicializado correctamente")

    def _init_detector(self):
        """Inicializa el detector."""
        det_config = self.config.get('detection', {})

        self.detector = BasketballDetector(
            model_path=det_config.get('model', 'yolov8n.pt'),
            conf_threshold=det_config.get('confidence_threshold', 0.5),
            iou_threshold=det_config.get('iou_threshold', 0.45),
            device=det_config.get('device', 'cuda')
        )

        self.logger.info("Detector inicializado")

    def _init_tracker(self):
        """Inicializa el tracker."""
        track_config = self.config.get('tracking', {})

        self.tracker = BasketballTracker(
            max_age=track_config.get('max_age', 30),
            min_hits=track_config.get('min_hits', 3),
            iou_threshold=track_config.get('iou_threshold', 0.3),
            trajectory_length=50
        )

        self.logger.info("Tracker inicializado")

    def _init_classifier(self):
        """Inicializa el clasificador de jugadas."""
        class_config = self.config.get('classification', {})

        self.classifier = PlayClassifier(
            model_type="lstm",
            sequence_length=class_config.get('sequence_length', 30),
            device=self.config.get('detection', {}).get('device', 'cuda'),
            model_path=class_config.get('model_path')
        )

        self.logger.info("Clasificador inicializado")

    def _init_visualizer(self):
        """Inicializa componentes de visualización."""
        self.visualizer = Visualizer()
        self.heatmap_gen = HeatmapGenerator()
        self.stats_calc = StatisticsCalculator()

        self.logger.info("Componentes de visualización inicializados")

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        generate_heatmap: bool = True,
        classify_plays: bool = True,
        show_progress: bool = True
    ) -> Dict:
        """
        Procesa un video completo.

        Args:
            video_path: Ruta al video de entrada
            output_path: Ruta para video de salida (opcional)
            generate_heatmap: Generar mapas de calor
            classify_plays: Clasificar jugadas
            show_progress: Mostrar barra de progreso

        Returns:
            Diccionario con resultados del análisis
        """
        self.logger.info(f"Procesando video: {video_path}")

        # Validar entrada
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video no encontrado: {video_path}")

        # Abrir video
        with VideoProcessor(video_path) as vp:
            video_props = vp.get_properties()

            # Preparar writer si se requiere salida
            writer = None
            if output_path:
                writer = VideoWriter(
                    output_path,
                    video_props['fps'],
                    video_props['width'],
                    video_props['height']
                )

            # Almacenamiento de resultados
            all_detections = []
            all_tracking = []
            play_classifications = []

            # Secuencia para clasificación
            tracking_sequence = []

            # Procesar frames
            frame_iterator = vp.frame_generator()

            if show_progress:
                frame_iterator = tqdm(
                    frame_iterator,
                    total=video_props['total_frames'],
                    desc="Procesando"
                )

            for frame_num, frame in frame_iterator:
                # 1. Detección
                detections, _ = self.detector.detect_frame(frame)

                # 2. Tracking
                tracking_data = self.tracker.update(detections, frame_num)

                # 3. Almacenar
                all_detections.append(detections)
                all_tracking.append(tracking_data)

                # 4. Actualizar estadísticas
                self.stats_calc.update(tracking_data, frame_num)

                # 5. Clasificación de jugadas (cada N frames)
                if classify_plays:
                    tracking_sequence.append(tracking_data)

                    class_config = self.config.get('classification', {})
                    seq_len = class_config.get('sequence_length', 30)

                    if len(tracking_sequence) >= seq_len:
                        play_type, confidence = self.classifier.classify(
                            tracking_sequence[-seq_len:]
                        )

                        play_classifications.append({
                            'frame': frame_num,
                            'type': play_type.name,
                            'confidence': confidence
                        })

                # 6. Visualización
                if output_path:
                    # Dibujar tracking
                    vis_frame = self.visualizer.draw_tracking(
                        frame,
                        tracking_data,
                        show_trajectories=True
                    )

                    # Dibujar clasificación si disponible
                    if play_classifications and \
                       play_classifications[-1]['frame'] == frame_num:
                        last_play = play_classifications[-1]
                        vis_frame = self.visualizer.draw_play_classification(
                            vis_frame,
                            last_play['type'],
                            last_play['confidence']
                        )

                    # Agregar info
                    cv2.putText(
                        vis_frame,
                        f"Frame: {frame_num}/{video_props['total_frames']}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    writer.write(vis_frame)

            # Cerrar writer
            if writer:
                writer.release()

        # Generar resultados
        results = {
            'video_properties': video_props,
            'detections': all_detections,
            'tracking': all_tracking,
            'play_classifications': play_classifications,
            'statistics': self.stats_calc.generate_report(
                all_tracking,
                play_classifications
            )
        }

        # Generar heatmaps
        if generate_heatmap:
            self.logger.info("Generando heatmaps...")
            results['heatmaps'] = self._generate_heatmaps(all_tracking)

        self.logger.info("Procesamiento completado")

        return results

    def _generate_heatmaps(self, tracking_history: List[Dict]) -> Dict:
        """Genera heatmaps de actividad."""
        heatmaps = {}

        # Heatmap de jugadores
        heatmaps['players'] = self.heatmap_gen.generate_player_heatmap(
            tracking_history
        )

        # Heatmap del balón
        heatmaps['ball'] = self.heatmap_gen.generate_ball_heatmap(
            tracking_history
        )

        return heatmaps

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0
    ) -> Dict:
        """
        Procesa un solo frame.

        Args:
            frame: Frame a procesar
            frame_number: Número de frame

        Returns:
            Resultados del procesamiento
        """
        # Detección
        detections, _ = self.detector.detect_frame(frame)

        # Tracking
        tracking_data = self.tracker.update(detections, frame_number)

        return {
            'detections': detections,
            'tracking': tracking_data
        }

    def save_results(
        self,
        results: Dict,
        output_dir: str
    ):
        """
        Guarda resultados del análisis.

        Args:
            results: Resultados del análisis
            output_dir: Directorio de salida
        """
        import pickle

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Guardar resultados completos
        with open(output_path / 'results.pkl', 'wb') as f:
            pickle.dump(results, f)

        # Guardar reporte de estadísticas
        if 'statistics' in results:
            report_text = self.stats_calc.format_report(results['statistics'])
            with open(output_path / 'report.txt', 'w', encoding='utf-8') as f:
                f.write(report_text)

        # Guardar heatmaps
        if 'heatmaps' in results:
            for name, heatmap in results['heatmaps'].items():
                self.heatmap_gen.visualize_heatmap(
                    heatmap,
                    title=f"Heatmap - {name}",
                    save_path=str(output_path / f'heatmap_{name}.png')
                )

        self.logger.info(f"Resultados guardados en {output_dir}")

    def generate_highlights(
        self,
        video_path: str,
        results: Dict,
        output_path: str,
        confidence_threshold: float = 0.7
    ):
        """
        Genera un video con highlights de jugadas importantes.

        Args:
            video_path: Video original
            results: Resultados del análisis
            output_path: Ruta de salida
            confidence_threshold: Umbral de confianza para incluir jugada
        """
        play_classifications = results.get('play_classifications', [])

        # Filtrar jugadas de alta confianza
        highlights = [
            p for p in play_classifications
            if p['confidence'] >= confidence_threshold and p['type'] != 'NORMAL'
        ]

        if not highlights:
            self.logger.warning("No se encontraron highlights")
            return

        self.logger.info(f"Generando highlights con {len(highlights)} jugadas...")

        # Implementación básica - se puede mejorar
        # Por ahora solo registra las jugadas destacadas
        with open(output_path.replace('.mp4', '_highlights.txt'), 'w') as f:
            f.write("HIGHLIGHTS DEL PARTIDO\n")
            f.write("=" * 50 + "\n\n")

            for i, play in enumerate(highlights, 1):
                f.write(f"{i}. Frame {play['frame']}: ")
                f.write(f"{play['type']} (Confianza: {play['confidence']:.2%})\n")

        self.logger.info(f"Lista de highlights guardada")


def main():
    """Función principal de ejemplo."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sistema de Análisis de Baloncesto"
    )
    parser.add_argument(
        '--video',
        required=True,
        help='Ruta al video de entrada'
    )
    parser.add_argument(
        '--output',
        help='Ruta al video de salida'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Ruta al archivo de configuración'
    )
    parser.add_argument(
        '--save-results',
        help='Directorio para guardar resultados'
    )

    args = parser.parse_args()

    # Inicializar analizador
    analyzer = BasketballAnalyzer(config_path=args.config)

    # Procesar video
    results = analyzer.process_video(
        video_path=args.video,
        output_path=args.output
    )

    # Guardar resultados
    if args.save_results:
        analyzer.save_results(results, args.save_results)

    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
    print(analyzer.stats_calc.format_report(results['statistics']))


if __name__ == "__main__":
    main()
