"""
Visualizador principal para resultados de análisis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import logging


class Visualizer:
    """
    Clase principal para visualizar resultados del análisis de baloncesto.
    """

    def __init__(
        self,
        court_width: int = 1920,
        court_height: int = 1080
    ):
        """
        Inicializa el visualizador.

        Args:
            court_width: Ancho de la cancha
            court_height: Alto de la cancha
        """
        self.logger = logging.getLogger(__name__)
        self.court_width = court_width
        self.court_height = court_height

        # Configurar estilo de matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Dibuja detecciones en un frame.

        Args:
            frame: Frame original
            detections: Lista de detecciones
            show_confidence: Mostrar confianza

        Returns:
            Frame con detecciones dibujadas
        """
        output = frame.copy()

        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            obj_type = det['type']

            x1, y1, x2, y2 = map(int, bbox)

            # Color según tipo
            if obj_type == "player":
                color = (0, 255, 0)  # Verde
                label = f"Player"
            else:  # ball
                color = (0, 0, 255)  # Rojo
                label = f"Ball"

            if show_confidence:
                label += f" {conf:.2f}"

            # Dibujar bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Dibujar label
            self._draw_label(output, label, (x1, y1), color)

        return output

    def draw_tracking(
        self,
        frame: np.ndarray,
        tracking_data: Dict,
        show_trajectories: bool = True,
        trajectory_length: int = 30
    ) -> np.ndarray:
        """
        Dibuja resultados de tracking en un frame.

        Args:
            frame: Frame original
            tracking_data: Datos de tracking
            show_trajectories: Mostrar trayectorias
            trajectory_length: Longitud de trayectorias

        Returns:
            Frame con tracking dibujado
        """
        output = frame.copy()

        # Dibujar jugadores
        players = tracking_data.get('players', np.empty((0, 5)))
        for track in players:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)

            # Bbox
            cv2.rectangle(
                output,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            # ID
            label = f"P{track_id}"
            self._draw_label(output, label, (int(x1), int(y1)), (0, 255, 0))

            # Trayectoria
            if show_trajectories:
                trajectory = tracking_data.get('player_trajectories', {}).get(track_id, [])
                self._draw_trajectory(output, trajectory, (0, 255, 0), trajectory_length)

        # Dibujar balón
        ball = tracking_data.get('ball', np.empty((0, 5)))
        for track in ball:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)

            # Bbox
            cv2.rectangle(
                output,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                3
            )

            # Label
            self._draw_label(output, "BALL", (int(x1), int(y1)), (0, 0, 255))

            # Trayectoria
            if show_trajectories:
                trajectory = tracking_data.get('ball_trajectories', {}).get(track_id, [])
                self._draw_trajectory(output, trajectory, (0, 0, 255), trajectory_length)

        return output

    def draw_play_classification(
        self,
        frame: np.ndarray,
        play_type: str,
        confidence: float,
        position: Tuple[int, int] = (50, 50)
    ) -> np.ndarray:
        """
        Dibuja clasificación de jugada en el frame.

        Args:
            frame: Frame original
            play_type: Tipo de jugada
            confidence: Confianza de la predicción
            position: Posición del texto

        Returns:
            Frame con clasificación dibujada
        """
        output = frame.copy()

        # Fondo para el texto
        text = f"Jugada: {play_type}"
        conf_text = f"Confianza: {confidence:.2%}"

        # Dibujar recuadro
        cv2.rectangle(
            output,
            (position[0] - 10, position[1] - 40),
            (position[0] + 400, position[1] + 40),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            output,
            (position[0] - 10, position[1] - 40),
            (position[0] + 400, position[1] + 40),
            (255, 255, 255),
            2
        )

        # Texto
        cv2.putText(
            output,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            conf_text,
            (position[0], position[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if confidence > 0.7 else (0, 165, 255),
            2
        )

        return output

    def create_dashboard(
        self,
        frame: np.ndarray,
        stats: Dict,
        tracking_data: Dict
    ) -> np.ndarray:
        """
        Crea un dashboard con información del análisis.

        Args:
            frame: Frame del video
            stats: Estadísticas calculadas
            tracking_data: Datos de tracking

        Returns:
            Frame con dashboard
        """
        # Crear panel lateral
        dashboard_width = 400
        height = frame.shape[0]

        dashboard = np.zeros((height, dashboard_width, 3), dtype=np.uint8)
        dashboard[:] = (30, 30, 30)

        y_offset = 40

        # Título
        cv2.putText(
            dashboard,
            "ESTADISTICAS",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        y_offset += 50

        # Estadísticas
        stat_items = [
            f"Jugadores: {len(tracking_data.get('players', []))}",
            f"Frame: {stats.get('frame_number', 0)}",
            f"Detecciones: {stats.get('total_detections', 0)}",
        ]

        for item in stat_items:
            cv2.putText(
                dashboard,
                item,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )
            y_offset += 30

        # Combinar frame con dashboard
        output = np.hstack([frame, dashboard])

        return output

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int]
    ):
        """Dibuja una etiqueta con fondo."""
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        x, y = position

        # Fondo
        cv2.rectangle(
            frame,
            (x, y - text_height - baseline - 4),
            (x + text_width + 4, y),
            color,
            -1
        )

        # Texto
        cv2.putText(
            frame,
            text,
            (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    def _draw_trajectory(
        self,
        frame: np.ndarray,
        trajectory: List[Dict],
        color: Tuple[int, int, int],
        max_length: int = 30
    ):
        """Dibuja una trayectoria."""
        if len(trajectory) < 2:
            return

        points = trajectory[-max_length:]

        for i in range(1, len(points)):
            pt1 = tuple(map(int, points[i-1]['center']))
            pt2 = tuple(map(int, points[i]['center']))

            # Grosor con gradiente
            alpha = i / len(points)
            thickness = max(1, int(3 * alpha))

            cv2.line(frame, pt1, pt2, color, thickness)

    def plot_timeline(
        self,
        events: List[Dict],
        save_path: Optional[str] = None
    ):
        """
        Crea un gráfico de línea de tiempo de eventos.

        Args:
            events: Lista de eventos con 'frame' y 'type'
            save_path: Ruta para guardar
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        event_types = list(set(e['type'] for e in events))
        colors = plt.cm.tab10(np.linspace(0, 1, len(event_types)))

        for i, event_type in enumerate(event_types):
            event_frames = [e['frame'] for e in events if e['type'] == event_type]
            ax.scatter(
                event_frames,
                [i] * len(event_frames),
                c=[colors[i]],
                label=event_type,
                s=100,
                alpha=0.7
            )

        ax.set_yticks(range(len(event_types)))
        ax.set_yticklabels(event_types)
        ax.set_xlabel('Frame')
        ax.set_title('Línea de Tiempo de Jugadas')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_statistics_summary(
        self,
        stats: Dict,
        save_path: Optional[str] = None
    ):
        """
        Crea un resumen visual de estadísticas.

        Args:
            stats: Diccionario de estadísticas
            save_path: Ruta para guardar
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Placeholder - se puede expandir con estadísticas reales
        axes[0, 0].text(
            0.5, 0.5,
            'Estadísticas del Partido',
            ha='center',
            va='center',
            fontsize=16
        )
        axes[0, 0].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


if __name__ == "__main__":
    # Ejemplo de uso
    viz = Visualizer()
    print("Visualizer inicializado")
