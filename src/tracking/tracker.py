"""
Tracker principal para jugadores y balón en baloncesto
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict, deque

from .sort_tracker import SORTTracker


class BasketballTracker:
    """
    Tracker especializado para baloncesto que mantiene seguimiento de
    jugadores y el balón a través del tiempo.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        trajectory_length: int = 50
    ):
        """
        Inicializa el tracker de baloncesto.

        Args:
            max_age: Frames sin detección antes de eliminar track
            min_hits: Detecciones mínimas para confirmar track
            iou_threshold: Umbral de IoU para matching
            trajectory_length: Longitud del historial de trayectorias
        """
        self.logger = logging.getLogger(__name__)

        # Trackers separados para jugadores y balón
        self.player_tracker = SORTTracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )

        self.ball_tracker = SORTTracker(
            max_age=max_age // 2,  # Balón puede desaparecer más rápido
            min_hits=max(1, min_hits - 1),
            iou_threshold=iou_threshold
        )

        # Historial de trayectorias
        self.trajectory_length = trajectory_length
        self.player_trajectories = defaultdict(lambda: deque(maxlen=trajectory_length))
        self.ball_trajectories = defaultdict(lambda: deque(maxlen=trajectory_length))

        # Estadísticas
        self.stats = {
            'total_frames': 0,
            'players_tracked': set(),
            'ball_tracked_frames': 0
        }

    def update(
        self,
        detections: List[Dict],
        frame_number: int
    ) -> Dict[str, np.ndarray]:
        """
        Actualiza los tracks con nuevas detecciones.

        Args:
            detections: Lista de detecciones del detector
            frame_number: Número de frame actual

        Returns:
            Diccionario con tracks de jugadores y balón
        """
        self.stats['total_frames'] += 1

        # Separar detecciones de jugadores y balón
        player_dets = []
        ball_dets = []

        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']

            if det['type'] == 'player':
                player_dets.append([bbox[0], bbox[1], bbox[2], bbox[3], conf])
            elif det['type'] == 'ball':
                ball_dets.append([bbox[0], bbox[1], bbox[2], bbox[3], conf])

        # Convertir a numpy arrays
        player_dets = np.array(player_dets) if player_dets else np.empty((0, 5))
        ball_dets = np.array(ball_dets) if ball_dets else np.empty((0, 5))

        # Actualizar trackers
        player_tracks = self.player_tracker.update(player_dets)
        ball_tracks = self.ball_tracker.update(ball_dets)

        # Actualizar trayectorias
        self._update_trajectories(player_tracks, ball_tracks, frame_number)

        # Actualizar estadísticas
        for track in player_tracks:
            self.stats['players_tracked'].add(int(track[4]))

        if len(ball_tracks) > 0:
            self.stats['ball_tracked_frames'] += 1

        return {
            'players': player_tracks,
            'ball': ball_tracks,
            'player_trajectories': dict(self.player_trajectories),
            'ball_trajectories': dict(self.ball_trajectories)
        }

    def _update_trajectories(
        self,
        player_tracks: np.ndarray,
        ball_tracks: np.ndarray,
        frame_number: int
    ):
        """
        Actualiza el historial de trayectorias.

        Args:
            player_tracks: Tracks de jugadores [[x1,y1,x2,y2,id],...]
            ball_tracks: Tracks del balón [[x1,y1,x2,y2,id],...]
            frame_number: Número de frame
        """
        # Actualizar trayectorias de jugadores
        for track in player_tracks:
            x1, y1, x2, y2, track_id = track
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            self.player_trajectories[int(track_id)].append({
                'frame': frame_number,
                'center': (center_x, center_y),
                'bbox': (x1, y1, x2, y2)
            })

        # Actualizar trayectorias del balón
        for track in ball_tracks:
            x1, y1, x2, y2, track_id = track
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            self.ball_trajectories[int(track_id)].append({
                'frame': frame_number,
                'center': (center_x, center_y),
                'bbox': (x1, y1, x2, y2)
            })

    def get_trajectory(
        self,
        track_id: int,
        object_type: str = 'player'
    ) -> List[Dict]:
        """
        Obtiene la trayectoria de un objeto específico.

        Args:
            track_id: ID del track
            object_type: 'player' o 'ball'

        Returns:
            Lista de posiciones en el tiempo
        """
        if object_type == 'player':
            return list(self.player_trajectories.get(track_id, []))
        else:
            return list(self.ball_trajectories.get(track_id, []))

    def get_active_tracks(self) -> Dict:
        """
        Obtiene todos los tracks activos.

        Returns:
            Diccionario con IDs de tracks activos
        """
        return {
            'players': list(self.player_trajectories.keys()),
            'ball': list(self.ball_trajectories.keys())
        }

    def get_statistics(self) -> Dict:
        """
        Obtiene estadísticas del tracking.

        Returns:
            Diccionario con estadísticas
        """
        stats = self.stats.copy()
        stats['unique_players'] = len(stats['players_tracked'])
        stats['ball_tracking_rate'] = (
            stats['ball_tracked_frames'] / max(1, stats['total_frames'])
        )
        return stats

    def reset(self):
        """Reinicia el tracker."""
        self.player_tracker.reset()
        self.ball_tracker.reset()
        self.player_trajectories.clear()
        self.ball_trajectories.clear()
        self.stats = {
            'total_frames': 0,
            'players_tracked': set(),
            'ball_tracked_frames': 0
        }


class TrackVisualizer:
    """
    Clase auxiliar para visualizar tracks y trayectorias.
    """

    @staticmethod
    def draw_tracks(
        frame: np.ndarray,
        tracks_data: Dict,
        show_trajectory: bool = True,
        trajectory_length: int = 30
    ) -> np.ndarray:
        """
        Dibuja tracks y trayectorias en un frame.

        Args:
            frame: Frame original
            tracks_data: Datos de tracking
            show_trajectory: Si mostrar trayectorias
            trajectory_length: Longitud de trayectorias a mostrar

        Returns:
            Frame con visualización
        """
        import cv2

        output = frame.copy()

        # Dibujar jugadores
        player_tracks = tracks_data.get('players', np.empty((0, 5)))
        for track in player_tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)

            # Bounding box
            cv2.rectangle(
                output,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            # ID del jugador
            label = f"P{track_id}"
            cv2.putText(
                output,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Trayectoria
            if show_trajectory:
                trajectory = tracks_data.get('player_trajectories', {}).get(track_id, [])
                TrackVisualizer._draw_trajectory(
                    output, trajectory, (0, 255, 0), trajectory_length
                )

        # Dibujar balón
        ball_tracks = tracks_data.get('ball', np.empty((0, 5)))
        for track in ball_tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)

            # Bounding box
            cv2.rectangle(
                output,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                2
            )

            # Etiqueta
            cv2.putText(
                output,
                "BALL",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            # Trayectoria
            if show_trajectory:
                trajectory = tracks_data.get('ball_trajectories', {}).get(track_id, [])
                TrackVisualizer._draw_trajectory(
                    output, trajectory, (0, 0, 255), trajectory_length
                )

        return output

    @staticmethod
    def _draw_trajectory(
        frame: np.ndarray,
        trajectory: List[Dict],
        color: Tuple[int, int, int],
        max_length: int = 30
    ):
        """Dibuja una trayectoria en el frame."""
        import cv2

        if len(trajectory) < 2:
            return

        # Tomar últimos N puntos
        points = trajectory[-max_length:]

        for i in range(1, len(points)):
            pt1 = tuple(map(int, points[i-1]['center']))
            pt2 = tuple(map(int, points[i]['center']))

            # Línea con transparencia gradual
            alpha = i / len(points)
            thickness = max(1, int(3 * alpha))

            cv2.line(frame, pt1, pt2, color, thickness)


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)

    tracker = BasketballTracker()
    print("Basketball Tracker inicializado")
    print(f"Estadísticas: {tracker.get_statistics()}")
