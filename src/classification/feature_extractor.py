"""
Extractor de características para clasificación de jugadas
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging


class FeatureExtractor:
    """
    Extrae características relevantes de secuencias de tracking
    para clasificación de jugadas.
    """

    def __init__(
        self,
        court_width: int = 1920,
        court_height: int = 1080
    ):
        """
        Inicializa el extractor.

        Args:
            court_width: Ancho de la cancha en píxeles
            court_height: Alto de la cancha en píxeles
        """
        self.logger = logging.getLogger(__name__)
        self.court_width = court_width
        self.court_height = court_height

    def extract_spatial_features(
        self,
        frame_data: Dict
    ) -> Dict[str, float]:
        """
        Extrae características espaciales de un frame.

        Args:
            frame_data: Datos de tracking de un frame

        Returns:
            Diccionario con características espaciales
        """
        features = {}

        players = frame_data.get('players', np.empty((0, 5)))
        ball = frame_data.get('ball', np.empty((0, 5)))

        # Número de jugadores
        features['num_players'] = len(players)

        if len(players) > 0:
            # Centros de jugadores
            player_centers = np.array([
                [(p[0] + p[2])/2, (p[1] + p[3])/2]
                for p in players
            ])

            # Centro de masa
            center_of_mass = np.mean(player_centers, axis=0)
            features['center_of_mass_x'] = center_of_mass[0] / self.court_width
            features['center_of_mass_y'] = center_of_mass[1] / self.court_height

            # Dispersión espacial
            features['spatial_dispersion'] = np.std(player_centers) / self.court_width

            # Área ocupada
            x_span = np.ptp(player_centers[:, 0])
            y_span = np.ptp(player_centers[:, 1])
            features['occupied_area'] = (x_span * y_span) / (self.court_width * self.court_height)

            # Densidad (jugadores por área)
            features['density'] = len(players) / max(features['occupied_area'], 0.01)

        else:
            features['center_of_mass_x'] = 0.0
            features['center_of_mass_y'] = 0.0
            features['spatial_dispersion'] = 0.0
            features['occupied_area'] = 0.0
            features['density'] = 0.0

        # Características del balón
        if len(ball) > 0:
            ball_center = [(ball[0][0] + ball[0][2])/2, (ball[0][1] + ball[0][3])/2]
            features['ball_x'] = ball_center[0] / self.court_width
            features['ball_y'] = ball_center[1] / self.court_height
            features['ball_height'] = (ball[0][3] - ball[0][1]) / self.court_height
        else:
            features['ball_x'] = 0.0
            features['ball_y'] = 0.0
            features['ball_height'] = 0.0

        return features

    def extract_temporal_features(
        self,
        sequence: List[Dict],
        window_size: int = 10
    ) -> Dict[str, float]:
        """
        Extrae características temporales de una secuencia.

        Args:
            sequence: Secuencia de datos de tracking
            window_size: Tamaño de ventana para análisis

        Returns:
            Diccionario con características temporales
        """
        features = {}

        if len(sequence) < 2:
            return self._empty_temporal_features()

        # Velocidades medias
        velocities = self._compute_velocities(sequence)
        features['mean_player_speed'] = np.mean(velocities['player_speeds'])
        features['max_player_speed'] = np.max(velocities['player_speeds']) if len(velocities['player_speeds']) > 0 else 0.0
        features['ball_speed'] = np.mean(velocities['ball_speeds'])

        # Aceleraciones
        accelerations = self._compute_accelerations(velocities)
        features['mean_acceleration'] = np.mean(accelerations)
        features['max_acceleration'] = np.max(accelerations) if len(accelerations) > 0 else 0.0

        # Cambios de dirección
        features['direction_changes'] = self._count_direction_changes(sequence)

        # Actividad temporal (variación en el tiempo)
        features['temporal_activity'] = self._compute_temporal_activity(sequence)

        return features

    def extract_interaction_features(
        self,
        frame_data: Dict
    ) -> Dict[str, float]:
        """
        Extrae características de interacción entre jugadores y balón.

        Args:
            frame_data: Datos de tracking

        Returns:
            Diccionario con características de interacción
        """
        features = {}

        players = frame_data.get('players', np.empty((0, 5)))
        ball = frame_data.get('ball', np.empty((0, 5)))

        if len(players) == 0 or len(ball) == 0:
            return self._empty_interaction_features()

        # Distancias al balón
        ball_center = np.array([(ball[0][0] + ball[0][2])/2, (ball[0][1] + ball[0][3])/2])
        player_centers = np.array([
            [(p[0] + p[2])/2, (p[1] + p[3])/2]
            for p in players
        ])

        distances = np.linalg.norm(player_centers - ball_center, axis=1)
        distances_norm = distances / np.sqrt(self.court_width**2 + self.court_height**2)

        features['min_distance_to_ball'] = np.min(distances_norm)
        features['mean_distance_to_ball'] = np.mean(distances_norm)
        features['std_distance_to_ball'] = np.std(distances_norm)

        # Jugadores cerca del balón
        threshold = 0.1  # 10% de la diagonal
        features['players_near_ball'] = np.sum(distances_norm < threshold)

        # Distancias entre jugadores
        if len(players) > 1:
            player_distances = []
            for i in range(len(player_centers)):
                for j in range(i+1, len(player_centers)):
                    dist = np.linalg.norm(player_centers[i] - player_centers[j])
                    player_distances.append(dist)

            player_distances = np.array(player_distances) / np.sqrt(self.court_width**2 + self.court_height**2)
            features['mean_player_distance'] = np.mean(player_distances)
            features['min_player_distance'] = np.min(player_distances)
        else:
            features['mean_player_distance'] = 0.0
            features['min_player_distance'] = 0.0

        return features

    def _compute_velocities(
        self,
        sequence: List[Dict]
    ) -> Dict[str, List[float]]:
        """Calcula velocidades de jugadores y balón."""
        player_speeds = []
        ball_speeds = []

        for i in range(1, len(sequence)):
            curr = sequence[i]
            prev = sequence[i-1]

            # Velocidades de jugadores
            curr_players = curr.get('players', np.empty((0, 5)))
            prev_players = prev.get('players', np.empty((0, 5)))

            for j in range(min(len(curr_players), len(prev_players))):
                curr_center = np.array([
                    (curr_players[j][0] + curr_players[j][2])/2,
                    (curr_players[j][1] + curr_players[j][3])/2
                ])
                prev_center = np.array([
                    (prev_players[j][0] + prev_players[j][2])/2,
                    (prev_players[j][1] + prev_players[j][3])/2
                ])

                speed = np.linalg.norm(curr_center - prev_center)
                player_speeds.append(speed / self.court_width)

            # Velocidad del balón
            curr_ball = curr.get('ball', np.empty((0, 5)))
            prev_ball = prev.get('ball', np.empty((0, 5)))

            if len(curr_ball) > 0 and len(prev_ball) > 0:
                curr_ball_center = np.array([
                    (curr_ball[0][0] + curr_ball[0][2])/2,
                    (curr_ball[0][1] + curr_ball[0][3])/2
                ])
                prev_ball_center = np.array([
                    (prev_ball[0][0] + prev_ball[0][2])/2,
                    (prev_ball[0][1] + prev_ball[0][3])/2
                ])

                ball_speed = np.linalg.norm(curr_ball_center - prev_ball_center)
                ball_speeds.append(ball_speed / self.court_width)

        return {
            'player_speeds': player_speeds,
            'ball_speeds': ball_speeds
        }

    def _compute_accelerations(
        self,
        velocities: Dict[str, List[float]]
    ) -> List[float]:
        """Calcula aceleraciones."""
        speeds = velocities['player_speeds']

        if len(speeds) < 2:
            return []

        accelerations = [abs(speeds[i] - speeds[i-1]) for i in range(1, len(speeds))]
        return accelerations

    def _count_direction_changes(
        self,
        sequence: List[Dict],
        threshold: float = 0.5
    ) -> int:
        """Cuenta cambios significativos de dirección."""
        if len(sequence) < 3:
            return 0

        changes = 0

        for i in range(2, len(sequence)):
            curr_players = sequence[i].get('players', np.empty((0, 5)))
            prev_players = sequence[i-1].get('players', np.empty((0, 5)))
            prev2_players = sequence[i-2].get('players', np.empty((0, 5)))

            for j in range(min(len(curr_players), len(prev_players), len(prev2_players))):
                # Vectores de movimiento
                v1 = np.array([
                    (prev_players[j][0] + prev_players[j][2])/2 - (prev2_players[j][0] + prev2_players[j][2])/2,
                    (prev_players[j][1] + prev_players[j][3])/2 - (prev2_players[j][1] + prev2_players[j][3])/2
                ])
                v2 = np.array([
                    (curr_players[j][0] + curr_players[j][2])/2 - (prev_players[j][0] + prev_players[j][2])/2,
                    (curr_players[j][1] + curr_players[j][3])/2 - (prev_players[j][1] + prev_players[j][3])/2
                ])

                # Producto punto normalizado
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    if cos_angle < threshold:
                        changes += 1

        return changes

    def _compute_temporal_activity(
        self,
        sequence: List[Dict]
    ) -> float:
        """Calcula actividad temporal (variación)."""
        if len(sequence) < 2:
            return 0.0

        activities = []

        for i in range(1, len(sequence)):
            curr_players = sequence[i].get('players', np.empty((0, 5)))
            prev_players = sequence[i-1].get('players', np.empty((0, 5)))

            if len(curr_players) > 0 and len(prev_players) > 0:
                # Calcular diferencia en número de detecciones
                diff = abs(len(curr_players) - len(prev_players))
                activities.append(diff)

        return np.mean(activities) if activities else 0.0

    def _empty_temporal_features(self) -> Dict[str, float]:
        """Retorna características temporales vacías."""
        return {
            'mean_player_speed': 0.0,
            'max_player_speed': 0.0,
            'ball_speed': 0.0,
            'mean_acceleration': 0.0,
            'max_acceleration': 0.0,
            'direction_changes': 0,
            'temporal_activity': 0.0
        }

    def _empty_interaction_features(self) -> Dict[str, float]:
        """Retorna características de interacción vacías."""
        return {
            'min_distance_to_ball': 0.0,
            'mean_distance_to_ball': 0.0,
            'std_distance_to_ball': 0.0,
            'players_near_ball': 0,
            'mean_player_distance': 0.0,
            'min_player_distance': 0.0
        }


if __name__ == "__main__":
    # Ejemplo de uso
    extractor = FeatureExtractor()
    print("Feature Extractor inicializado")
