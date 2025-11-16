"""
Calculador de estadísticas del juego
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
import logging


class StatisticsCalculator:
    """
    Calcula estadísticas del partido a partir de datos de tracking.
    """

    def __init__(self):
        """Inicializa el calculador de estadísticas."""
        self.logger = logging.getLogger(__name__)
        self.reset()

    def reset(self):
        """Reinicia las estadísticas."""
        self.stats = {
            'total_frames': 0,
            'total_players': set(),
            'ball_possessions': [],
            'player_distances': defaultdict(float),
            'player_speeds': defaultdict(list),
            'zone_activity': defaultdict(int),
            'play_types': Counter()
        }

    def update(
        self,
        frame_data: Dict,
        frame_number: int
    ):
        """
        Actualiza estadísticas con datos de un nuevo frame.

        Args:
            frame_data: Datos de tracking del frame
            frame_number: Número de frame
        """
        self.stats['total_frames'] += 1

        # Actualizar jugadores únicos
        players = frame_data.get('players', np.empty((0, 5)))
        for track in players:
            player_id = int(track[4])
            self.stats['total_players'].add(player_id)

        # Calcular posesión del balón
        possession = self._calculate_ball_possession(frame_data)
        if possession is not None:
            self.stats['ball_possessions'].append({
                'frame': frame_number,
                'player_id': possession
            })

    def _calculate_ball_possession(
        self,
        frame_data: Dict,
        threshold: float = 100.0
    ) -> Optional[int]:
        """
        Determina qué jugador tiene posesión del balón.

        Args:
            frame_data: Datos del frame
            threshold: Distancia máxima para considerar posesión

        Returns:
            ID del jugador con posesión, o None
        """
        players = frame_data.get('players', np.empty((0, 5)))
        ball = frame_data.get('ball', np.empty((0, 5)))

        if len(players) == 0 or len(ball) == 0:
            return None

        # Centro del balón
        ball_center = np.array([
            (ball[0][0] + ball[0][2]) / 2,
            (ball[0][1] + ball[0][3]) / 2
        ])

        # Encontrar jugador más cercano
        min_dist = float('inf')
        closest_player = None

        for track in players:
            player_center = np.array([
                (track[0] + track[2]) / 2,
                (track[1] + track[3]) / 2
            ])

            dist = np.linalg.norm(player_center - ball_center)

            if dist < min_dist:
                min_dist = dist
                closest_player = int(track[4])

        if min_dist < threshold:
            return closest_player

        return None

    def calculate_player_statistics(
        self,
        tracking_history: List[Dict],
        player_id: int
    ) -> Dict:
        """
        Calcula estadísticas de un jugador específico.

        Args:
            tracking_history: Historial de tracking
            player_id: ID del jugador

        Returns:
            Diccionario con estadísticas del jugador
        """
        stats = {
            'player_id': player_id,
            'frames_detected': 0,
            'total_distance': 0.0,
            'average_speed': 0.0,
            'max_speed': 0.0,
            'positions': [],
            'time_with_ball': 0,
            'zones_visited': set()
        }

        prev_position = None

        for i, frame_data in enumerate(tracking_history):
            players = frame_data.get('players', np.empty((0, 5)))

            # Buscar jugador
            player_track = None
            for track in players:
                if int(track[4]) == player_id:
                    player_track = track
                    break

            if player_track is None:
                continue

            stats['frames_detected'] += 1

            # Posición actual
            curr_position = np.array([
                (player_track[0] + player_track[2]) / 2,
                (player_track[1] + player_track[3]) / 2
            ])

            stats['positions'].append(curr_position)

            # Calcular distancia recorrida
            if prev_position is not None:
                distance = np.linalg.norm(curr_position - prev_position)
                stats['total_distance'] += distance

            prev_position = curr_position

            # Verificar posesión del balón
            possession = self._calculate_ball_possession(frame_data)
            if possession == player_id:
                stats['time_with_ball'] += 1

        # Calcular estadísticas derivadas
        if stats['frames_detected'] > 0:
            # Velocidad promedio (asumiendo 30 fps)
            fps = 30
            time_seconds = stats['frames_detected'] / fps
            stats['average_speed'] = stats['total_distance'] / time_seconds if time_seconds > 0 else 0

            # Porcentaje de tiempo con balón
            stats['possession_percentage'] = (
                stats['time_with_ball'] / stats['frames_detected'] * 100
            )

        return stats

    def calculate_team_statistics(
        self,
        tracking_history: List[Dict]
    ) -> Dict:
        """
        Calcula estadísticas del equipo.

        Args:
            tracking_history: Historial de tracking

        Returns:
            Diccionario con estadísticas del equipo
        """
        stats = {
            'total_players': len(self.stats['total_players']),
            'average_players_per_frame': 0.0,
            'ball_movement_distance': 0.0,
            'average_ball_speed': 0.0,
            'total_frames': len(tracking_history)
        }

        # Calcular promedio de jugadores por frame
        player_counts = []
        prev_ball_pos = None
        ball_distances = []

        for frame_data in tracking_history:
            players = frame_data.get('players', np.empty((0, 5)))
            player_counts.append(len(players))

            # Movimiento del balón
            ball = frame_data.get('ball', np.empty((0, 5)))
            if len(ball) > 0:
                curr_ball_pos = np.array([
                    (ball[0][0] + ball[0][2]) / 2,
                    (ball[0][1] + ball[0][3]) / 2
                ])

                if prev_ball_pos is not None:
                    dist = np.linalg.norm(curr_ball_pos - prev_ball_pos)
                    ball_distances.append(dist)
                    stats['ball_movement_distance'] += dist

                prev_ball_pos = curr_ball_pos

        if player_counts:
            stats['average_players_per_frame'] = np.mean(player_counts)

        if ball_distances:
            stats['average_ball_speed'] = np.mean(ball_distances)

        return stats

    def calculate_play_statistics(
        self,
        play_classifications: List[Dict]
    ) -> Dict:
        """
        Calcula estadísticas de jugadas clasificadas.

        Args:
            play_classifications: Lista de clasificaciones de jugadas

        Returns:
            Diccionario con estadísticas de jugadas
        """
        stats = {
            'total_plays': len(play_classifications),
            'play_type_distribution': Counter(),
            'average_confidence': 0.0,
            'high_confidence_plays': 0
        }

        confidences = []

        for play in play_classifications:
            play_type = play.get('type', 'unknown')
            confidence = play.get('confidence', 0.0)

            stats['play_type_distribution'][play_type] += 1
            confidences.append(confidence)

            if confidence > 0.7:
                stats['high_confidence_plays'] += 1

        if confidences:
            stats['average_confidence'] = np.mean(confidences)

        # Porcentajes
        if stats['total_plays'] > 0:
            stats['play_type_percentages'] = {
                play_type: (count / stats['total_plays'] * 100)
                for play_type, count in stats['play_type_distribution'].items()
            }

        return stats

    def generate_report(
        self,
        tracking_history: List[Dict],
        play_classifications: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Genera un reporte completo de estadísticas.

        Args:
            tracking_history: Historial de tracking
            play_classifications: Clasificaciones de jugadas

        Returns:
            Reporte completo
        """
        report = {
            'summary': {},
            'team_stats': {},
            'player_stats': {},
            'play_stats': {}
        }

        # Estadísticas del equipo
        report['team_stats'] = self.calculate_team_statistics(tracking_history)

        # Estadísticas de jugadores
        unique_players = set()
        for frame_data in tracking_history:
            players = frame_data.get('players', np.empty((0, 5)))
            for track in players:
                unique_players.add(int(track[4]))

        for player_id in unique_players:
            report['player_stats'][player_id] = self.calculate_player_statistics(
                tracking_history,
                player_id
            )

        # Estadísticas de jugadas
        if play_classifications:
            report['play_stats'] = self.calculate_play_statistics(play_classifications)

        # Resumen
        report['summary'] = {
            'total_frames': len(tracking_history),
            'unique_players': len(unique_players),
            'duration_seconds': len(tracking_history) / 30,  # Asumiendo 30 fps
            'total_plays_classified': len(play_classifications) if play_classifications else 0
        }

        return report

    def format_report(self, report: Dict) -> str:
        """
        Formatea el reporte como texto legible.

        Args:
            report: Reporte generado

        Returns:
            String formateado
        """
        lines = []
        lines.append("=" * 60)
        lines.append("REPORTE DE ANÁLISIS DE BALONCESTO")
        lines.append("=" * 60)
        lines.append("")

        # Resumen
        lines.append("RESUMEN:")
        lines.append("-" * 60)
        summary = report.get('summary', {})
        for key, value in summary.items():
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
        lines.append("")

        # Estadísticas del equipo
        lines.append("ESTADÍSTICAS DEL EQUIPO:")
        lines.append("-" * 60)
        team_stats = report.get('team_stats', {})
        for key, value in team_stats.items():
            if isinstance(value, float):
                lines.append(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
        lines.append("")

        # Estadísticas de jugadas
        if 'play_stats' in report and report['play_stats']:
            lines.append("ESTADÍSTICAS DE JUGADAS:")
            lines.append("-" * 60)
            play_stats = report['play_stats']

            if 'play_type_distribution' in play_stats:
                lines.append("Distribución de tipos de jugadas:")
                for play_type, count in play_stats['play_type_distribution'].items():
                    percentage = play_stats.get('play_type_percentages', {}).get(play_type, 0)
                    lines.append(f"  {play_type}: {count} ({percentage:.1f}%)")

            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def get_statistics(self) -> Dict:
        """Retorna estadísticas actuales."""
        stats = self.stats.copy()
        stats['total_players'] = len(stats['total_players'])
        return stats


if __name__ == "__main__":
    # Ejemplo de uso
    calc = StatisticsCalculator()
    print("Statistics Calculator inicializado")
