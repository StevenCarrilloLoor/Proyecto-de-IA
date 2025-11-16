"""
Generador de mapas de calor para análisis espacial
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import List, Dict, Optional, Tuple
import logging


class HeatmapGenerator:
    """
    Genera mapas de calor de posiciones de jugadores y balón.
    """

    def __init__(
        self,
        court_width: int = 1920,
        court_height: int = 1080,
        resolution: Tuple[int, int] = (192, 108)
    ):
        """
        Inicializa el generador.

        Args:
            court_width: Ancho de la cancha
            court_height: Alto de la cancha
            resolution: Resolución del heatmap
        """
        self.logger = logging.getLogger(__name__)
        self.court_width = court_width
        self.court_height = court_height
        self.resolution = resolution

    def generate_player_heatmap(
        self,
        tracking_history: List[Dict],
        player_id: Optional[int] = None,
        sigma: float = 5.0
    ) -> np.ndarray:
        """
        Genera mapa de calor de posiciones de jugadores.

        Args:
            tracking_history: Historial de tracking
            player_id: ID de jugador específico (None = todos)
            sigma: Desviación estándar para suavizado Gaussiano

        Returns:
            Mapa de calor normalizado
        """
        # Crear grid vacío
        heatmap = np.zeros(self.resolution[::-1], dtype=np.float32)

        # Acumular posiciones
        for frame_data in tracking_history:
            players = frame_data.get('players', np.empty((0, 5)))

            for track in players:
                x1, y1, x2, y2, tid = track
                tid = int(tid)

                # Filtrar por player_id si se especifica
                if player_id is not None and tid != player_id:
                    continue

                # Calcular centro
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Mapear a resolución del heatmap
                hmap_x = int(center_x / self.court_width * self.resolution[0])
                hmap_y = int(center_y / self.court_height * self.resolution[1])

                # Asegurar dentro de límites
                hmap_x = np.clip(hmap_x, 0, self.resolution[0] - 1)
                hmap_y = np.clip(hmap_y, 0, self.resolution[1] - 1)

                heatmap[hmap_y, hmap_x] += 1

        # Aplicar suavizado Gaussiano
        if sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=sigma)

        # Normalizar
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def generate_ball_heatmap(
        self,
        tracking_history: List[Dict],
        sigma: float = 3.0
    ) -> np.ndarray:
        """
        Genera mapa de calor de posiciones del balón.

        Args:
            tracking_history: Historial de tracking
            sigma: Desviación estándar para suavizado

        Returns:
            Mapa de calor del balón
        """
        heatmap = np.zeros(self.resolution[::-1], dtype=np.float32)

        for frame_data in tracking_history:
            ball = frame_data.get('ball', np.empty((0, 5)))

            if len(ball) > 0:
                x1, y1, x2, y2, _ = ball[0]

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                hmap_x = int(center_x / self.court_width * self.resolution[0])
                hmap_y = int(center_y / self.court_height * self.resolution[1])

                hmap_x = np.clip(hmap_x, 0, self.resolution[0] - 1)
                hmap_y = np.clip(hmap_y, 0, self.resolution[1] - 1)

                heatmap[hmap_y, hmap_x] += 1

        if sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=sigma)

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def overlay_heatmap_on_frame(
        self,
        frame: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.6,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Superpone mapa de calor sobre un frame.

        Args:
            frame: Frame original
            heatmap: Mapa de calor
            alpha: Transparencia del heatmap
            colormap: Mapa de colores de OpenCV

        Returns:
            Frame con heatmap superpuesto
        """
        # Redimensionar heatmap al tamaño del frame
        heatmap_resized = cv2.resize(
            heatmap,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

        # Convertir a 0-255
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)

        # Aplicar colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

        # Crear máscara para áreas con actividad
        mask = heatmap_resized > 0.1

        # Combinar
        output = frame.copy()
        output[mask] = cv2.addWeighted(
            frame[mask],
            1 - alpha,
            heatmap_colored[mask],
            alpha,
            0
        )

        return output

    def visualize_heatmap(
        self,
        heatmap: np.ndarray,
        title: str = "Heatmap",
        save_path: Optional[str] = None,
        cmap: str = 'hot'
    ):
        """
        Visualiza un mapa de calor con matplotlib.

        Args:
            heatmap: Mapa de calor
            title: Título del gráfico
            save_path: Ruta para guardar
            cmap: Colormap de matplotlib
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Mostrar heatmap
        im = ax.imshow(heatmap, cmap=cmap, aspect='auto', origin='upper')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensidad (normalizada)', rotation=270, labelpad=20)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Posición X')
        ax.set_ylabel('Posición Y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def generate_zone_heatmap(
        self,
        tracking_history: List[Dict],
        zones: List[Tuple[int, int, int, int]],
        zone_labels: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Calcula actividad por zonas de la cancha.

        Args:
            tracking_history: Historial de tracking
            zones: Lista de zonas [(x1, y1, x2, y2), ...]
            zone_labels: Etiquetas opcionales para zonas

        Returns:
            Diccionario con conteo por zona
        """
        if zone_labels is None:
            zone_labels = [f"Zona_{i}" for i in range(len(zones))]

        zone_counts = {label: 0 for label in zone_labels}

        for frame_data in tracking_history:
            players = frame_data.get('players', np.empty((0, 5)))

            for track in players:
                x1, y1, x2, y2, _ = track
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Verificar en qué zona está
                for zone, label in zip(zones, zone_labels):
                    zx1, zy1, zx2, zy2 = zone
                    if zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2:
                        zone_counts[label] += 1
                        break

        return zone_counts

    def visualize_zone_activity(
        self,
        zone_counts: Dict[str, int],
        save_path: Optional[str] = None
    ):
        """
        Visualiza actividad por zonas.

        Args:
            zone_counts: Conteo por zona
            save_path: Ruta para guardar
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        zones = list(zone_counts.keys())
        counts = list(zone_counts.values())

        bars = ax.bar(zones, counts, color='steelblue', alpha=0.7)

        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom'
            )

        ax.set_xlabel('Zona')
        ax.set_ylabel('Número de Detecciones')
        ax.set_title('Actividad por Zona', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def generate_temporal_heatmap(
        self,
        tracking_history: List[Dict],
        time_bins: int = 10
    ) -> np.ndarray:
        """
        Genera heatmap temporal (cambios a lo largo del tiempo).

        Args:
            tracking_history: Historial de tracking
            time_bins: Número de bins temporales

        Returns:
            Heatmap temporal de forma (time_bins, height, width)
        """
        frames_per_bin = max(1, len(tracking_history) // time_bins)

        temporal_heatmaps = []

        for i in range(time_bins):
            start_idx = i * frames_per_bin
            end_idx = min((i + 1) * frames_per_bin, len(tracking_history))

            bin_data = tracking_history[start_idx:end_idx]
            heatmap = self.generate_player_heatmap(bin_data)

            temporal_heatmaps.append(heatmap)

        return np.array(temporal_heatmaps)

    def visualize_temporal_heatmap(
        self,
        temporal_heatmap: np.ndarray,
        save_path: Optional[str] = None,
        cmap: str = 'hot'
    ):
        """
        Visualiza evolución temporal del heatmap.

        Args:
            temporal_heatmap: Heatmap temporal
            save_path: Ruta para guardar
            cmap: Colormap
        """
        n_bins = temporal_heatmap.shape[0]
        cols = min(5, n_bins)
        rows = (n_bins + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))

        if n_bins == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for i in range(n_bins):
            im = axes[i].imshow(temporal_heatmap[i], cmap=cmap, aspect='auto')
            axes[i].set_title(f'Período {i+1}')
            axes[i].axis('off')

        # Ocultar ejes sobrantes
        for i in range(n_bins, len(axes)):
            axes[i].axis('off')

        plt.suptitle('Evolución Temporal de Actividad', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


if __name__ == "__main__":
    # Ejemplo de uso
    generator = HeatmapGenerator()
    print("Heatmap Generator inicializado")
