"""
Utilidades para procesamiento de video
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Generator
from pathlib import Path
import logging


class VideoProcessor:
    """
    Procesador de videos con utilidades comunes.
    """

    def __init__(self, video_path: str):
        """
        Inicializa el procesador.

        Args:
            video_path: Ruta al video
        """
        self.logger = logging.getLogger(__name__)
        self.video_path = video_path

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        # Propiedades del video
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(
            f"Video cargado: {self.width}x{self.height} @ {self.fps}fps, "
            f"{self.total_frames} frames"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lee el siguiente frame.

        Returns:
            Tupla (success, frame)
        """
        return self.cap.read()

    def frame_generator(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Genera frames del video.

        Args:
            start_frame: Frame inicial
            end_frame: Frame final (None = hasta el final)
            step: Paso entre frames

        Yields:
            Tupla (frame_number, frame)
        """
        if end_frame is None:
            end_frame = self.total_frames

        # Ir al frame inicial
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_num = start_frame

        while frame_num < end_frame:
            ret, frame = self.cap.read()

            if not ret:
                break

            yield frame_num, frame

            # Saltar frames según step
            if step > 1:
                for _ in range(step - 1):
                    self.cap.read()

            frame_num += step

    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Obtiene un frame específico.

        Args:
            frame_number: Número de frame

        Returns:
            Frame o None si falla
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        return frame if ret else None

    def resize_frame(
        self,
        frame: np.ndarray,
        width: Optional[int] = None,
        height: Optional[int] = None,
        keep_aspect: bool = True
    ) -> np.ndarray:
        """
        Redimensiona un frame.

        Args:
            frame: Frame a redimensionar
            width: Nuevo ancho
            height: Nuevo alto
            keep_aspect: Mantener aspect ratio

        Returns:
            Frame redimensionado
        """
        h, w = frame.shape[:2]

        if width is None and height is None:
            return frame

        if keep_aspect:
            if width is not None:
                height = int(h * (width / w))
            elif height is not None:
                width = int(w * (height / h))

        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    def release(self):
        """Libera recursos del video."""
        if self.cap is not None:
            self.cap.release()

    def get_properties(self) -> dict:
        """
        Obtiene propiedades del video.

        Returns:
            Diccionario con propiedades
        """
        return {
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'total_frames': self.total_frames,
            'duration_seconds': self.total_frames / self.fps if self.fps > 0 else 0
        }


class VideoWriter:
    """
    Escritor de videos con configuración simplificada.
    """

    def __init__(
        self,
        output_path: str,
        fps: int,
        width: int,
        height: int,
        codec: str = 'mp4v'
    ):
        """
        Inicializa el escritor.

        Args:
            output_path: Ruta de salida
            fps: Frames por segundo
            width: Ancho del video
            height: Alto del video
            codec: Codec a usar
        """
        self.logger = logging.getLogger(__name__)
        self.output_path = output_path

        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Configurar writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not self.writer.isOpened():
            raise ValueError(f"No se pudo crear el video: {output_path}")

        self.logger.info(f"Video writer creado: {output_path}")

    def write(self, frame: np.ndarray):
        """
        Escribe un frame.

        Args:
            frame: Frame a escribir
        """
        self.writer.write(frame)

    def release(self):
        """Libera recursos."""
        if self.writer is not None:
            self.writer.release()
            self.logger.info(f"Video guardado: {self.output_path}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


def extract_frames(
    video_path: str,
    output_dir: str,
    prefix: str = "frame",
    num_frames: Optional[int] = None,
    step: int = 1
):
    """
    Extrae frames de un video y los guarda como imágenes.

    Args:
        video_path: Ruta al video
        output_dir: Directorio de salida
        prefix: Prefijo para nombres de archivo
        num_frames: Número de frames a extraer (None = todos)
        step: Paso entre frames
    """
    logger = logging.getLogger(__name__)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with VideoProcessor(video_path) as vp:
        frame_count = 0

        for frame_num, frame in vp.frame_generator(step=step):
            if num_frames and frame_count >= num_frames:
                break

            output_file = output_path / f"{prefix}_{frame_num:06d}.jpg"
            cv2.imwrite(str(output_file), frame)

            frame_count += 1

            if frame_count % 100 == 0:
                logger.info(f"Extraídos {frame_count} frames")

    logger.info(f"Extracción completada: {frame_count} frames en {output_dir}")


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)

    # Nota: reemplazar con una ruta real para probar
    # vp = VideoProcessor("path/to/video.mp4")
    # print(vp.get_properties())
    # vp.release()

    print("VideoProcessor definido correctamente")
