"""
Detector de jugadores y balón usando YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Dict, Tuple, Optional
import logging


class BasketballDetector:
    """
    Detector de jugadores y balón en videos de baloncesto usando YOLOv8.

    Attributes:
        model: Modelo YOLO cargado
        conf_threshold: Umbral de confianza para detecciones
        iou_threshold: Umbral de IoU para NMS
        device: Dispositivo de cómputo (cuda/cpu)
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda"
    ):
        """
        Inicializa el detector.

        Args:
            model_path: Ruta al modelo YOLO
            conf_threshold: Umbral de confianza (0-1)
            iou_threshold: Umbral de IoU para NMS
            device: 'cuda' o 'cpu'
        """
        self.logger = logging.getLogger(__name__)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Configurar dispositivo
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            self.logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            self.logger.info("Usando CPU")

        # Cargar modelo
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.logger.info(f"Modelo {model_path} cargado exitosamente")
        except Exception as e:
            self.logger.error(f"Error al cargar modelo: {e}")
            raise

        # Clases de interés (COCO dataset)
        self.PERSON_CLASS = 0
        self.SPORTS_BALL_CLASS = 32

        self.stats = {
            'total_detections': 0,
            'player_detections': 0,
            'ball_detections': 0
        }

    def detect_frame(
        self,
        frame: np.ndarray,
        return_annotated: bool = False
    ) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """
        Detecta jugadores y balón en un frame.

        Args:
            frame: Imagen en formato numpy array (BGR)
            return_annotated: Si True, retorna frame con anotaciones

        Returns:
            Tupla de (detecciones, frame_anotado)
            - detecciones: Lista de diccionarios con info de cada detección
            - frame_anotado: Frame con bounding boxes dibujados (opcional)
        """
        try:
            # Realizar inferencia
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=[self.PERSON_CLASS, self.SPORTS_BALL_CLASS],
                verbose=False
            )

            detections = []

            # Procesar resultados
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes

                for i, box in enumerate(boxes):
                    # Extraer información
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    # Determinar tipo de objeto
                    if cls == self.PERSON_CLASS:
                        obj_type = "player"
                        self.stats['player_detections'] += 1
                    elif cls == self.SPORTS_BALL_CLASS:
                        obj_type = "ball"
                        self.stats['ball_detections'] += 1
                    else:
                        continue

                    # Crear diccionario de detección
                    detection = {
                        'bbox': xyxy.tolist(),  # [x1, y1, x2, y2]
                        'confidence': conf,
                        'class': cls,
                        'type': obj_type,
                        'center': self._get_center(xyxy)
                    }

                    detections.append(detection)
                    self.stats['total_detections'] += 1

            # Anotar frame si se solicita
            annotated_frame = None
            if return_annotated:
                annotated_frame = self._annotate_frame(frame.copy(), detections)

            return detections, annotated_frame

        except Exception as e:
            self.logger.error(f"Error en detección: {e}")
            return [], frame if return_annotated else None

    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        save_detections: bool = True
    ) -> List[List[Dict]]:
        """
        Procesa un video completo y detecta jugadores y balón.

        Args:
            video_path: Ruta al video de entrada
            output_path: Ruta para guardar video anotado (opcional)
            save_detections: Si guardar las detecciones

        Returns:
            Lista de detecciones por frame
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            self.logger.error(f"No se pudo abrir el video: {video_path}")
            return []

        # Obtener propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(f"Procesando video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Preparar writer si se va a guardar
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_detections = []
        frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Detectar en el frame
                detections, annotated = self.detect_frame(
                    frame,
                    return_annotated=(output_path is not None)
                )

                all_detections.append(detections)

                # Guardar frame anotado
                if writer and annotated is not None:
                    writer.write(annotated)

                frame_count += 1

                if frame_count % 30 == 0:
                    self.logger.info(f"Procesados {frame_count}/{total_frames} frames")

        finally:
            cap.release()
            if writer:
                writer.release()

        self.logger.info(f"Detección completada. Total frames: {frame_count}")
        self.logger.info(f"Estadísticas: {self.stats}")

        return all_detections

    def _get_center(self, bbox: np.ndarray) -> Tuple[float, float]:
        """Calcula el centro de un bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """
        Dibuja bounding boxes y etiquetas en el frame.

        Args:
            frame: Frame original
            detections: Lista de detecciones

        Returns:
            Frame anotado
        """
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            obj_type = det['type']

            x1, y1, x2, y2 = map(int, bbox)

            # Color según tipo
            if obj_type == "player":
                color = (0, 255, 0)  # Verde para jugadores
                label = f"Player {conf:.2f}"
            else:  # ball
                color = (0, 0, 255)  # Rojo para balón
                label = f"Ball {conf:.2f}"

            # Dibujar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Dibujar etiqueta
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 4),
                (x1 + text_width, y1),
                color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

            # Dibujar centro
            center = det['center']
            cv2.circle(frame, (int(center[0]), int(center[1])), 3, color, -1)

        return frame

    def get_statistics(self) -> Dict:
        """Retorna estadísticas de detección."""
        return self.stats.copy()

    def reset_statistics(self):
        """Reinicia las estadísticas."""
        self.stats = {
            'total_detections': 0,
            'player_detections': 0,
            'ball_detections': 0
        }


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)

    detector = BasketballDetector(
        model_path="yolov8n.pt",
        conf_threshold=0.5,
        device="cuda"
    )

    print("Detector inicializado correctamente")
    print(f"Estadísticas: {detector.get_statistics()}")
