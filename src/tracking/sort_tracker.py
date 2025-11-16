"""
Implementación de SORT (Simple Online and Realtime Tracking)
Basado en el paper: https://arxiv.org/abs/1602.00763
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from typing import List, Tuple, Dict
import logging


class KalmanBoxTracker:
    """
    Tracker de Kalman Filter para un solo objeto usando coordenadas del bounding box.
    El estado es [x, y, s, r, dx, dy, ds] donde:
    - x, y: centro del bbox
    - s: área
    - r: aspect ratio
    - dx, dy, ds: velocidades
    """

    count = 0

    def __init__(self, bbox: np.ndarray):
        """
        Inicializa un tracker con un bounding box.

        Args:
            bbox: [x1, y1, x2, y2]
        """
        # Define el filtro de Kalman
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # Matriz de transición de estado
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        # Matriz de medición
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        # Covarianza de medición
        self.kf.R[2:, 2:] *= 10.0

        # Covarianza del proceso
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        # Covarianza del ruido del proceso
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Inicializar estado
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox: np.ndarray):
        """
        Actualiza el estado con una nueva medición.

        Args:
            bbox: [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        """
        Avanza el estado y retorna la predicción del bounding box.

        Returns:
            Predicción del bbox [x1, y1, x2, y2]
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))

        return self.history[-1]

    def get_state(self) -> np.ndarray:
        """
        Retorna el estado actual del bounding box.

        Returns:
            bbox [x1, y1, x2, y2]
        """
        return self._convert_x_to_bbox(self.kf.x)

    @staticmethod
    def _convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """
        Convierte bbox [x1, y1, x2, y2] a formato [x, y, s, r].

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            [x, y, s, r] donde x,y es el centro, s el área, r el aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h) if h != 0 else 1.0
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def _convert_x_to_bbox(x: np.ndarray, score: float = None) -> np.ndarray:
        """
        Convierte [x, y, s, r] a bbox [x1, y1, x2, y2].

        Args:
            x: Estado [x, y, s, r, ...]
            score: Score opcional

        Returns:
            bbox [x1, y1, x2, y2] o [x1, y1, x2, y2, score]
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w != 0 else 0
        if score is None:
            return np.array([
                x[0] - w / 2.0,
                x[1] - h / 2.0,
                x[0] + w / 2.0,
                x[1] + h / 2.0
            ]).reshape((1, 4))
        else:
            return np.array([
                x[0] - w / 2.0,
                x[1] - h / 2.0,
                x[0] + w / 2.0,
                x[1] + h / 2.0,
                score
            ]).reshape((1, 5))


class SORTTracker:
    """
    Tracker SORT (Simple Online and Realtime Tracking).
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Inicializa el tracker SORT.

        Args:
            max_age: Máximo número de frames sin detección antes de eliminar
            min_hits: Mínimo número de detecciones antes de confirmar track
            iou_threshold: Umbral de IoU para matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.logger = logging.getLogger(__name__)

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Actualiza los tracks con nuevas detecciones.

        Args:
            detections: Array de detecciones [[x1, y1, x2, y2, score], ...]

        Returns:
            Array de tracks [[x1, y1, x2, y2, id], ...]
        """
        self.frame_count += 1

        # Predecir nuevas localizaciones
        trks = np.zeros((len(self.trackers), 5))
        to_del = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # Eliminar tracks inválidos
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Matching usando Hungarian algorithm
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, trks, self.iou_threshold
        )

        # Actualizar trackers matched
        for m in matched:
            self.trackers[m[1]].update(detections[m[0], :4])

        # Crear nuevos trackers para detecciones no matched
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i, :4])
            self.trackers.append(trk)

        # Preparar output
        ret = []
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1

            # Eliminar tracks muertos
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    @staticmethod
    def _iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
        """
        Calcula IoU entre dos conjuntos de bboxes.

        Args:
            bb_test: Detecciones [[x1, y1, x2, y2], ...]
            bb_gt: Ground truth [[x1, y1, x2, y2], ...]

        Returns:
            Matriz de IoU
        """
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)

        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        intersection = w * h

        area_test = (bb_test[..., 2] - bb_test[..., 0]) * \
                    (bb_test[..., 3] - bb_test[..., 1])
        area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * \
                  (bb_gt[..., 3] - bb_gt[..., 1])

        union = area_test + area_gt - intersection

        iou = intersection / (union + 1e-6)

        return iou

    def _associate_detections_to_trackers(
        self,
        detections: np.ndarray,
        trackers: np.ndarray,
        iou_threshold: float = 0.3
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Asocia detecciones a trackers usando IoU.

        Args:
            detections: Detecciones [[x1, y1, x2, y2, score], ...]
            trackers: Trackers [[x1, y1, x2, y2, id], ...]
            iou_threshold: Umbral de IoU

        Returns:
            Tupla de (matches, unmatched_detections, unmatched_trackers)
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), \
                   list(range(len(detections))), []

        iou_matrix = self._iou_batch(detections[:, :4], trackers[:, :4])

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                # Hungarian algorithm
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_indices = np.stack([row_ind, col_ind], axis=1)
        else:
            matched_indices = np.empty(shape=(0, 2))

        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # Filtrar matches con bajo IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, unmatched_detections, unmatched_trackers

    def reset(self):
        """Reinicia el tracker."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0


if __name__ == "__main__":
    # Ejemplo de uso
    tracker = SORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)
    print("SORT Tracker inicializado")
