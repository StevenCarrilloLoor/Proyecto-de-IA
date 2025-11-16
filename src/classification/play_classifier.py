"""
Clasificador de jugadas de baloncesto
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum
import logging


class PlayType(Enum):
    """Tipos de jugadas reconocidas."""
    TIRO_LIBRE = 0
    TRIPLE = 1
    CONTRAATAQUE = 2
    PICK_AND_ROLL = 3
    NORMAL = 4


class PlayClassifierNet(nn.Module):
    """
    Red neuronal para clasificación de jugadas.
    Utiliza características espaciotemporales extraídas de secuencias de frames.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [256, 128, 64],
        num_classes: int = 5,
        dropout: float = 0.3
    ):
        """
        Inicializa la red.

        Args:
            input_size: Tamaño de las características de entrada
            hidden_sizes: Tamaños de las capas ocultas
            num_classes: Número de clases a predecir
            dropout: Tasa de dropout
        """
        super(PlayClassifierNet, self).__init__()

        layers = []
        prev_size = input_size

        # Capas ocultas
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        # Capa de salida
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class LSTMPlayClassifier(nn.Module):
    """
    Clasificador basado en LSTM para secuencias temporales.
    Mejor para capturar dinámicas temporales de las jugadas.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3
    ):
        """
        Inicializa el clasificador LSTM.

        Args:
            input_size: Tamaño de características por frame
            hidden_size: Tamaño de la capa oculta LSTM
            num_layers: Número de capas LSTM
            num_classes: Número de clases
            dropout: Tasa de dropout
        """
        super(LSTMPlayClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor de forma (batch, sequence_length, input_size)

        Returns:
            Logits de forma (batch, num_classes)
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Usar último estado oculto
        out = self.fc(hidden[-1])

        return out


class PlayClassifier:
    """
    Clasificador de jugadas de baloncesto que procesa secuencias de tracking.
    """

    def __init__(
        self,
        model_type: str = "lstm",
        sequence_length: int = 30,
        device: str = "cuda",
        model_path: Optional[str] = None
    ):
        """
        Inicializa el clasificador.

        Args:
            model_type: Tipo de modelo ("lstm" o "mlp")
            sequence_length: Longitud de secuencia a analizar
            device: Dispositivo de cómputo
            model_path: Ruta a modelo pre-entrenado
        """
        self.logger = logging.getLogger(__name__)
        self.sequence_length = sequence_length
        self.model_type = model_type

        # Configurar device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Crear modelo
        self.input_size = self._calculate_input_size()

        if model_type == "lstm":
            self.model = LSTMPlayClassifier(
                input_size=self.input_size,
                hidden_size=128,
                num_layers=2,
                num_classes=len(PlayType)
            )
        else:
            self.model = PlayClassifierNet(
                input_size=self.input_size * sequence_length,
                num_classes=len(PlayType)
            )

        self.model.to(self.device)

        # Cargar modelo si existe
        if model_path:
            try:
                self.load_model(model_path)
                self.logger.info(f"Modelo cargado desde {model_path}")
            except Exception as e:
                self.logger.warning(f"No se pudo cargar modelo: {e}")
                self.logger.info("Usando modelo sin entrenar")

        self.model.eval()

    def _calculate_input_size(self) -> int:
        """
        Calcula el tamaño de entrada basado en las características.

        Features por frame:
        - Posiciones de jugadores (max 10 jugadores * 2 coords = 20)
        - Posición del balón (2)
        - Velocidades de jugadores (10 * 2 = 20)
        - Velocidad del balón (2)
        - Distancias entre jugadores y balón (10)
        - Área de actividad (1)
        - Dispersión de jugadores (1)
        Total: ~56 features
        """
        return 56

    def extract_features(
        self,
        tracking_sequence: List[Dict]
    ) -> np.ndarray:
        """
        Extrae características de una secuencia de tracking.

        Args:
            tracking_sequence: Lista de datos de tracking por frame

        Returns:
            Array de características de forma (sequence_length, input_size)
        """
        features = []

        for i, frame_data in enumerate(tracking_sequence):
            frame_features = []

            # Posiciones de jugadores (normalizar a 0-1)
            players = frame_data.get('players', np.empty((0, 5)))
            player_positions = []

            for j in range(10):  # Max 10 jugadores
                if j < len(players):
                    x1, y1, x2, y2, _ = players[j]
                    center_x = (x1 + x2) / 2 / 1920.0  # Normalizar
                    center_y = (y1 + y2) / 2 / 1080.0
                    player_positions.extend([center_x, center_y])
                else:
                    player_positions.extend([0.0, 0.0])

            frame_features.extend(player_positions)

            # Posición del balón
            ball = frame_data.get('ball', np.empty((0, 5)))
            if len(ball) > 0:
                x1, y1, x2, y2, _ = ball[0]
                ball_x = (x1 + x2) / 2 / 1920.0
                ball_y = (y1 + y2) / 2 / 1080.0
            else:
                ball_x, ball_y = 0.0, 0.0

            frame_features.extend([ball_x, ball_y])

            # Velocidades (diferencia con frame anterior)
            if i > 0:
                prev_players = tracking_sequence[i-1].get('players', np.empty((0, 5)))
                velocities = self._calculate_velocities(players, prev_players)
                frame_features.extend(velocities[:20])  # Max 10 jugadores * 2

                prev_ball = tracking_sequence[i-1].get('ball', np.empty((0, 5)))
                ball_vel = self._calculate_ball_velocity(ball, prev_ball)
                frame_features.extend(ball_vel)
            else:
                frame_features.extend([0.0] * 22)

            # Distancias jugador-balón
            if len(ball) > 0:
                distances = self._calculate_distances_to_ball(players, ball[0])
                frame_features.extend(distances[:10])
            else:
                frame_features.extend([0.0] * 10)

            # Métricas espaciales
            if len(players) > 0:
                area = self._calculate_activity_area(players)
                dispersion = self._calculate_dispersion(players)
            else:
                area, dispersion = 0.0, 0.0

            frame_features.extend([area, dispersion])

            features.append(frame_features[:self.input_size])

        return np.array(features)

    def _calculate_velocities(
        self,
        current: np.ndarray,
        previous: np.ndarray
    ) -> List[float]:
        """Calcula velocidades de jugadores."""
        velocities = []

        for i in range(min(10, len(current))):
            if i < len(previous):
                curr_x = (current[i][0] + current[i][2]) / 2
                curr_y = (current[i][1] + current[i][3]) / 2
                prev_x = (previous[i][0] + previous[i][2]) / 2
                prev_y = (previous[i][1] + previous[i][3]) / 2

                vx = (curr_x - prev_x) / 1920.0
                vy = (curr_y - prev_y) / 1080.0
                velocities.extend([vx, vy])
            else:
                velocities.extend([0.0, 0.0])

        while len(velocities) < 20:
            velocities.extend([0.0, 0.0])

        return velocities

    def _calculate_ball_velocity(
        self,
        current: np.ndarray,
        previous: np.ndarray
    ) -> List[float]:
        """Calcula velocidad del balón."""
        if len(current) > 0 and len(previous) > 0:
            curr_x = (current[0][0] + current[0][2]) / 2
            curr_y = (current[0][1] + current[0][3]) / 2
            prev_x = (previous[0][0] + previous[0][2]) / 2
            prev_y = (previous[0][1] + previous[0][3]) / 2

            vx = (curr_x - prev_x) / 1920.0
            vy = (curr_y - prev_y) / 1080.0
            return [vx, vy]

        return [0.0, 0.0]

    def _calculate_distances_to_ball(
        self,
        players: np.ndarray,
        ball: np.ndarray
    ) -> List[float]:
        """Calcula distancias de jugadores al balón."""
        if len(ball) == 0:
            return [0.0] * 10

        ball_x = (ball[0] + ball[2]) / 2
        ball_y = (ball[1] + ball[3]) / 2

        distances = []
        for i in range(min(10, len(players))):
            player_x = (players[i][0] + players[i][2]) / 2
            player_y = (players[i][1] + players[i][3]) / 2

            dist = np.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)
            dist_norm = dist / np.sqrt(1920**2 + 1080**2)
            distances.append(dist_norm)

        while len(distances) < 10:
            distances.append(0.0)

        return distances

    def _calculate_activity_area(self, players: np.ndarray) -> float:
        """Calcula área de actividad de jugadores."""
        if len(players) == 0:
            return 0.0

        centers = [(p[0] + p[2])/2, (p[1] + p[3])/2 for p in players]
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]

        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        return area / (1920.0 * 1080.0)

    def _calculate_dispersion(self, players: np.ndarray) -> float:
        """Calcula dispersión de jugadores."""
        if len(players) < 2:
            return 0.0

        centers = np.array([(p[0] + p[2])/2, (p[1] + p[3])/2 for p in players])
        mean_center = np.mean(centers, axis=0)
        dispersion = np.mean(np.linalg.norm(centers - mean_center, axis=1))

        return dispersion / np.sqrt(1920**2 + 1080**2)

    def classify(
        self,
        tracking_sequence: List[Dict]
    ) -> Tuple[PlayType, float]:
        """
        Clasifica una secuencia de tracking.

        Args:
            tracking_sequence: Secuencia de datos de tracking

        Returns:
            Tupla (tipo_de_jugada, confianza)
        """
        # Extraer características
        features = self.extract_features(tracking_sequence)

        # Convertir a tensor
        if self.model_type == "lstm":
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        else:
            x = torch.FloatTensor(features.flatten()).unsqueeze(0).to(self.device)

        # Inferencia
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)

        play_type = PlayType(pred_class.item())
        confidence_val = confidence.item()

        return play_type, confidence_val

    def load_model(self, path: str):
        """Carga un modelo desde disco."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def save_model(self, path: str):
        """Guarda el modelo a disco."""
        torch.save(self.model.state_dict(), path)


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)

    classifier = PlayClassifier(model_type="lstm")
    print(f"Clasificador inicializado con input_size={classifier.input_size}")
