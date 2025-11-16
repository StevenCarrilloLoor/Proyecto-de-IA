#!/usr/bin/env python3
"""
Script para entrenar el clasificador de jugadas
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from classification import PlayClassifier, PlayType
from utils import setup_logger


class PlayDataset(Dataset):
    """
    Dataset para entrenamiento del clasificador de jugadas.

    Nota: Este es un dataset de ejemplo. En producción, deberías
    cargar datos reales de jugadas etiquetadas.
    """

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 30,
        train: bool = True
    ):
        """
        Inicializa el dataset.

        Args:
            data_dir: Directorio con datos
            sequence_length: Longitud de secuencias
            train: Si es dataset de entrenamiento
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.train = train

        # Cargar datos (placeholder - implementar según tus datos)
        self.sequences = []
        self.labels = []

        # TODO: Implementar carga de datos reales
        # Por ahora, generamos datos sintéticos para demostración
        self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Genera datos sintéticos para demostración."""
        num_samples = 1000 if self.train else 200

        for _ in range(num_samples):
            # Generar secuencia sintética (56 features por frame)
            sequence = np.random.randn(self.sequence_length, 56).astype(np.float32)

            # Label aleatorio
            label = np.random.randint(0, len(PlayType))

            self.sequences.append(sequence)
            self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': torch.FloatTensor(self.sequences[idx]),
            'label': torch.LongTensor([self.labels[idx]])[0]
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Entrena una época."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Entrenando"):
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Evalúa el modelo."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluando"):
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Entrenar clasificador de jugadas"
    )

    parser.add_argument(
        '--data',
        required=True,
        help='Directorio con datos de entrenamiento'
    )

    parser.add_argument(
        '--output',
        default='data/models/play_classifier.pth',
        help='Ruta para guardar modelo'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Número de épocas'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamaño de batch'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )

    parser.add_argument(
        '--model-type',
        choices=['lstm', 'mlp'],
        default='lstm',
        help='Tipo de modelo'
    )

    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Dispositivo de entrenamiento'
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(level='INFO')

    # Device
    device = torch.device(
        args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    )
    logger.info(f"Usando dispositivo: {device}")

    # Crear datasets
    logger.info("Cargando datos...")
    train_dataset = PlayDataset(args.data, train=True)
    val_dataset = PlayDataset(args.data, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    logger.info(f"Dataset de entrenamiento: {len(train_dataset)} muestras")
    logger.info(f"Dataset de validación: {len(val_dataset)} muestras")

    # Crear modelo
    logger.info(f"Creando modelo: {args.model_type}")
    classifier = PlayClassifier(
        model_type=args.model_type,
        device=str(device)
    )

    model = classifier.model
    model.to(device)

    # Optimizer y loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Entrenamiento
    logger.info("Iniciando entrenamiento...")
    best_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nÉpoca {epoch+1}/{args.epochs}")
        print("-" * 40)

        # Entrenar
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # Evaluar
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        # Actualizar LR
        scheduler.step(val_loss)

        # Log
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Guardar mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), args.output)
            logger.info(f"Modelo guardado en {args.output} (Acc: {val_acc:.2f}%)")

    logger.info(f"Entrenamiento completado. Mejor accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
