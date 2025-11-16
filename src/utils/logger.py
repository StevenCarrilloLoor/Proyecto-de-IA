"""
Configuración de logging
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "basketball_analysis",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Configura el sistema de logging.

    Args:
        name: Nombre del logger
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
        log_file: Ruta al archivo de log (opcional)
        console: Si mostrar logs en consola

    Returns:
        Logger configurado
    """
    # Crear logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Evitar duplicar handlers
    if logger.handlers:
        return logger

    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler de consola
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Handler de archivo
    if log_file:
        # Crear directorio si no existe
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger existente.

    Args:
        name: Nombre del logger

    Returns:
        Logger
    """
    return logging.getLogger(name)


if __name__ == "__main__":
    # Ejemplo de uso
    logger = setup_logger(
        name="test_logger",
        level="DEBUG",
        log_file="logs/test.log"
    )

    logger.debug("Mensaje de debug")
    logger.info("Mensaje de info")
    logger.warning("Mensaje de warning")
    logger.error("Mensaje de error")
