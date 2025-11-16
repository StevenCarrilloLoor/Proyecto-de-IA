"""
Cargador de configuración
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Carga configuración desde archivo YAML.

    Args:
        config_path: Ruta al archivo de configuración

    Returns:
        Diccionario con configuración

    Raises:
        FileNotFoundError: Si el archivo no existe
        yaml.YAMLError: Si hay error al parsear YAML
    """
    logger = logging.getLogger(__name__)

    config_file = Path(config_path)

    if not config_file.exists():
        logger.error(f"Archivo de configuración no encontrado: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logger.info(f"Configuración cargada desde {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error al parsear configuración: {e}")
        raise


def save_config(config: Dict[str, Any], output_path: str):
    """
    Guarda configuración a archivo YAML.

    Args:
        config: Diccionario de configuración
        output_path: Ruta de salida
    """
    logger = logging.getLogger(__name__)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Configuración guardada en {output_path}")

    except Exception as e:
        logger.error(f"Error al guardar configuración: {e}")
        raise


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Combina dos configuraciones, con override prioritario.

    Args:
        base_config: Configuración base
        override_config: Configuración que sobrescribe

    Returns:
        Configuración combinada
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


if __name__ == "__main__":
    # Ejemplo de uso
    config = load_config("config/config.yaml")
    print("Configuración cargada:")
    print(yaml.dump(config, default_flow_style=False))
