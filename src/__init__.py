"""
Sistema de Análisis de Baloncesto mediante Visión por Computador

Módulos principales:
- detection: Detección de jugadores y balón usando YOLOv8
- tracking: Tracking multi-objeto con SORT
- classification: Clasificación de jugadas
- visualization: Visualización y estadísticas
- utils: Utilidades generales
"""

from .main import BasketballAnalyzer

__version__ = "1.0.0"
__author__ = "Steven Carrillo"

__all__ = ['BasketballAnalyzer']
