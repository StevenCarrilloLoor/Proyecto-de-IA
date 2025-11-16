# Tests

Este directorio contiene los tests unitarios y de integración del proyecto.

## Ejecutar Tests

```bash
# Todos los tests
pytest tests/

# Con verbose
pytest -v tests/

# Con coverage
pytest --cov=src tests/

# Test específico
pytest tests/test_detector.py
```

## Estructura de Tests

```
tests/
├── test_detector.py          # Tests de detección
├── test_tracker.py           # Tests de tracking
├── test_classifier.py        # Tests de clasificación
├── test_visualizer.py        # Tests de visualización
└── test_integration.py       # Tests de integración
```

## Escribir Nuevos Tests

### Ejemplo de Test Unitario

```python
import pytest
from src.detection import BasketballDetector

def test_detector_initialization():
    detector = BasketballDetector()
    assert detector is not None
    assert detector.conf_threshold == 0.5

def test_detection_on_sample_image():
    detector = BasketballDetector()
    # Implementar test con imagen de ejemplo
    pass
```

### Ejemplo de Test de Integración

```python
def test_full_pipeline():
    from src.main import BasketballAnalyzer

    analyzer = BasketballAnalyzer()
    # Test del pipeline completo
    pass
```

## Convenciones

- Usar pytest fixtures para setup común
- Nombrar tests con prefijo `test_`
- Agrupar tests relacionados en clases
- Usar mocks para componentes externos

## Coverage

Mantener coverage > 70% en todos los módulos principales.
