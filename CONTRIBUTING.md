# Guía de Contribución

## Proyecto Académico

Este es un proyecto académico desarrollado para el curso de Inteligencia Artificial 1 en UDLA.

## Estructura del Proyecto

```
Proyecto-de-IA/
├── src/                    # Código fuente
│   ├── detection/         # Módulo de detección
│   ├── tracking/          # Módulo de tracking
│   ├── classification/    # Módulo de clasificación
│   ├── visualization/     # Módulo de visualización
│   └── utils/            # Utilidades
├── scripts/               # Scripts de análisis
├── notebooks/             # Jupyter notebooks
├── config/                # Archivos de configuración
├── data/                  # Datos y modelos
└── tests/                 # Tests unitarios

## Cómo Agregar Nuevas Funcionalidades

### 1. Agregar un Nuevo Tipo de Jugada

Para agregar un nuevo tipo de jugada al clasificador:

1. Editar `src/classification/play_classifier.py`
2. Agregar el nuevo tipo al enum `PlayType`
3. Actualizar el modelo para incluir la nueva clase
4. Reentrenar el clasificador con datos etiquetados

### 2. Mejorar la Detección

Para mejorar la detección:

1. Fine-tune YOLOv8 con datos de baloncesto específicos
2. Ajustar los umbrales en `config/config.yaml`
3. Implementar post-procesamiento adicional en `src/detection/detector.py`

### 3. Agregar Nuevas Métricas

Para agregar nuevas métricas de análisis:

1. Editar `src/visualization/statistics.py`
2. Implementar el cálculo de la nueva métrica
3. Actualizar el reporte de estadísticas

## Estándares de Código

- Seguir PEP 8 para estilo de Python
- Documentar todas las funciones con docstrings
- Agregar type hints cuando sea posible
- Escribir tests para nuevas funcionalidades

## Testing

```bash
# Ejecutar tests
pytest tests/

# Con coverage
pytest --cov=src tests/
```

## Contacto

**Estudiante:** Steven Carrillo
**Profesor:** Enrique Vinicio Carrera
**Curso:** Inteligencia Artificial 1 - UDLA
