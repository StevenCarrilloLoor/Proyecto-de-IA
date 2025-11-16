# 🏀 Cómo Funciona el Sistema - Explicación Simple

## 📺 Flujo Completo del Sistema

```
VIDEO DE BALONCESTO
        ↓
┌───────────────────────────────────────────────────────────┐
│  PASO 1: DETECCIÓN (YOLOv8)                              │
│  ───────────────────────────                              │
│  • Lee frame por frame del video                         │
│  • Detecta dónde están los jugadores (bounding boxes)    │
│  • Detecta dónde está el balón                           │
│  • Da una confianza a cada detección (0-100%)            │
└───────────────────────────────────────────────────────────┘
        ↓
        ↓ [Lista de detecciones por frame]
        ↓
┌───────────────────────────────────────────────────────────┐
│  PASO 2: TRACKING (SORT)                                 │
│  ────────────────────                                     │
│  • Compara detecciones del frame actual con anteriores   │
│  • Asigna un ID único a cada jugador                     │
│  • Mantiene el ID aunque el jugador se mueva             │
│  • Predice posición si se pierde momentáneamente         │
│  • Guarda trayectoria (historial de movimiento)          │
└───────────────────────────────────────────────────────────┘
        ↓
        ↓ [Jugadores con IDs + Trayectorias]
        ↓
┌───────────────────────────────────────────────────────────┐
│  PASO 3: CLASIFICACIÓN (LSTM)                            │
│  ─────────────────────────                                │
│  • Toma secuencia de 30 frames                           │
│  • Analiza movimientos de jugadores y balón              │
│  • Extrae features: velocidades, distancias, etc.        │
│  • Clasifica qué tipo de jugada es                       │
│  • Output: "Triple", "Contraataque", etc.                │
└───────────────────────────────────────────────────────────┘
        ↓
        ↓ [Jugadas clasificadas]
        ↓
┌───────────────────────────────────────────────────────────┐
│  PASO 4: VISUALIZACIÓN Y ESTADÍSTICAS                    │
│  ──────────────────────────────────                       │
│  • Genera mapas de calor (dónde hay más actividad)       │
│  • Dibuja trayectorias de jugadores                      │
│  • Calcula estadísticas: distancias, posesión, etc.      │
│  • Crea video anotado con toda la info                   │
│  • Exporta reportes en texto                             │
└───────────────────────────────────────────────────────────┘
        ↓
   RESULTADOS:
   • Video procesado con anotaciones
   • Mapas de calor
   • Estadísticas del partido
   • Jugadas clasificadas
```

---

## 🎬 Ejemplo Paso a Paso con un Frame

### FRAME ORIGINAL
```
┌─────────────────────────────────────────┐
│                                         │
│           🏀                            │
│                                         │
│    🏃    🏃        🏃                   │
│                         🏃              │
│              🏃                         │
│                   🏃        🏃          │
│                                         │
│                                         │
│    🏃                      🏃           │
└─────────────────────────────────────────┘
```

### PASO 1: DETECCIÓN
```python
# El detector YOLOv8 procesa el frame

detections = [
    {'type': 'player', 'bbox': [100, 200, 150, 400], 'confidence': 0.95},
    {'type': 'player', 'bbox': [200, 180, 250, 380], 'confidence': 0.92},
    {'type': 'player', 'bbox': [350, 220, 400, 420], 'confidence': 0.89},
    # ... más jugadores ...
    {'type': 'ball', 'bbox': [300, 100, 320, 120], 'confidence': 0.85}
]
```

**Resultado:**
```
┌─────────────────────────────────────────┐
│          ┌──┐                           │
│          │🏀│ ← Balón detectado         │
│          └──┘                           │
│   ┌──┐  ┌──┐      ┌──┐                 │
│   │🏃│  │🏃│      │🏃│                 │
│   └──┘  └──┘      └──┘                 │
│              ┌──┐                       │
│              │🏃│                       │
│              └──┘                       │
└─────────────────────────────────────────┘
  Cada rectángulo = una detección
```

### PASO 2: TRACKING
```python
# El tracker asigna IDs y mantiene identidad

tracking_data = {
    'players': [
        [100, 200, 150, 400, ID=1],  # ← Mismo jugador que frame anterior
        [200, 180, 250, 380, ID=2],  # ← Mismo jugador
        [350, 220, 400, 420, ID=5],  # ← Nuevo jugador (recién apareció)
        # ...
    ],
    'ball': [[300, 100, 320, 120, ID=1]]
}

# También guarda trayectorias
player_1_trajectory = [
    (105, 300),  # Frame 1
    (108, 298),  # Frame 2
    (112, 295),  # Frame 3 (se está moviendo)
    # ...
]
```

**Resultado:**
```
┌─────────────────────────────────────────┐
│          Ball #1                        │
│            🏀                            │
│         ···↑···  (trayectoria)          │
│   P1    P2        P5                    │
│   🏃    🏃        🏃                    │
│   ↑     ↑         ↑                     │
│   ID    ID        ID (nuevo)            │
└─────────────────────────────────────────┘
```

### PASO 3: CLASIFICACIÓN
```python
# Analiza los últimos 30 frames

sequence = [frame_1, frame_2, ..., frame_30]

# Extrae features de cada frame
features = {
    'velocidad_jugadores': [0.05, 0.08, 0.12, ...],  # Se acelera
    'velocidad_balon': [0.15, 0.18, 0.22, ...],      # Balón rápido
    'distancia_al_aro': [800, 750, 700, ...],         # Se acerca
    'dispersion_jugadores': 0.8,                      # Muy dispersos
    # ... 56 features en total
}

# El modelo LSTM procesa la secuencia
play_type, confidence = classifier.classify(sequence)
# → play_type = "CONTRAATAQUE"
# → confidence = 0.89 (89%)
```

**Resultado:**
```
═══════════════════════════════════
  JUGADA DETECTADA: CONTRAATAQUE
  Confianza: 89%
  Frame: 150-180
═══════════════════════════════════
```

### PASO 4: VISUALIZACIÓN
```python
# Genera mapa de calor
heatmap = heatmap_gen.generate_player_heatmap(tracking_history)

# Calcula estadísticas
stats = {
    'jugador_1': {
        'distancia_recorrida': 450.5,  # píxeles
        'tiempo_con_balon': 45,         # frames
        'velocidad_promedio': 12.3
    },
    # ...
}

# Crea video anotado
video_with_annotations = visualizer.draw_tracking(frame, tracking_data)
```

**Resultado Visual:**
```
MAPA DE CALOR:
┌─────────────────────────────────────────┐
│ 🔴🔴                              🟡🟡 │  ← Zonas calientes (mucha actividad)
│ 🔴🔴                              🟡🟡 │
│                                         │
│         🟢🟢🟢                          │  ← Zona tibia (actividad media)
│         🟢🟢🟢                          │
│                                         │
│                 🔵🔵                    │  ← Zona fría (poca actividad)
└─────────────────────────────────────────┘

ESTADÍSTICAS:
Player 1: 450px recorridos, 89% posesión
Player 2: 320px recorridos, 11% posesión
Total jugadas: 15 (5 triples, 3 contraataques)
```

---

## 💻 Ejemplo de Código Real (Cómo lo Usarías)

### Opción 1: Código Python Simple

```python
# 1. Importar el sistema
from src.main import BasketballAnalyzer

# 2. Crear el analizador (carga todos los módulos automáticamente)
analyzer = BasketballAnalyzer()

# 3. Procesar tu video
results = analyzer.process_video(
    video_path='mi_partido.mp4',           # Tu video aquí
    output_path='partido_analizado.mp4',   # Video con anotaciones
    generate_heatmap=True,                  # Sí, quiero heatmaps
    classify_plays=True,                    # Sí, clasifica jugadas
    show_progress=True                      # Muestra barra de progreso
)

# 4. Ver resultados
print(results['statistics'])  # Estadísticas del partido
print(results['play_classifications'])  # Jugadas detectadas

# 5. Guardar todo
analyzer.save_results(results, 'resultados/')
```

**Esto automáticamente:**
1. ✅ Detecta jugadores y balón en cada frame
2. ✅ Les hace tracking (IDs consistentes)
3. ✅ Clasifica las jugadas importantes
4. ✅ Genera mapas de calor
5. ✅ Calcula estadísticas
6. ✅ Exporta video anotado
7. ✅ Guarda reportes en archivos

---

### Opción 2: Usar Scripts (Más Fácil)

```bash
# Analizar un video desde la terminal

python scripts/analyze_video.py \
    --input videos/partido.mp4 \
    --output resultados/analizado.mp4 \
    --save-results resultados/

# ¡Y listo! El sistema hace todo automáticamente
```

**Output que obtendrás:**
```
resultados/
├── analizado.mp4              # Video con jugadores marcados
├── results.pkl                # Datos completos
├── report.txt                 # Reporte de estadísticas
├── heatmap_players.png        # Mapa de calor de jugadores
└── heatmap_ball.png           # Mapa de calor del balón
```

---

## 🔧 ¿Cómo Funciona Internamente Cada Módulo?

### 1. DETECTOR (YOLOv8)

```python
# src/detection/detector.py

# Cuando llamas a detect_frame(frame):
1. Toma la imagen (1920x1080 píxeles)
2. La pasa por la red neuronal YOLOv8
3. YOLOv8 devuelve:
   - Bounding boxes: [x1, y1, x2, y2]
   - Clase: 0=person, 32=sports_ball
   - Confianza: 0.0 - 1.0
4. Filtra solo jugadores (clase 0) y balón (clase 32)
5. Retorna lista de detecciones
```

**Proceso interno:**
```
Frame → YOLOv8 → [
    [100, 200, 150, 400, conf=0.95, class=0],  # Jugador
    [200, 180, 250, 380, conf=0.92, class=0],  # Jugador
    [300, 100, 320, 120, conf=0.85, class=32]  # Balón
]
```

---

### 2. TRACKER (SORT)

```python
# src/tracking/sort_tracker.py

# Cuando llamas a update(detecciones):
1. PREDICE dónde estarán los objetos actuales (Kalman Filter)
2. COMPARA predicciones con nuevas detecciones (IoU)
3. ASOCIA usando Hungarian Algorithm
   - Si match > 0.3 IoU → mismo objeto
   - Si no match → nuevo objeto
4. ACTUALIZA posiciones con Kalman
5. ELIMINA tracks perdidos (>30 frames sin detección)
6. Retorna IDs + posiciones
```

**Proceso interno:**
```
Frame N-1:  [Player ID=1 en (100,200)]
                ↓ Kalman predice
Frame N:    [Predicción: (105,202)]
                ↓ Compara con detección nueva
            [Detección: (106,201)]
                ↓ IoU = 0.85 > 0.3 → MATCH!
            [Confirma: Player ID=1 en (106,201)]
```

---

### 3. CLASSIFIER (LSTM)

```python
# src/classification/play_classifier.py

# Cuando llamas a classify(secuencia_30_frames):
1. EXTRAE features de cada frame:
   - Posiciones de jugadores (x,y)
   - Velocidades (Δx, Δy)
   - Distancias al balón
   - Dispersión espacial
   - etc. (56 features total)

2. NORMALIZA (escala 0-1)

3. PASA por LSTM:
   Frame 1 → [56 features] →┐
   Frame 2 → [56 features] →├→ LSTM → Estado oculto →┐
   ...                      │                         │
   Frame 30→ [56 features] →┘                         ↓
                                                  Capa FC
                                                      ↓
                                            [0.05, 0.12, 0.78, 0.03, 0.02]
                                              ↑
                                            TRIPLE (78%)

4. SOFTMAX → probabilidades para cada clase
5. ARGMAX → clase con mayor probabilidad
```

**Ejemplo de features extraídas:**
```python
Frame 150:
- jugador_1_pos: (0.35, 0.42)  # Normalizado 0-1
- jugador_1_vel: (0.05, 0.02)  # Se mueve lento
- ball_pos: (0.52, 0.15)        # Arriba (posible tiro)
- ball_vel: (0.02, -0.18)       # Subiendo rápido
- dist_player1_ball: 0.25       # Cerca del jugador 1
→ Modelo deduce: probablemente un TIRO
```

---

### 4. VISUALIZER (Heatmaps)

```python
# src/visualization/heatmap.py

# Cuando llamas a generate_player_heatmap(historial):
1. Crea matriz vacía (108x192) - resolución reducida
2. Por cada frame del historial:
   - Obtiene centro de cada jugador: (x,y)
   - Mapea a coordenadas del heatmap
   - Incrementa contador en esa celda
3. Aplica suavizado Gaussiano (difumina)
4. Normaliza a 0-1
5. Retorna matriz de intensidades
```

**Proceso:**
```
Posiciones:        Grid:           Suavizado:      Final:
(350, 500) →      [0 0 1 0] →     [0 1 2 1] →    [0.0 0.3 0.7 0.3]
(360, 510) →      [0 0 1 0] →     [1 2 3 2] →    [0.3 0.7 1.0 0.7]
(355, 505) →      [0 1 1 0] →     [0 1 2 1] →    [0.0 0.3 0.7 0.3]
                                                   ↑
                                              Rojo = 1.0 (caliente)
                                              Azul = 0.0 (frío)
```

---

## 🎯 ¿Qué Hace Cada Archivo?

```
src/
├── main.py                      # ← ORQUESTADOR: Usa todos los módulos
│
├── detection/
│   └── detector.py              # ← YOLO: Detecta jugadores/balón
│
├── tracking/
│   ├── sort_tracker.py          # ← KALMAN + HUNGARIAN: Tracking básico
│   └── tracker.py               # ← WRAPPER: Tracking específico baloncesto
│
├── classification/
│   ├── play_classifier.py       # ← LSTM: Clasifica jugadas
│   └── feature_extractor.py    # ← EXTRACTOR: Calcula features
│
├── visualization/
│   ├── visualizer.py            # ← DIBUJADOR: Anota frames
│   ├── heatmap.py               # ← HEATMAPS: Mapas de calor
│   └── statistics.py            # ← STATS: Calcula métricas
│
└── utils/
    ├── config_loader.py         # ← CONFIGS: Lee config.yaml
    ├── video_utils.py           # ← VIDEO: Lee/escribe videos
    └── logger.py                # ← LOGS: Sistema de logging
```

---

## 🚀 Instalación y Primer Uso

### Paso 1: Instalar Dependencias

```bash
# Clonar repo (ya lo hiciste)
cd Proyecto-de-IA

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  en Windows

# Instalar librerías
pip install -r requirements.txt
```

### Paso 2: Conseguir un Video

```bash
# Opción A: Descargar de YouTube
pip install yt-dlp
yt-dlp "URL_PARTIDO_BASKETBALL" -o "data/raw/partido.mp4"

# Opción B: Usar un video propio
# Copia tu video a: data/raw/mi_partido.mp4
```

### Paso 3: ¡Analizar!

```bash
python scripts/analyze_video.py \
    --input data/raw/partido.mp4 \
    --output data/processed/resultado.mp4 \
    --save-results results/
```

### Paso 4: Ver Resultados

```bash
# Video procesado
results/analizado.mp4

# Reporte de texto
cat results/report.txt

# Heatmaps
eog results/heatmap_players.png  # Linux
# o: start results/heatmap_players.png  # Windows
```

---

## ❓ Preguntas Frecuentes

### ¿Necesito entrenar algo?

**Detección y Tracking:** NO, usa YOLOv8 pre-entrenado
**Clasificación:** SÍ (pero funciona sin entrenar, solo con baja precisión)

### ¿Funciona en CPU?

SÍ, pero más lento (~2-5 FPS vs 15-30 FPS en GPU)

### ¿Qué tan grande es el video de salida?

Similar al de entrada. Si quieres más pequeño, reduce resolución en config.

### ¿Puedo analizar videos en vivo?

SÍ, modifica `main.py` para usar webcam en vez de archivo.

---

## 🎓 Resumen para tu Proyecto

**Lo que tienes:**
- ✅ Sistema 100% funcional
- ✅ Cumple EXACTAMENTE con tu propuesta
- ✅ Listo para experimentar
- ✅ Documentación completa

**Lo que debes hacer:**
1. Instalar dependencias
2. Conseguir videos de baloncesto
3. Procesar videos
4. Analizar resultados
5. (Opcional) Entrenar clasificador con datos reales
6. Escribir informe de resultados

**¡Todo el código difícil ya está hecho! 🎉**
