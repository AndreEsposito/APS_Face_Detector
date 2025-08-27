from __future__ import annotations

# Tamanhos e thresholds
FACE_MIN_SIZE = (80, 80)
FACE_IMG_SIZE = (200, 200)

# Liveness (EAR)
EAR_THRESH = 0.22
EAR_CONSEC_FRAMES_FOR_BLINK = 3
LIVENESS_WINDOW_SEC = 6
REQUIRE_BLINKS_IN_WINDOW = 1

# LBPH
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
LBPH_THRESHOLD_DECISION = 70.0  # ajuste por calibração

# RBAC – recursos por nível
NIVEIS_RECURSOS = {
    1: ["Consulta pública de dados ambientais"],
    2: ["Consulta pública", "Relatórios internos (Diretoria)"],
    3: ["Consulta pública", "Relatórios internos", "Operações restritas (Ministro)"],
}
