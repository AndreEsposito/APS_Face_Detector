# -*- coding: utf-8 -*-
"""
Detecção Unificada
- MediaPipe Hands + FaceMesh (EAR/piscadas)
- YOLOv8 (objetos) [opcional]
- face_recognition (identidade) [opcional]
- HUD com FPS, atalhos e overlays
- Proteções contra ausências (sem YOLO, sem known_faces, etc.)

Atalhos:
  q = sair
  p = pausar/continuar
  s = salvar frame
  1 = YOLO on/off
  2 = Face Recognition on/off
  3 = Desenho das mãos on/off
  b = Banner de piscada on/off
  h = ajuda (mostrar/ocultar)

Requisitos (se necessário):
  pip install opencv-python mediapipe ultralytics face_recognition
  # Se tiver GPU NVIDIA, instale torch compatível com sua CUDA.
"""

import os
import cv2
import time
import math
import numpy as np

# ----- Dependências opcionais com proteção -----
has_torch = True
try:
    import torch
except Exception:
    has_torch = False

has_ultra = True
try:
    from ultralytics import YOLO
except Exception:
    has_ultra = False

has_facerec = True
try:
    import face_recognition
except Exception:
    has_facerec = False

import mediapipe as mp

# -------------------- Configurações --------------------
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
CAPTURE_INDEX = 0

YOLO_MODEL_PATH    = "yolov8n.pt"
YOLO_CONF_THRESHOLD = 0.45
KNOWN_FACES_DIR     = "known_faces"

DEVICE = "cuda" if (has_torch and torch.cuda.is_available()) else "cpu"

# -------------------- Utilidades de desenho --------------------
def draw_text(img, txt, org, scale=0.7, color=(0,255,0), thickness=2, line=cv2.LINE_AA):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, line)

def draw_panel(img, x, y, w, h, color=(0,0,0), alpha=0.35):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

# -------------------- MediaPipe Detector --------------------
class MediaPipeDetector:
    def __init__(self, face_conf=0.7, hand_conf=0.7):
        # Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=hand_conf,
            min_tracking_confidence=0.5
        )
        # FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=face_conf,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Índices dos olhos (EAR)
        self.LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

        # Piscada
        self.blink_threshold   = 0.24    # ajuste fino se necessário
        self.consec_frames     = 2       # nº de frames abaixo do limiar para validar piscada
        self._closed_frames    = 0
        self.blink_count       = 0
        self._last_blink_time  = 0.0
        self.blink_display_time = 0.8    # segundos visível
        self.blinking          = False

    # ---- dedos levantados (mão direita/esquerda) ----
    def _fingers_up(self, hand_landmarks, hand_label):
        tips = [4, 8, 12, 16, 20]
        if hand_landmarks is None:
            return [0]*5
        lm = hand_landmarks.landmark
        fingers = []
        # Polegar (eixo X inverte entre mãos)
        if hand_label == "Right":
            fingers.append(int(lm[tips[0]].x < lm[tips[0]-1].x))
        else:
            fingers.append(int(lm[tips[0]].x > lm[tips[0]-1].x))
        # Demais dedos (eixo Y: ponta acima da junta)
        for i in range(1, 5):
            fingers.append(int(lm[tips[i]].y < lm[tips[i]-2].y))
        return fingers

    # ---- EAR ----
    @staticmethod
    def _dist(p1, p2):
        return math.dist((p1.x, p1.y), (p2.x, p2.y))

    def _ear(self, landmarks, idx):
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idx]
        vert1 = self._dist(p2, p6)
        vert2 = self._dist(p1, p5)
        horiz = self._dist(p3, p4)
        return (vert1 + vert2) / (2.0 * horiz) if horiz != 0 else 0.0

    # ---- processar frame ----
    def process(self, frame, draw=True, draw_hands=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MÃOS
        hands_info = []
        hands_res = self.hands.process(rgb)
        if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
            for lm, handed in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
                label = handed.classification[0].label  # "Left"/"Right"
                fng = self._fingers_up(lm, label)
                hands_info.append({"label": label, "fingers": fng, "count": int(sum(fng))})
                if draw and draw_hands:
                    self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

        # FACE + PISCADA
        avg_ear = None
        face_res = self.face_mesh.process(rgb)
        now = time.time()
        if face_res.multi_face_landmarks:
            for face in face_res.multi_face_landmarks:
                lnd = face.landmark
                left  = self._ear(lnd, self.LEFT_EYE_IDX)
                right = self._ear(lnd, self.RIGHT_EYE_IDX)
                avg_ear = (left + right) / 2.0

                # lógica de piscada
                if avg_ear < self.blink_threshold:
                    self._closed_frames += 1
                else:
                    if self._closed_frames >= self.consec_frames:
                        self.blink_count += 1
                        self._last_blink_time = now
                    self._closed_frames = 0

                self.blinking = (now - self._last_blink_time) < self.blink_display_time

                if draw:
                    self.mp_draw.draw_landmarks(frame, face, self.mp_face_mesh.FACEMESH_CONTOURS)

        return {
            "frame": frame,
            "hands_info": hands_info,
            "avg_ear": avg_ear,
            "blink_count": self.blink_count,
            "blinking": self.blinking
        }

# -------------------- Face Recognition --------------------
def load_known_faces(directory):
    encs, names = [], []
    if not (has_facerec and os.path.isdir(directory)):
        return encs, names
    for fn in os.listdir(directory):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(directory, fn)
            try:
                img = face_recognition.load_image_file(path)
                e = face_recognition.face_encodings(img)
                if e:
                    encs.append(e[0])
                    names.append(os.path.splitext(fn)[0])
            except Exception:
                pass
    return encs, names

# -------------------- YOLO Loader (opcional) --------------------
def load_yolo(model_path):
    if not has_ultra:
        return None
    if not os.path.isfile(model_path):
        return None
    try:
        return YOLO(model_path)
    except Exception:
        return None

# -------------------- HUD de ajuda --------------------
HELP_LINES = [
    "Atalhos:",
    "  q = sair",
    "  p = pausar/continuar",
    "  s = salvar frame",
    "  1 = YOLO on/off",
    "  2 = Face Recognition on/off",
    "  3 = Desenho das maos on/off",
    "  b = Banner piscada on/off",
    "  h = mostrar/ocultar ajuda"
]

def draw_hud(frame, avg_ear, blink_count, fps, show_blink_banner,
             yolo_on, fr_on, hands_draw_on, thr):
    # canto superior esquerdo
    draw_text(
        frame,
        f"EAR: {avg_ear:.2f} (thr {thr:.2f})" if avg_ear is not None else "EAR: --",
        (12, 34), 0.8, (0, 255, 255), 2
    )
    draw_text(frame, f"Piscadas: {blink_count}", (12, 64), 0.9, (0, 255, 255), 2)
    draw_text(frame, f"FPS: {fps:.1f}", (12, frame.shape[0] - 12), 0.8, (0, 255, 0), 2)

    # status no topo
    status = f"[YOLO:{'ON' if yolo_on else 'OFF'} | FACE_REC:{'ON' if fr_on else 'OFF'} | MAOS_DRAW:{'ON' if hands_draw_on else 'OFF'}]"
    draw_text(frame, status, (12, 96), 0.6, (255, 255, 255), 2)

    # banner piscada
    if show_blink_banner:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
        draw_text(frame, "PISCADA DETECTADA!", (12, 35), 1.0, (255, 255, 255), 2)

def draw_help(frame):
    pad = 10
    w = 380
    h = 22 * (len(HELP_LINES) + 2)
    draw_panel(frame, pad, pad, w, h, (0, 0, 0), 0.55)
    y = pad + 28
    for line in HELP_LINES:
        draw_text(frame, line, (pad + 14, y), 0.7, (255, 255, 255), 2)
        y += 22

# -------------------- Main --------------------
def main():
    # Captura simples (sem CAP_PROP_BUFFER_SIZE para evitar erro)
    cap = cv2.VideoCapture(CAPTURE_INDEX)

    # Tenta MJPG para reduzir latência/buffer (ignora se não suportar)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERRO] Câmera não encontrada.")
        return

    # Inicializações
    mpd = MediaPipeDetector()
    yolo = load_yolo(YOLO_MODEL_PATH)
    known_enc, known_names = load_known_faces(KNOWN_FACES_DIR)

    use_yolo        = (yolo is not None)                  # toggle inicial
    use_facerec     = (len(known_enc) > 0) and has_facerec
    draw_hands      = True
    show_help_panel = True
    show_blink_banner = True
    paused          = False

    window = "Detecção Unificada"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    fps_smooth = 0.0
    t_prev = time.time()

    print("[INFO] Iniciando. Pressione 'h' para ajuda, 'q' para sair.")
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[ERRO] Falha ao ler frame da câmera.")
                break

            # Garante resolução alvo
            if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)

            # ---- MediaPipe ----
            out = mpd.process(frame, draw=True, draw_hands=draw_hands)
            frame = out["frame"]

            # ---- YOLO (opcional) ----
            if use_yolo:
                try:
                    results = yolo.predict(frame, conf=YOLO_CONF_THRESHOLD, device=DEVICE, verbose=False)
                    for r in results:
                        if r.boxes is None:
                            continue
                        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = r.names[int(cls)]
                            draw_text(frame, f"{label} {float(conf):.2f}", (x1, max(15, y1 - 8)), 0.6, (255, 255, 0), 2)
                except Exception:
                    # falha pontual do YOLO: ignora este frame
                    pass

            # ---- Face Recognition (opcional) ----
            if use_facerec:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    locs = face_recognition.face_locations(rgb, model="hog")
                    encs = face_recognition.face_encodings(rgb, locs)
                    for (top, right, bottom, left), enc in zip(locs, encs):
                        name = "Desconhecido"
                        if known_enc:
                            matches = face_recognition.compare_faces(known_enc, enc)
                            dists = face_recognition.face_distance(known_enc, enc)
                            if len(dists) > 0:
                                idx = int(np.argmin(dists))
                                if matches[idx]:
                                    name = known_names[idx]
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        draw_text(frame, name, (left, max(15, top - 8)), 0.65, (0, 255, 0), 2)
                except Exception:
                    pass

            # ---- HUD ----
            now = time.time()
            fps = 1.0 / max(1e-6, (now - t_prev))
            t_prev = now
            fps_smooth = fps_smooth * 0.90 + fps * 0.10

            draw_hud(
                frame=frame,
                avg_ear=out["avg_ear"],
                blink_count=out["blink_count"],
                fps=fps_smooth,
                show_blink_banner=(out["blinking"] and show_blink_banner),
                yolo_on=use_yolo,
                fr_on=use_facerec,
                hands_draw_on=draw_hands,
                thr=mpd.blink_threshold,
            )

            if show_help_panel:
                draw_help(frame)

            # Mostra contagem de dedos (texto simples)
            for i, hinfo in enumerate(out["hands_info"]):
                draw_text(frame, f"Mao {i+1}: {hinfo['count']} dedos", (12, 130 + 30 * i), 0.7, (0, 255, 0), 2)

            cv2.imshow(window, frame)
        else:
            # pausado: ainda mostra a última imagem
            cv2.imshow(window, frame)

        # ---- Teclado ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("s"):
            ts = int(time.time())
            fn = f"capture_{ts}.jpg"
            try:
                cv2.imwrite(fn, frame)
                print(f"[INFO] Salvou {fn}")
            except Exception as e:
                print(f"[ERRO] Falha ao salvar: {e}")
        elif key == ord("1"):
            use_yolo = not use_yolo
            print(f"[TOGGLE] YOLO -> {'ON' if use_yolo else 'OFF'}")
        elif key == ord("2"):
            # só liga se tiver encodings carregados
            use_facerec = not use_facerec and (len(known_enc) > 0) and has_facerec
            print(f"[TOGGLE] Face Recognition -> {'ON' if use_facerec else 'OFF'}")
        elif key == ord("3"):
            draw_hands = not draw_hands
            print(f"[TOGGLE] Desenho das mãos -> {'ON' if draw_hands else 'OFF'}")
        elif key == ord("b"):
            show_blink_banner = not show_blink_banner
            print(f"[TOGGLE] Banner piscada -> {'ON' if show_blink_banner else 'OFF'}")
        elif key == ord("h"):
            show_help_panel = not show_help_panel

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
