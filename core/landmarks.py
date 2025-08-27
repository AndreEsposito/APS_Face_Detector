from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from .config import FACE_IMG_SIZE
from .utils import put_text

def ensure_lbf_exists(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo LBF não encontrado em: {model_path}\n"
            "Baixe 'lbfmodel.yaml' e salve nesse caminho."
        )
    if model_path.stat().st_size < 5 * 1024 * 1024:
        raise RuntimeError("Arquivo lbfmodel.yaml parece incompleto/corrompido (tamanho inesperado).")

def load_facemark(model_path: Path):
    ensure_lbf_exists(model_path)
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "createFacemarkLBF"):
        raise RuntimeError(
            "cv2.face indisponível. Instale opencv-contrib-python.\n"
            "  pip uninstall -y opencv-python\n  pip install opencv-contrib-python"
        )
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(str(model_path))
    return facemark

def facemark_fit(gray, frame_for_draw, facemark, faces_np, draw=True) -> list[np.ndarray]:
    if faces_np.size == 0:
        return []
    success, landmarks = facemark.fit(gray, faces_np)  # precisa grayscale 8-bit
    out = []
    if success:
        for (x, y, w, h), lms in zip(faces_np, landmarks):
            pts = lms[0]  # (68,2)
            out.append(pts)
            if draw:
                for (px, py) in pts:
                    cv2.circle(frame_for_draw, (int(px), int(py)), 1, (0, 255, 0), -1)
                cv2.rectangle(frame_for_draw, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
    return out

def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def compute_ear_from_landmarks(pts68: np.ndarray) -> float:
    left = pts68[36:42]
    right = pts68[42:48]
    return (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0

def crop_face(gray, rect, size: tuple[int,int] = FACE_IMG_SIZE):
    x, y, w, h = rect
    roi = gray[max(0, y): y + h, max(0, x): x + w]
    if roi.size == 0:
        return None
    roi = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
    roi = cv2.equalizeHist(roi)
    return roi

def _center(pts):
    return pts.mean(axis=0)

def estimate_yaw_pitch(pts68: np.ndarray) -> tuple[float, float]:
    """
    Retorna (yaw, pitch) ~ [-1..1] (aproximação).
    yaw  >0: olhando p/ direita; <0: p/ esquerda
    pitch>0: olhando p/ baixo;   <0: p/ cima
    """
    left_eye  = pts68[36:42]   # 6 pts
    right_eye = pts68[42:48]   # 6 pts
    nose_tip  = pts68[30]      # 1 pt
    nose_bridge = pts68[27]    # 1 pt
    chin     = pts68[8]        # 1 pt

    c_left  = _center(left_eye)
    c_right = _center(right_eye)
    c_eyes  = (c_left + c_right) / 2.0
    inter_eye = np.linalg.norm(c_right - c_left) + 1e-6

    # yaw: deslocamento horizontal do nariz em relação ao centro dos olhos
    yaw = float((nose_tip[0] - c_eyes[0]) / inter_eye)

    # pitch: razão de distâncias verticais (nariz->olhos vs queixo->nariz)
    up = float((nose_tip[1] - c_eyes[1]) / (abs(chin[1] - nose_tip[1]) + 1e-6))
    pitch = up  # >0 olhando para baixo, <0 para cima (aprox.)

    # clamp leve
    yaw = max(-1.5, min(1.5, yaw))
    pitch = max(-1.5, min(1.5, pitch))
    return yaw, pitch