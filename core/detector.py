from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from .config import FACE_MIN_SIZE

def load_haar_detector() -> cv2.CascadeClassifier:
    haar_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not haar_path.exists():
        raise RuntimeError(f"Haarcascade não encontrado em: {haar_path}")
    face_cascade = cv2.CascadeClassifier(str(haar_path))
    if face_cascade.empty():
        raise RuntimeError(f"Falha ao carregar Haarcascade: {haar_path}")
    return face_cascade

def detect_faces(gray, face_cascade) -> tuple[list[tuple[int,int,int,int]], np.ndarray]:
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=FACE_MIN_SIZE)
    rects = [tuple(map(int, f)) for f in faces]
    faces_np = np.array(rects, dtype=np.int32) if rects else np.empty((0, 4), dtype=np.int32)
    return rects, faces_np
