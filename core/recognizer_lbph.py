from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from .paths import SAMPLES_DIR, LBPH_MODEL_PATH
from .config import LBPH_RADIUS, LBPH_NEIGHBORS, LBPH_GRID_X, LBPH_GRID_Y

def create_lbph():
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError("cv2.face indisponível. Instale opencv-contrib-python.")
    return cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS, neighbors=LBPH_NEIGHBORS, grid_x=LBPH_GRID_X, grid_y=LBPH_GRID_Y
    )

def load_or_create_lbph():
    model = create_lbph()
    if Path(LBPH_MODEL_PATH).exists():
        model.read(str(LBPH_MODEL_PATH))
    return model

def load_dataset_samples(face_img_size: tuple[int,int]) -> tuple[list[np.ndarray], list[int]]:
    images, labels = [], []
    for user_dir in SAMPLES_DIR.glob("*"):
        if not user_dir.is_dir():
            continue
        try:
            uid = int(user_dir.name)
        except ValueError:
            continue
        for img_path in user_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != face_img_size[::-1]:
                img = cv2.resize(img, face_img_size, interpolation=cv2.INTER_AREA)
            images.append(img)
            labels.append(uid)
    return images, labels

def train_lbph_from_samples(face_img_size: tuple[int,int], save: bool = True):
    imgs, labels = load_dataset_samples(face_img_size)
    if len(imgs) < 1:
        raise RuntimeError("Sem amostras para treinar o LBPH. Faça ENROLL primeiro.")
    model = create_lbph()
    model.train(imgs, np.array(labels))
    if save:
        Path(LBPH_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        model.write(str(LBPH_MODEL_PATH))
    return model

def predict_lbph(face_img_gray, model) -> tuple[int, float]:
    return model.predict(face_img_gray)  # (label, distance) – menor = melhor
