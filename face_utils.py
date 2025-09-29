import cv2
import numpy as np
from deepface import DeepFace
from db import add_user, get_all_users
import time
import os

# ----------------- CONFIGURAÇÕES -----------------
MODEL = 'Facenet'              # DeepFace models: VGG-Face, Facenet, OpenFace, DeepFace, DeepID
DISTANCE_METRIC = 'cosine'     # 'cosine' geralmente funciona bem
THRESHOLD = 0.45               # Ajuste empiricamente: valores mais baixos = mais rigoroso
DETECTOR_BACKEND = "retinaface"  # "retinaface", "mtcnn", "opencv", etc.
TMP_DIR = "tmp_faces"
os.makedirs(TMP_DIR, exist_ok=True)

# ----------------- FUNÇÕES -----------------
def capture_image_from_webcam(window_name="capture", show=True, skip_frames=5):
    """Captura uma imagem da webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a webcam.")
    ret, frame = cap.read()
    for _ in range(skip_frames):
        ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Falha ao capturar frame.")
    if show:
        cv2.imshow(window_name, frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    cap.release()
    return frame

def enroll_user(name: str, role_level: int, img=None):
    """Cadastra usuário com embedding do rosto."""
    if img is None:
        img = capture_image_from_webcam()
    tmp_path = os.path.join(TMP_DIR, f"enroll_{int(time.time())}.jpg")
    cv2.imwrite(tmp_path, img)

    # Gera embedding
    try:
        obj = DeepFace.represent(
            img_path=tmp_path,
            model_name=MODEL,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
    except Exception as e:
        os.remove(tmp_path)
        raise RuntimeError(f"Falha no reconhecimento facial: {e}")

    if isinstance(obj, list) and len(obj) > 0:
        emb = obj[0]["embedding"] if isinstance(obj[0], dict) and "embedding" in obj[0] else obj[0]
    else:
        emb = obj

    emb_list = list(map(float, emb))
    user = add_user(name, role_level, emb_list)
    os.remove(tmp_path)
    return user

def _embedding_from_image(img):
    """Extrai embedding de uma imagem."""
    tmp_path = os.path.join(TMP_DIR, f"rec_{int(time.time())}.jpg")
    cv2.imwrite(tmp_path, img)
    try:
        obj = DeepFace.represent(
            img_path=tmp_path,
            model_name=MODEL,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
    except Exception as e:
        os.remove(tmp_path)
        raise RuntimeError(f"Falha ao extrair embedding: {e}")
    os.remove(tmp_path)

    if isinstance(obj, list) and len(obj) > 0:
        emb = obj[0]["embedding"] if isinstance(obj[0], dict) and "embedding" in obj[0] else obj[0]
    else:
        emb = obj
    return np.array(emb, dtype=float)

def _cosine_distance(a, b):
    """Calcula a distância coseno entre dois vetores."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 1.0
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize(img=None):
    """
    Reconhece usuário comparando com DB.
    Retorna (user, distância, erro).
    """
    if img is None:
        img = capture_image_from_webcam()
    try:
        emb = _embedding_from_image(img)
    except Exception as e:
        return None, None, f"Erro ao extrair embedding: {e}"

    users = get_all_users()
    if not users:
        return None, None, "Nenhum usuário cadastrado no DB."

    best_user = None
    best_dist = 10.0

    for u in users:
        stored = np.array(list(map(float, u.embedding.split(","))))
        dist = _cosine_distance(emb, stored)
        if dist < best_dist:
            best_dist = dist
            best_user = u

    if best_user and best_dist <= THRESHOLD:
        return best_user, best_dist, None
    else:
        return None, best_dist, "Nenhuma correspondência confiável encontrada."
