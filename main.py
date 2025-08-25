# main.py - OpenCV + Facemark LBF (Python 3.13 friendly)
# ----------------------------------------------
# Requisitos:
#   pip install --upgrade pip wheel setuptools
#   pip install opencv-contrib-python opencv-python numpy
#   (Opcional YOLO: torch + ultralytics, mas está DESLIGADO por padrão)
#
# Modelos:
#   - Landmarks LBF: coloque o arquivo "lbfmodel.yaml" em ./models/
#     (doc OpenCV Facemark LBF)
#
# Execução:
#   python main.py --camera-index 0 --model-path models/lbfmodel.yaml
#
# Controles:
#   - 'S' : salva um frame em ./captures/
#   - 'Q' : encerra

import argparse
import time
from pathlib import Path
import sys

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face detector + landmarks + blink (OpenCV Facemark LBF)"
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="Índice da câmera (default: 0)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path("models") / "lbfmodel.yaml"),
        help="Caminho para o modelo LBF (lbfmodel.yaml)",
    )
    parser.add_argument(
        "--blink-thresh",
        type=float,
        default=0.22,
        help="Limiar EAR para indicar blink (default: 0.22)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="captures",
        help="Diretório de saída para capturas (default: captures/)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["haar"],
        default="haar",
        help="Detector de faces: por enquanto apenas 'haar' (default)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Largura desejada do frame (0 = manter padrão da câmera)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Altura desejada do frame (0 = manter padrão da câmera)",
    )
    return parser.parse_args()


def ensure_opencv_contrib():
    # Verifica se o módulo cv2.face existe (só no contrib)
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "createFacemarkLBF"):
        msg = (
            "\n[ERRO] cv2.face não disponível. Instale o pacote correto:\n"
            "  pip install opencv-contrib-python\n"
            "Dica: desinstale o opencv-python se houver conflito:\n"
            "  pip uninstall -y opencv-python && pip install opencv-contrib-python\n"
        )
        print(msg)
        sys.exit(1)


def load_facemark(model_path: str):
    ensure_opencv_contrib()
    facemark = cv2.face.createFacemarkLBF()
    model_file = Path(model_path)
    if not model_file.exists():
        print(
            f"\n[ERRO] Modelo LBF não encontrado em: {model_file}\n"
            "Baixe o 'lbfmodel.yaml' (modelo de 68 pontos) e salve em ./models/\n"
            "Depois rode novamente.\n"
        )
        sys.exit(1)
    facemark.loadModel(str(model_file))
    return facemark


def create_face_detector_haar():
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        print(
            f"\n[ERRO] Haarcascade não encontrado em {cascade_path}\n"
            "Reinstale o OpenCV ou forneça o caminho correto do XML.\n"
        )
        sys.exit(1)
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        print(
            f"\n[ERRO] Falha ao carregar o Haarcascade em {cascade_path}\n"
            "Arquivo corrompido? Reinstale o OpenCV.\n"
        )
        sys.exit(1)
    return face_cascade


def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    # EAR baseado nos 6 pontos do olho (68 landmarks)
    # Indices p/ olho: 0..5 referem-se a um slice já ajustado (ex.: 36..41 ou 42..47)
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def compute_ear_from_landmarks(shape: np.ndarray) -> float:
    # shape: (68,2)
    left = shape[36:42]
    right = shape[42:48]
    return (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0


def draw_landmarks(frame: np.ndarray, pts: np.ndarray):
    # pts: (68,2)
    for (px, py) in pts:
        cv2.circle(frame, (int(px), int(py)), 1, (0, 255, 0), -1)


def put_text(frame, text, org, scale=0.6, color=(0, 255, 0), thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    args = parse_args()

    # Detector de face
    if args.detector == "haar":
        face_detector = create_face_detector_haar()
    else:
        print("[ERRO] Apenas 'haar' está disponível no momento.")
        sys.exit(1)

    # Facemark LBF
    facemark = load_facemark(args.model_path)

    # Captura de vídeo
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(
            f"\n[ERRO] Não foi possível abrir a webcam no índice {args.camera_index}.\n"
            "Tente outro índice, ex.: --camera-index 1\n"
        )
        sys.exit(1)

    # Ajuste opcional de resolução
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    blink_thresh = args.blink_thresh
    win_name = "Face/Landmarks - OpenCV (3.13)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps = 0.0

    print(
        "\nControles:\n"
        "  [S] salvar frame em ./captures/\n"
        "  [Q] sair\n"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[AVISO] Frame não capturado da câmera. Encerrando...")
            break

        # FPS básico
        now = time.time()
        dt = now - prev_time
        if dt > 0:
            fps = 1.0 / dt
        prev_time = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecção de faces (Haar)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )
        faces_rects = [tuple(map(int, f)) for f in faces]

        if faces_rects:
            # Converte lista -> NumPy (n,4) int32
            faces_np = np.array(faces_rects, dtype=np.int32)

            # Use a imagem em tons de cinza no fit (LBF espera 8-bit single-channel)
            success, landmarks = facemark.fit(gray, faces_np)

            if success:
                for (x, y, w, h), lms in zip(faces_rects, landmarks):
                    pts = lms[0]  # (68, 2)
                    draw_landmarks(frame, pts)
                    ear = compute_ear_from_landmarks(pts)

                    put_text(frame, f"EAR: {ear:.3f}", (x, max(0, y - 10)))
                    if ear < blink_thresh:
                        put_text(frame, "BLINK", (x, y + h + 20), scale=0.8, color=(0, 0, 255))

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                # Pode falhar ocasionalmente para alguns retângulos; tudo bem
                pass

        # FPS na tela
        put_text(frame, f"FPS: {fps:5.1f}", (10, 25))

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            out_path = save_dir / f"frame_{int(time.time())}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"[OK] Frame salvo em: {out_path}")

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
