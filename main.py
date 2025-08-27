from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from core.paths import ensure_dirs, LBF_MODEL_PATH_DEFAULT
from core.utils import setup_logging, put_text, save_frame
from core.config import (
    EAR_THRESH, EAR_CONSEC_FRAMES_FOR_BLINK, LIVENESS_WINDOW_SEC, REQUIRE_BLINKS_IN_WINDOW,
    FACE_IMG_SIZE, LBPH_THRESHOLD_DECISION
)
from core.detector import load_haar_detector, detect_faces
from core.landmarks import load_facemark, facemark_fit, compute_ear_from_landmarks, crop_face
from core.liveness import BlinkLiveness
from core.recognizer_lbph import load_or_create_lbph, train_lbph_from_samples, predict_lbph
from core.storage import db_init, db_upsert_user, db_inc_samples, db_get_user_by_id, log_access
from core.rbac import draw_rbac_overlay

# ---------- Pose helper (yaw/pitch) ----------
def _center(pts: np.ndarray) -> np.ndarray:
    return pts.mean(axis=0)

def estimate_yaw_pitch(pts68: np.ndarray) -> tuple[float, float]:
    left_eye  = pts68[36:42]
    right_eye = pts68[42:48]
    nose_tip  = pts68[30]
    chin      = pts68[8]

    c_left  = _center(left_eye)
    c_right = _center(right_eye)
    c_eyes  = (c_left + c_right) / 2.0
    inter_eye = float(np.linalg.norm(c_right - c_left) + 1e-6)

    yaw = float((nose_tip[0] - c_eyes[0]) / inter_eye)
    pitch = float((nose_tip[1] - c_eyes[1]) / (abs(chin[1] - nose_tip[1]) + 1e-6))

    yaw = max(-1.5, min(1.5, yaw))
    pitch = max(-1.5, min(1.5, pitch))
    return yaw, pitch

# ---------- UI: traffic light ("semáforo") ----------
def draw_semaforo(frame, ready: bool, face_ok: bool, interval_ok: bool, blink_ok: bool, pose_ok: bool):
    h, w = frame.shape[:2]
    panel_w, panel_h = 170, 120
    x, y = w - panel_w - 10, 10

    # background
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (80, 80, 80), 1)

    # main light
    light_color = (0, 180, 0) if ready else (0, 0, 230)
    cv2.circle(frame, (x + 25, y + 25), 12, light_color, -1)
    put_text(frame, "CAPTURA", (x + 45, y + 30), scale=0.55, color=(200, 200, 200))

    def badge(row_y, label, ok):
        c = (0, 180, 0) if ok else (0, 0, 230)
        cv2.rectangle(frame, (x + 10, row_y - 10), (x + 22, row_y + 2), c, -1)
        put_text(frame, label, (x + 30, row_y + 2), scale=0.5, color=(220, 220, 220))

    # detail rows
    badge(y + 50,  "Face",    face_ok)
    badge(y + 70,  "Interval",interval_ok)
    badge(y + 90,  "Blink",   blink_ok)
    badge(y + 110, "Pose",    pose_ok)

# ---------------- CLI ---------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="APS Face Access - Reconhecimento Facial + RBAC + Liveness (Python 3.13)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ENROLL
    pe = sub.add_parser("enroll", help="Cadastrar (coletar amostras) e treinar LBPH")
    pe.add_argument("--user", type=str, required=True, help="Nome do usuario para cadastro")
    pe.add_argument("--nivel", type=int, required=True, choices=[1, 2, 3], help="Nivel de acesso (1/2/3)")
    pe.add_argument("--samples", type=int, default=20, help="Qtde de amostras a capturar (padrao=20)")
    pe.add_argument("--camera-index", type=int, default=0, help="Indice da webcam (padrao=0)")
    pe.add_argument("--model-path", type=Path, default=LBF_MODEL_PATH_DEFAULT, help="Caminho do lbfmodel.yaml")
    # NOVAS FLAGS
    pe.add_argument("--capture-interval", type=float, default=0.7,
                    help="Tempo minimo (s) entre capturas consecutivas (padrao=0.7s)")
    pe.add_argument("--capture-on-blink", action="store_true",
                    help="Capturar somente quando ocorrer um blink (evento de liveness)")
    pe.add_argument("--pose-diversity", action="store_true",
                    help="Somente capturar quando a pose (yaw/pitch) variar o suficiente")
    pe.add_argument("--yaw-thresh", type=float, default=0.15,
                    help="Variação minima de yaw para capturar quando --pose-diversity")
    pe.add_argument("--pitch-thresh", type=float, default=0.12,
                    help="Variação minima de pitch para capturar quando --pose-diversity")

    # AUTH
    pa = sub.add_parser("auth", help="Autenticar usuario com liveness + RBAC")
    pa.add_argument("--camera-index", type=int, default=0, help="Indice da webcam (padrao=0)")
    pa.add_argument("--model-path", type=Path, default=LBF_MODEL_PATH_DEFAULT, help="Caminho do lbfmodel.yaml")
    pa.add_argument("--blink-thresh", type=float, default=EAR_THRESH, help=f"Limiar EAR (padrao={EAR_THRESH})")
    pa.add_argument("--lbph-thresh", type=float, default=LBPH_THRESHOLD_DECISION,
                    help=f"Limiar LBPH (padrao={LBPH_THRESHOLD_DECISION:.1f})")

    return p.parse_args()

# -------------- ENROLL --------------- #
def run_enroll(args):
    ensure_dirs()
    setup_logging()
    db_init()

    user_id = db_upsert_user(args.user, args.nivel)
    user_dir = Path("data") / "samples" / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    logging.info("ENROLL user_id=%s nome='%s' nivel=%s", user_id, args.user, args.nivel)

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir webcam índice {args.camera_index}")

    face_cascade = load_haar_detector()
    facemark = load_facemark(args.model_path)
    liveness = BlinkLiveness(EAR_THRESH, EAR_CONSEC_FRAMES_FOR_BLINK, LIVENESS_WINDOW_SEC, REQUIRE_BLINKS_IN_WINDOW)

    collected = 0
    prev = time.time()
    last_capture_ts = 0.0
    last_yaw_pitch: tuple[float, float] | None = None

    cv2.namedWindow("ENROLL", cv2.WINDOW_NORMAL)
    logging.info("Controles: [S] salvar frame / [Q] sair (encerra ao atingir --samples)")

    while True:
        ok, frame = cap.read()
        if not ok:
            logging.warning("Frame não capturado. Encerrando...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rects, faces_np = detect_faces(gray, face_cascade)
        pts_list = facemark_fit(gray, frame, facemark, faces_np, draw=True)

        blink_event = False
        yaw = pitch = None

        if pts_list:
            ear = compute_ear_from_landmarks(pts_list[0])
            before = liveness.count()
            liveness.update(ear)
            after = liveness.count()
            blink_event = (after > before)

            try:
                yaw, pitch = estimate_yaw_pitch(pts_list[0])
                put_text(frame, f"EAR:{ear:.3f}  yaw:{yaw:+.2f} pitch:{pitch:+.2f}", (10, 20), color=(200, 255, 200))
            except Exception:
                put_text(frame, f"EAR:{ear:.3f}", (10, 20), color=(200, 255, 200))
        else:
            put_text(frame, "Face/landmarks nao encontrados", (10, 20), color=(0, 0, 255))

        # Hints
        hint_y = 44
        if args.pose_diversity:
            put_text(frame, "Varie a pose: esquerda/direita/cima/baixo", (10, hint_y), color=(180, 220, 255)); hint_y += 20
        if args.capture_on_blink:
            put_text(frame, "Piscar para capturar", (10, hint_y), color=(180, 220, 255)); hint_y += 20

        # Policy
        now_ts = time.time()
        interval_ok = (now_ts - last_capture_ts) >= args.capture_interval
        blink_ok    = (not args.capture_on_blink) or blink_event
        face_ok     = bool(faces_rects)

        pose_ok = True
        if args.pose_diversity and (yaw is not None and pitch is not None):
            if last_yaw_pitch is None:
                pose_ok = True
            else:
                dyaw = abs(yaw - last_yaw_pitch[0])
                dpitch = abs(pitch - last_yaw_pitch[1])
                pose_ok = (dyaw >= args.yaw_thresh) or (dpitch >= args.pitch_thresh)

        should_capture = interval_ok and blink_ok and pose_ok and face_ok

        # Draw traffic light
        draw_semaforo(frame, should_capture, face_ok, interval_ok, blink_ok, pose_ok)

        # Capture
        if should_capture:
            roi = crop_face(gray, faces_rects[0], FACE_IMG_SIZE)
            if roi is not None:
                out = user_dir / f"{int(time.time()*1000)}.jpg"
                cv2.imwrite(str(out), roi)
                collected += 1
                db_inc_samples(user_id, 1)
                last_capture_ts = now_ts
                if yaw is not None and pitch is not None:
                    last_yaw_pitch = (yaw, pitch)
                put_text(frame, f"Amostra {collected}/{args.samples}", (10, hint_y + 10), color=(255, 255, 0))

        # FPS
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev))
        prev = now
        put_text(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10))

        cv2.imshow("ENROLL", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            save_frame(frame)
        if key == ord("q"):
            break
        if collected >= args.samples:
            break

    cap.release()
    cv2.destroyAllWindows()

    train_lbph_from_samples(FACE_IMG_SIZE, save=True)
    logging.info("ENROLL concluído: %s amostras capturadas de user_id=%s", collected, user_id)

# --------------- AUTH ---------------- #
def run_auth(args):
    ensure_dirs()
    setup_logging()
    db_init()

    face_cascade = load_haar_detector()
    facemark = load_facemark(args.model_path)
    lbph = load_or_create_lbph()

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir webcam índice {args.camera_index}")

    liveness = BlinkLiveness(args.blink_thresh, EAR_CONSEC_FRAMES_FOR_BLINK, LIVENESS_WINDOW_SEC, REQUIRE_BLINKS_IN_WINDOW)

    cv2.namedWindow("AUTH", cv2.WINDOW_NORMAL)
    logging.info("Controles: [S] salvar frame / [Q] sair")
    prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            logging.warning("Frame não capturado. Encerrando...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rects, faces_np = detect_faces(gray, face_cascade)
        pts_list = facemark_fit(gray, frame, facemark, faces_np, draw=True)

        if pts_list:
            ear = compute_ear_from_landmarks(pts_list[0])
            liveness.update(ear)
            put_text(frame, f"EAR: {ear:.3f}  blinks:{liveness.count()}", (10, 20))
        else:
            put_text(frame, "Face/landmarks nao encontrados", (10, 20), color=(0, 0, 255))

        nivel_concedido = 1
        obs = ""

        if len(faces_rects) == 1:
            face_img = crop_face(gray, faces_rects[0], FACE_IMG_SIZE)
            if face_img is not None:
                try:
                    label_pred, distance = predict_lbph(face_img, lbph)
                    if distance <= args.lbph_thresh and liveness.ok():
                        row = db_get_user_by_id(label_pred)
                        if row:
                            _, nome, nivel, *_ = row
                            nivel_concedido = int(nivel)
                            put_text(frame, f"Usuario: {nome}  Dist: {distance:.1f}", (10, 45), color=(0, 255, 255))
                            draw_rbac_overlay(frame, nivel_concedido)
                            obs = "ACESSO_CONCEDIDO"
                        else:
                            put_text(frame, f"Usuario desconhecido (id={label_pred})", (10, 45), color=(0, 0, 255))
                            obs = "DESCONHECIDO_ID"
                    else:
                        reason = []
                        if distance > args.lbph_thresh:
                            reason.append(f"dist>{args.lbph_thresh:.1f}({distance:.1f})")
                        if not liveness.ok():
                            reason.append("liveness_fail")
                        put_text(frame, "Acesso negado: " + ", ".join(reason), (10, 45), color=(0, 0, 255))
                        obs = "NEGADO_" + "_".join(reason) if reason else "NEGADO"
                    log_access(user_pred_id=label_pred, dist=distance,
                               liveness_ok=liveness.ok(), nivel_concedido=nivel_concedido, obs=obs)
                except cv2.error:
                    put_text(frame, "Modelo LBPH vazio. Execute ENROLL antes.", (10, 45), color=(0, 0, 255))
                    log_access(user_pred_id=None, dist=None, liveness_ok=liveness.ok(),
                               nivel_concedido=nivel_concedido, obs="LBPH_VAZIO")

        elif len(faces_rects) > 1:
            put_text(frame, "Multiplas faces detectadas - aproxime apenas 1 usuario", (10, 45), color=(0, 0, 255))
        else:
            put_text(frame, "Nenhuma face detectada", (10, 45), color=(0, 0, 255))

        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev))
        prev = now
        put_text(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10))

        cv2.imshow("AUTH", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            save_frame(frame)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------- ENTRYPOINT ----------- #
def main():
    args = parse_args()
    if args.cmd == "enroll":
        run_enroll(args)
    elif args.cmd == "auth":
        run_auth(args)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        setup_logging()
        logging.exception("ERRO: %s", e)
        raise
