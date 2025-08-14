import cv2
import mediapipe as mp
import torch
from ultralytics import YOLO
import face_recognition
import numpy as np
import time
import math

# Configurações
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
YOLO_CONF_THRESHOLD = 0.45
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# MediaPipeDetector (sem mudanças)

class MediaPipeDetector:
    def __init__(self, face_conf=0.7, hand_conf=0.7):
        # Inicializa Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=hand_conf,
            min_tracking_confidence=0.5
        )

        # Inicializa FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Mais precisão nos olhos/boca
            min_detection_confidence=face_conf,
            min_tracking_confidence=0.5
        )

        # Utilitário para desenhar
        self.mp_draw = mp.solutions.drawing_utils

        # Índices dos olhos (EAR)
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

        # Variáveis para piscadas
        self.blink_threshold = 0.25
        self._closed_frames = 0
        self.consec_frames = 2
        self.blink_count = 0
        self._last_blink_time = 0
        self.blink_display_time = 0.8
        self.blinking = False

    # -------------------- Dedos levantados --------------------
    def _fingers_up(self, hand_landmarks, hand_label):
        tips_ids = [4, 8, 12, 16, 20]
        fingers = []

        if not hand_landmarks:
            return [0, 0, 0, 0, 0]

        # Polegar
        if hand_label == "Right":
            fingers.append(int(hand_landmarks.landmark[tips_ids[0]].x <
                               hand_landmarks.landmark[tips_ids[0] - 1].x))
        else:  # Left
            fingers.append(int(hand_landmarks.landmark[tips_ids[0]].x >
                               hand_landmarks.landmark[tips_ids[0] - 1].x))

        # Outros dedos
        for i in range(1, 5):
            fingers.append(int(hand_landmarks.landmark[tips_ids[i]].y <
                                hand_landmarks.landmark[tips_ids[i] - 2].y))

        return fingers

    # -------------------- Cálculos EAR --------------------
    def _euclidean_dist(self, p1, p2):
        return math.dist([p1.x, p1.y], [p2.x, p2.y])

    def _eye_aspect_ratio(self, landmarks, eye_indices):
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
        vertical_1 = self._euclidean_dist(p2, p6)
        vertical_2 = self._euclidean_dist(p1, p5)
        horizontal = self._euclidean_dist(p3, p4)
        if horizontal == 0:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    # -------------------- Processamento completo --------------------
    def process(self, frame, draw_on_frame=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar mãos
        hands_results = self.hands.process(rgb_frame)
        fingers_list = []
        hands_info = []

        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for hand_landmarks, hand_handedness in zip(hands_results.multi_hand_landmarks,
                                                       hands_results.multi_handedness):
                hand_label = hand_handedness.classification[0].label
                fingers = self._fingers_up(hand_landmarks, hand_label)
                fingers_list.append({"label": hand_label, "fingers": fingers})
                hands_info.append({'count': sum(fingers), 'fingers': fingers})

                if draw_on_frame:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Processar face / piscadas
        face_results = self.face_mesh.process(rgb_frame)
        face_present = False
        avg_ear = None
        now = time.time()

        if face_results.multi_face_landmarks:
            face_present = True
            for face_landmarks in face_results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE_IDX)
                right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE_IDX)
                avg_ear = (left_ear + right_ear) / 2.0

                # Lógica piscada
                if avg_ear < self.blink_threshold:
                    self._closed_frames += 1
                else:
                    if self._closed_frames >= self.consec_frames:
                        self.blink_count += 1
                        self._last_blink_time = now
                    self._closed_frames = 0

                self.blinking = (now - self._last_blink_time) < self.blink_display_time

                if draw_on_frame:
                    self.mp_draw.draw_landmarks(frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)
                    cv2.putText(frame, f'EAR: {avg_ear:.2f}', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                    cv2.putText(frame, f'Piscadas: {self.blink_count}', (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        return {
            'frame': frame,
            'hands_info': hands_info,
            'fingers_up': fingers_list,
            'face_present': face_present,
            'avg_ear': avg_ear,
            'blink_count': self.blink_count,
            'blinking': self.blinking
        }

# FaceRecognitionModule e YOLODetector você pode manter igual se quiser.

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Câmera não encontrada.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    mp_detector = MediaPipeDetector()
    # yolo_detector = YOLODetector(model_name='yolov8n.pt', device=DEVICE, conf=YOLO_CONF_THRESHOLD)
    # face_recog = FaceRecognitionModule()

    # Apenas UMA janela
    window_name = "Detecçãoq Unificada"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fps_smoother = 0.0
    prev_time = time.time()

    print("[INFO] Iniciando. Pressione 'q' para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Falha ao ler frame da câmera.")
            break

        mp_out = mp_detector.process(frame, draw_on_frame=True)
        frame_proc = mp_out['frame']

        # Desenhar contagem dedos e piscadas
        for i, hand_info in enumerate(mp_out['hands_info']):
            cv2.putText(frame_proc, f'Mao {i+1}: {hand_info["count"]} dedos', (10, 130 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if mp_out['blinking']:
            cv2.rectangle(frame_proc, (0,0), (frame_proc.shape[1], 50), (0,0,255), -1)
            cv2.putText(frame_proc, "PISCADA DETECTADA!", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        prev_time = now
        fps_smoother = (fps_smoother*0.9) + (fps*0.1)
        cv2.putText(frame_proc, f'FPS: {fps_smoother:.1f}', (10, frame_proc.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow(window_name, frame_proc)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(f'capture_{ts}.jpg', frame_proc)
            print(f"[INFO] Salvou capture_{ts}.jpg")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
