import cv2
import mediapipe as mp
import torch
from ultralytics import YOLO
import face_recognition
import numpy as np
import time
import math
import sys

# ------------------ CONFIGURAÇÕES ------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
YOLO_CONF_THRESHOLD = 0.45      # confiança mínima para desenhar box YOLO
FRAME_WIDTH = 640               # opcional: redimensionar para performance
FRAME_HEIGHT = 480

# ------------------ DETECÇÃO COM MEDIA PIPE ------------------
class MediaPipeDetector:
    def __init__(self, face_conf=0.7, hand_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2,
                                         min_detection_confidence=hand_conf,
                                         min_tracking_confidence=0.5)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1,
                                                    min_detection_confidence=face_conf,
                                                    min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

        self.blink_threshold = 0.23
        self.prev_blink_time = 0
        self.blink_cooldown = 0.45
        self.blink_count = 0
        self.blinking = False

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

    def _fingers_up(self, hand_landmarks):
        tips_ids = [4, 8, 12, 16, 20]
        fingers = []
        try:
            fingers.append(int(hand_landmarks.landmark[tips_ids[0]].x <
                               hand_landmarks.landmark[tips_ids[0] - 1].x))
            for i in range(1, 5):
                fingers.append(int(hand_landmarks.landmark[tips_ids[i]].y <
                                    hand_landmarks.landmark[tips_ids[i] - 2].y))
        except Exception:
            return [0, 0, 0, 0, 0]
        return fingers

    def process(self, frame, draw_on_frame=True):
        self.blinking = False
        frame_out = frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands_results = self.hands.process(img_rgb)
        hands_info = []
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                fingers = self._fingers_up(hand_landmarks)
                count_fingers = sum(fingers)
                hands_info.append({'count': count_fingers, 'fingers': fingers})
                if draw_on_frame:
                    self.mp_draw.draw_landmarks(frame_out, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        face_results = self.face_mesh.process(img_rgb)
        face_present = False
        avg_ear = None
        if face_results.multi_face_landmarks:
            face_present = True
            for face_landmarks in face_results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE_IDX)
                right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE_IDX)
                avg_ear = (left_ear + right_ear) / 2.0
                now = time.time()
                if avg_ear is not None and avg_ear < self.blink_threshold and (now - self.prev_blink_time) > self.blink_cooldown:
                    self.blink_count += 1
                    self.prev_blink_time = now
                    self.blinking = True
                if draw_on_frame:
                    self.mp_draw.draw_landmarks(frame_out, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)
                    if avg_ear is not None:
                        cv2.putText(frame_out, f'EAR: {avg_ear:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                        cv2.putText(frame_out, f'Piscadas: {self.blink_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        return {
            'frame': frame_out,
            'hands_info': hands_info,
            'face_present': face_present,
            'avg_ear': avg_ear,
            'blink_count': self.blink_count,
            'blinking': self.blinking
        }

# ------------------ YOLO (Ultralytics) ------------------
class YOLODetector:
    def __init__(self, model_name='yolov8n.pt', device=DEVICE, conf=YOLO_CONF_THRESHOLD):
        try:
            self.model = YOLO(model_name)
            self.model.to(device)
            self.model.conf = conf
            self.device = device
        except Exception as e:
            print("[ERRO] Não foi possível carregar YOLO (ultralytics).", e)
            print("Tente: pip install ultralytics")
            raise

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        if len(results) == 0:
            return np.empty((0,6))
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.empty((0,6))
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        out = np.concatenate([xyxy, confs.reshape(-1,1), cls.reshape(-1,1)], axis=1)
        return out

# ------------------ RECONHECIMENTO FACIAL (face_recognition) ------------------
class FaceRecognitionModule:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_known_faces(self, images_with_names):
        for img_path, name in images_with_names:
            try:
                img = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(img)
                if len(encs) == 0:
                    print(f"[WARN] Nenhum rosto encontrado em {img_path}")
                    continue
                encoding = encs[0]
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)
                print(f"[INFO] Carregou rosto: {name}")
            except Exception as e:
                print(f"[ERRO] Ao carregar {img_path}: {e}")

    def recognize(self, frame, upsample_times=1):
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=upsample_times)
        encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
        names = []
        for encoding in encodings:
            name = "Desconhecido"
            if len(self.known_face_encodings) > 0:
                matches = face_recognition.compare_faces(self.known_face_encodings, encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(self.known_face_encodings, encoding)
                best_idx = np.argmin(face_distances)
                if matches[best_idx]:
                    name = self.known_face_names[best_idx]
            names.append(name)
        return face_locations, names

# ------------------ UTIL ------------------
def draw_yolo_boxes(frame, boxes, label_names=None):
    h, w = frame.shape[:2]
    for b in boxes:
        x1, y1, x2, y2, conf, cls = b
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = float(conf)
        cls = int(cls)
        if conf < YOLO_CONF_THRESHOLD:
            continue
        color = (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f'{label_names[cls] if label_names is not None else cls}:{conf:.2f}'
        cv2.putText(frame, label, (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ------------------ MAIN ------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Câmera não encontrada.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    mp_detector = MediaPipeDetector()
    yolo_detector = YOLODetector(model_name='yolov8n.pt', device=DEVICE, conf=YOLO_CONF_THRESHOLD)
    face_recog = FaceRecognitionModule()

    # Exemplo de como carregar faces conhecidas
    # face_recog.load_known_faces([("fotos/pedro1.jpg","Pedro"), ("fotos/amigo.jpg","Amigo")])

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Detecção", cv2.WINDOW_NORMAL)

    coco_names = yolo_detector.model.names if hasattr(yolo_detector.model, 'names') else None

    fps_smoother = 0.0
    prev_time = time.time()

    print("[INFO] Iniciando. Pressione 'q' para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Falha ao ler frame da câmera.")
            break

        frame_orig = frame.copy()

        mp_out = mp_detector.process(frame.copy(), draw_on_frame=True)
        frame_proc = mp_out['frame']

        boxes = yolo_detector.detect(frame_orig)
        if boxes.size != 0:
            draw_yolo_boxes(frame_proc, boxes, label_names=coco_names)

        face_locations, face_names = face_recog.recognize(frame_orig)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame_proc, (left, top), (right, bottom), (0, 255, 255), 2)
            cv2.putText(frame_proc, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if mp_out['blinking']:
            cv2.rectangle(frame_proc, (0,0), (frame_proc.shape[1], 50), (0,0,255), -1)
            cv2.putText(frame_proc, "PISCADA DETECTADA!", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        prev_time = now
        fps_smoother = (fps_smoother*0.9) + (fps*0.1)
        cv2.putText(frame_proc, f'FPS: {fps_smoother:.1f}', (10, frame_proc.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Original", frame_orig)
        cv2.imshow("Detecção", frame_proc)

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
