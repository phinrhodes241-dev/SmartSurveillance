import cv2
import sys
import os
import glob
import time
import threading
import torch
import numpy as np
from collections import deque
from datetime import datetime
from flask import Flask, Response, jsonify
import atexit
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.neighbors import KNeighborsClassifier

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
CAPTURE_PATH = r'\\10.0.0.1\photos'
SSL_CERT = os.path.join(BASE_DIR, 'ssl', 'cert.pem')
SSL_KEY = os.path.join(BASE_DIR, 'ssl', 'key.pem')

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CAPTURE_PATH, exist_ok=True)


class Config:
    MIN_KNIFE_CONFIDENCE = 0.4
    MIN_PERSON_CONFIDENCE = 0.5
    MIN_KNIFE_SIZE = 20
    CAMERA_WIDTH = 480
    CAMERA_HEIGHT = 320
    FRAME_PROCESS_INTERVAL = 0.1  
    STREAM_WIDTH = 480
    STREAM_HEIGHT = 320
    STREAM_QUALITY = 65          
    STREAM_FPS = 30 
    ALERT_DURATION = 2         
    CLIMBING_ALERT_DELAY = 3     
    CLIMBING_COOLDOWN = 10       
    ALERT_COOLDOWN = 5


class State:
    def __init__(self):
        self.last_face_recognition_time = 0
        self.climbing_alert_start = None
        self.climbing_alert_cooldown = 0
        self.climbing_alert_active = False
        self.knife_alert_active = False


class FaceRecognizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.classifier = None
        self._load_embeddings()

    def _load_embeddings(self):
        embeddings, labels = [], []
        for person_name in os.listdir(DATASET_PATH):
            person_folder = os.path.join(DATASET_PATH, person_name)
            if not os.path.isdir(person_folder): continue
            for img_path in glob.glob(os.path.join(person_folder, '*.jpg')):
                try:
                    img = cv2.imread(img_path)
                    if img is None: continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = self.mtcnn(img_rgb)
                    if faces is None: continue
                    for face in faces:
                        face = face.unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            embed = self.resnet(face).cpu().numpy()[0]
                        embeddings.append(embed)
                        labels.append(person_name)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        if embeddings:
            self.classifier = KNeighborsClassifier(n_neighbors=1)
            self.classifier.fit(embeddings, labels)
        else:
            print("Warning: No face embeddings found")

    def recognize(self, frame):
        if self.classifier is None:
            return None, None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = self.mtcnn.detect(rgb)
            if boxes is None:
                return None, None
            x1, y1, x2, y2 = map(int, boxes[0])
            face_crop = rgb[y1:y2, x1:x2]
            face_tensor = self.mtcnn(face_crop)
            if face_tensor is None:
                return None, None
            if face_tensor.dim() == 5:
                face_tensor = face_tensor.squeeze(0)
            face_tensor = face_tensor.to(self.device)
            with torch.no_grad():
                emb = self.resnet(face_tensor).cpu().numpy()
            name = self.classifier.predict(emb)[0]
            return name, (x1, y1, x2, y2)
        except Exception as e:
            print(f"Face recognition error: {e}")
            return None, None


class YOLOv8Detector:
    def __init__(self, model_path, conf=0.4, iou=0.45):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.knife_class_id = None
        for k, v in self.model.names.items():
            if v.lower() == 'knife':
                self.knife_class_id = k
                break

    def detect(self, frame):
        results = self.model(frame, verbose=False, conf=self.conf, iou=self.iou)
        boxes = []
        for result in results:
            bboxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, score, cls in zip(bboxes, scores, classes):
                boxes.append((box[0], box[1], box[2], box[3], float(score), int(cls)))
        return boxes


class PoseAnalyzer:
    def __init__(self):
        self.pose_model = YOLO("yolov8s-pose.pt")

    def analyze_pose(self, frame):
        results = self.pose_model(frame, verbose=False, conf=0.5, iou=0.45)
        climbing_detections = []
        for result in results:
            if hasattr(result, "keypoints"):
                for kps in result.keypoints.data.cpu().numpy():
                    climbing_detections.append(self._is_climbing(kps))
        return climbing_detections

    def _is_climbing(self, keypoints):
        try:
            if len(keypoints.shape) < 2 or keypoints.shape[0] < 17:
                return False
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            if left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]:
                return True
            return False
        except Exception as e:
            print(f"[ERROR] Failed to analyze pose: {e}")
            return False


class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()


class KnifeDetector(threading.Thread):
    def __init__(self, detector, security_system=None):
        super().__init__(daemon=True)
        self.running = True
        self.current_frame = None
        self.lock = threading.Lock()
        self.detector = detector
        self.current_detection = None
        self.alert_sent = False
        self.security_system = security_system
        self.screenshot_count = 0
        self.max_screenshots = 5
        self.cooldown_period = 300  # 5 minutes
        self.last_alert_time = 0

    def update_frame(self, frame):
        with self.lock:
            self.current_frame = frame.copy() if frame is not None else None

    def run(self):
        while self.running:
            with self.lock:
                frame = self.current_frame
            if frame is not None:
                self._process_frame(frame)
            time.sleep(Config.FRAME_PROCESS_INTERVAL)

    def _process_frame(self, frame):
        pred = self.detector.detect(frame)
        best_knife = None
        for det in pred:
            x1, y1, x2, y2, conf, cls_id = det
            if int(cls_id) == 0:
                continue
            if self.detector.knife_class_id is not None and cls_id == self.detector.knife_class_id and conf >= Config.MIN_KNIFE_CONFIDENCE:
                width = x2 - x1
                height = y2 - y1
                if width > Config.MIN_KNIFE_SIZE and height > Config.MIN_KNIFE_SIZE:
                    best_knife = (int(x1), int(y1), int(x2), int(y2), float(conf))
                    break

        current_time = time.time()

        if best_knife:
            # Reset counter if cooldown has passed
            if (current_time - self.last_alert_time) > self.cooldown_period:
                self.screenshot_count = 0

            if self.screenshot_count < self.max_screenshots:
                print(f"[ALERT] Knife detected! Taking screenshot {self.screenshot_count + 1}/{self.max_screenshots}")
                if self.security_system:
                    self.security_system._take_screenshot(best_knife)
                self.screenshot_count += 1
                if self.screenshot_count == 1:
                    self.last_alert_time = current_time
        else:
            self.alert_sent = False

        self.current_detection = best_knife


class SecuritySystem:
    def __init__(self):
        self.app = Flask(__name__)
        self.video_stream = None
        self.knife_detector = None
        self.face_recognizer = None
        self.person_detector = None
        self.pose_analyzer = None
        self.state = State()
        self._setup_routes()
        atexit.register(self.cleanup)

    def _setup_routes(self):
        self.app.route('/')(self._index)
        self.app.route('/video')(self._video_feed)
        self.app.route('/screenshot')(self._capture_screenshot_route)

    def _take_screenshot(self, knife_detection=None):
        frame = self.video_stream.get_frame()
        if frame is None:
            print("[ERROR] Could not capture frame for screenshot.")
            return None
        annotated_frame = frame.copy()
        current_time = time.time()
        self._process_frame(annotated_frame, current_time)
        if knife_detection:
            x1, y1, x2, y2, conf = knife_detection
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_frame, f'Knife {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Add alert text without the red border
            cv2.putText(annotated_frame, "ALERT: KNIFE DETECTED!", 
                       (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 0, 255), 2)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(CAPTURE_PATH, f"capture_{timestamp}.jpg")
        try:
            cv2.imwrite(filename, annotated_frame)
            print(f"[INFO] Annotated screenshot saved to: {filename}")
            return filename
        except Exception as e:
            print(f"[ERROR] Failed to save screenshot: {e}")
            return None

    def _capture_screenshot_route(self):
        path = self._take_screenshot()
        if path:
            return {"status": "success", "path": path}
        else:
            return {"status": "error"}, 500

    def _index(self):
        return '''
        <html><body><img src="/video" width="100%"/></body></html>
        '''

    def _video_feed(self):
        return Response(self._generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _generate_frames(self):
        while True:
            frame = self.video_stream.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            annotated_frame = frame.copy()
            current_time = time.time()
            self._process_frame(annotated_frame, current_time)
            cv2.putText(annotated_frame, datetime.now().strftime('%H:%M:%S'),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            ret, buf = cv2.imencode('.jpg', annotated_frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    def _process_frame(self, frame, current_time):
        self.knife_detector.update_frame(frame)
        if self.knife_detector.current_detection:
            x1, y1, x2, y2, conf = self.knife_detector.current_detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'Knife {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if self.person_detector:
            person_detections = self.person_detector.detect(frame)
            for det in person_detections:
                x1, y1, x2, y2, conf, cls_id = det
                if int(cls_id) == 0 and conf > Config.MIN_PERSON_CONFIDENCE:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if current_time - self.state.last_face_recognition_time > 2:
            name, box = self.face_recognizer.recognize(frame)
            if name and box:
                if name == 'Kholy':
                    color = (0, 255, 0)
                elif name == 'Karim':
                    color = (255, 0, 0)
                elif name == 'Mahmoud':
                    color = (255, 0, 255)
                else:
                    color = (0, 255, 255)
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            self.state.last_face_recognition_time = current_time

        if self.pose_analyzer:
            poses = self.pose_analyzer.analyze_pose(frame)
            for pose in poses:
                if pose:
                    if not self.state.climbing_alert_active:
                        print("[ALERT] Suspicious climbing posture detected!")
                        self.state.climbing_alert_active = True
                else:
                    self.state.climbing_alert_active = False

    def initialize(self):
        try:
            knife_model_path = os.path.join(BASE_DIR, 'models', 'knife', 'best.pt')
            person_model_path = os.path.join(BASE_DIR, 'models', 'yolov8', 'yolov8s.pt')
            print("Initializing detectors...")
            print(f"Loading knife model from: {knife_model_path}")
            knife_detector = YOLOv8Detector(knife_model_path, conf=0.5)
            self.knife_detector = KnifeDetector(knife_detector, security_system=self)
            self.knife_detector.start()
            print(f"Loading person model from: {person_model_path}")
            self.person_detector = YOLOv8Detector(person_model_path, conf=0.5)
            print("Loading face recognizer...")
            self.face_recognizer = FaceRecognizer()
            print("Loading pose analyzer...")
            self.pose_analyzer = PoseAnalyzer()
            print("Starting video stream...")
            self.video_stream = VideoStream(0)
            print("Initialization complete")
            return True
        except Exception as e:
            print(f"Initialization error: {e}")
            return False

    def cleanup(self):
        print("Cleaning up resources...")
        if self.video_stream:
            self.video_stream.stop()
        if self.knife_detector and self.knife_detector.is_alive():
            self.knife_detector.join()
        cv2.destroyAllWindows()

    def run(self):
        if not os.path.exists(SSL_CERT) or not os.path.exists(SSL_KEY):
            print("SSL certificates missing", file=sys.stderr)
            return False
        try:
            self.app.run(
                host='0.0.0.0',
                port=5000,
                ssl_context=(SSL_CERT, SSL_KEY),
                threaded=True,
                debug=False
            )
            return True
        except Exception as e:
            print(f"Failed to start server: {e}", file=sys.stderr)
            return False


if __name__ == '__main__':
    security_system = SecuritySystem()
    if not security_system.initialize():
        print("Failed to initialize system", file=sys.stderr)
        sys.exit(1)
    if not security_system.run():
        sys.exit(1)