import cv2
import sys
import os
import glob
import time
import threading
import torch
import numpy as np
from queue import Queue
from collections import deque
from datetime import datetime
from flask import Flask, Response
import atexit

# YOLOv5 imports
sys.path.append(os.path.abspath('yolov5'))
from models.common import DetectMultiBackend
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device

# Facenet-PyTorch for face detection & embedding
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.neighbors import KNeighborsClassifier

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
CAPTURE_PATH = r'\\10.0.0.1\photos'
SSL_CERT = os.path.join(BASE_DIR, 'ssl', 'cert.pem')
SSL_KEY = os.path.join(BASE_DIR, 'ssl', 'key.pem')

# Create directories if they don't exist
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CAPTURE_PATH, exist_ok=True)

class Config:
    """Configuration constants"""
    ALERT_DURATION = 5  # seconds
    ALERT_COOLDOWN = 5  # seconds
    CAPTURE_DELAY = 0.3  # seconds
    MIN_KNIFE_CONFIDENCE = 0.5
    MIN_KNIFE_SIZE = 50
    CAMERA_WIDTH = 800
    CAMERA_HEIGHT = 600
    CAMERA_FPS = 30
    FRAME_BUFFER_SIZE = 5

class State:
    """System state management"""
    def __init__(self):
        self.current = 'idle'
        self.last_person_time = 0
        self.last_face_recognition_time = 0
        self.alert_active = False
        self.alert_start_time = 0

class FaceRecognizer:
    """Handles face detection and recognition"""
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True, device=device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.classifier = None
        self._load_embeddings()

    def _load_embeddings(self):
        """Load face embeddings from dataset"""
        embeddings, labels = [], []
        
        for person_name in os.listdir(DATASET_PATH):
            person_folder = os.path.join(DATASET_PATH, person_name)
            if not os.path.isdir(person_folder):
                continue
                
            for img_path in glob.glob(os.path.join(person_folder, '*.jpg')):
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_rgb = img_rgb.copy()  # Fix negative stride
                    
                    faces = self.mtcnn(img_rgb)
                    if faces is None:
                        continue
                        
                    for face in faces:
                        if face is None:
                            continue
                            
                        if face.dim() == 3:
                            face = face.unsqueeze(0)
                        face = face.to(device)
                        
                        with torch.no_grad():
                            embed = self.resnet(face).cpu().numpy()[0]
                        embeddings.append(embed)
                        labels.append(person_name)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        if embeddings:
            self.classifier = KNeighborsClassifier(n_neighbors=1)
            self.classifier.fit(embeddings, labels)
        else:
            print("Warning: No face embeddings found in dataset")

    def recognize(self, frame):
        """Recognize faces in the given frame"""
        if self.classifier is None:
            return None, None
            
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = rgb.copy()  # Fix negative stride
            
            boxes, _ = self.mtcnn.detect(rgb)
            if boxes is None:
                return None, None
                
            x1, y1, x2, y2 = map(int, boxes[0])
            face_crop = rgb[y1:y2, x1:x2]
            
            face_tensor = self.mtcnn(face_crop)
            if face_tensor is None:
                return None, None
                
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
            face_tensor = face_tensor.to(device)
            
            with torch.no_grad():
                emb = self.resnet(face_tensor).cpu().numpy()[0]
            name = self.classifier.predict([emb])[0]
            return name, (x1, y1, x2, y2)
        except Exception as e:
            print(f"Face recognition error: {e}")
            return None, None

class ObjectDetector:
    """Base class for object detection using YOLOv5"""
    def __init__(self, model_path, img_size=640, conf_thresh=0.25, iou_thresh=0.45):
        self.model = DetectMultiBackend(model_path, device=device, dnn=False)
        self.stride = self.model.stride
        self.imgsz = check_img_size((img_size, img_size), s=self.stride)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

    def detect(self, frame):
        """Detect objects in frame"""
        try:
            img = letterbox(frame, self.imgsz, auto=True)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device).float() / 255.0
            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
                
            with torch.no_grad():
                pred = self.model(img, augment=False)
                
            pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh)
            return pred
        except Exception as e:
            print(f"Detection error: {e}")
            return None

class VideoStream:
    """Handles video streaming from camera"""
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Thread function to continuously capture frames"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.1)

    def get_frame(self):
        """Get the latest frame"""
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        """Stop the video stream"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

class KnifeDetector(threading.Thread):
    """Threaded knife detection system"""
    def __init__(self, detector):
        super().__init__(daemon=True)
        self.running = True
        self.current_frame = None
        self.lock = threading.Lock()
        self.detector = detector
        self.current_detection = None
        self.capture_counter = 0
        self.last_alert_time = 0
        self.last_capture_time = 0
        self.waiting_period = False
        self.waiting_start_time = 0
        self.knife_present = False
        self.annotated_frame = None  # Store the annotated frame

    def update_frame(self, frame):
        """Update the current frame for processing"""
        with self.lock:
            self.current_frame = frame.copy() if frame is not None else None

    def run(self):
        """Main detection loop"""
        while self.running:
            with self.lock:
                frame = self.current_frame
                
            if frame is not None:
                self._process_frame(frame)
            time.sleep(0.1)

    def _process_frame(self, frame):
        """Process a single frame for knife detection"""
        try:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
            pred = self.detector.detect(frame)
            current_time = time.time()
            
            if pred is not None:
                best_det = self._find_best_detection(pred, frame.shape)
                
                if best_det:
                    self._handle_detection(best_det, current_time, frame)
                    self.knife_present = True
                else:
                    self.current_detection = None
                    self.knife_present = False
                    
                self._handle_waiting_period(current_time, frame)
        except Exception as e:
            print(f"Knife detection error: {e}")

    def _handle_waiting_period(self, current_time, frame):
        """Handle the waiting period logic"""
        if self.waiting_period:
            if current_time - self.waiting_start_time >= 20:
                self.waiting_period = False
                if self.knife_present:
                    # Knife still present after 20 seconds, take 2 more pictures
                    self.capture_counter = 0
                    self._take_pictures(current_time, frame)
        elif self.knife_present and not self.waiting_period:
            # Initial knife detection, take 2 pictures and start waiting period
            self.capture_counter = 0
            self._take_pictures(current_time, frame)
            self.waiting_period = True
            self.waiting_start_time = current_time

    def _take_pictures(self, current_time, frame):
        """Take the specified number of pictures"""
        for _ in range(2):  # Take 2 pictures
            if self.capture_counter < 2:
                self.capture_counter += 1
                self.last_capture_time = current_time
                # Save the annotated frame if available, otherwise use the raw frame
                frame_to_save = self.annotated_frame.copy() if self.annotated_frame is not None else frame.copy()
                alert_queue.put(("knife", frame_to_save, self.current_detection))
                time.sleep(0.3)  # Small delay between pictures

    def _find_best_detection(self, pred, frame_shape):
        """Find the best detection from predictions"""
        best_det = None
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(self.detector.imgsz, det[:, :4], frame_shape).round()
                for *xyxy, conf, cls in det:
                    if conf > Config.MIN_KNIFE_CONFIDENCE:
                        x1, y1, x2, y2 = map(int, xyxy)
                        if (x2 - x1) > Config.MIN_KNIFE_SIZE and (y2 - y1) > Config.MIN_KNIFE_SIZE:
                            if best_det is None or conf > best_det[4]:
                                best_det = (x1, y1, x2, y2, float(conf))
        return best_det

    def _handle_detection(self, detection, current_time, frame):
        """Handle a knife detection event"""
        self.current_detection = detection
        
class SecuritySystem:
    """Main security system application"""
    def __init__(self):
        self.app = Flask(__name__)
        self.video_stream = None
        self.knife_detector = None
        self.face_recognizer = None
        self.person_detector = None
        self.state = State()
        self.frame_buffer = deque(maxlen=Config.FRAME_BUFFER_SIZE)
        
        self._setup_routes()
        atexit.register(self.cleanup)

    def _setup_routes(self):
        """Configure Flask routes"""
        self.app.route('/')(self._index)
        self.app.route('/video')(self._video_feed)

    def _index(self):
        """Home page with video stream"""
        return '''
        <html>
            <head>
                <title>Security System</title>
                <style>
                    body { margin: 0; background: #000; }
                    #video { width: 100%; height: auto; }
                </style>
            </head>
            <body>
                <img id="video" src="/video"/>
            </body>
        </html>
        '''

    def _video_feed(self):
        """Video streaming route"""
        return Response(self._generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _generate_frames(self):
        """Generate video frames with annotations"""
        while True:
            current_time = time.time()
            frame = self.video_stream.get_frame()
            
            if frame is None:
                time.sleep(0.1)
                continue
                
            # Create an annotated copy of the frame
            annotated_frame = frame.copy()
            self._process_frame(annotated_frame, current_time)
            self._update_state(current_time)
            
            # Store the annotated frame in the knife detector
            if self.knife_detector:
                with self.knife_detector.lock:
                    self.knife_detector.annotated_frame = annotated_frame
            
            ret, buf = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    def _process_frame(self, frame, current_time):
        """Process a single frame with all detectors"""
        if self.knife_detector:
            self.knife_detector.update_frame(frame)
            self._handle_knife_detections(current_time)
            self._draw_knife_detections(frame)
            
        self._draw_alert_status(frame, current_time)
        
        if current_time - self.state.last_face_recognition_time > 1:
            self._recognize_faces(frame)
            self.state.last_face_recognition_time = current_time
            
        self._draw_status_overlay(frame)

    def _handle_knife_detections(self, current_time):
        """Handle knife detection alerts"""
        if not alert_queue.empty():
            try:
                alert_item = alert_queue.get_nowait()
                if alert_item and len(alert_item) == 3 and alert_item[0] == "knife":
                    _, alert_frame, coords = alert_item
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    seq_num = self.knife_detector.capture_counter
                    
                    try:
                        # First try to save to network location
                        network_path = os.path.join(CAPTURE_PATH, f"knife_{ts}_seq{seq_num}.jpg")
                        cv2.imwrite(network_path, alert_frame)
                    except Exception as network_error:
                        print(f"Failed to save to network location: {network_error}")
                        # Fall back to local storage if network fails
                        local_path = os.path.join(BASE_DIR, 'captures', f"knife_{ts}_seq{seq_num}.jpg")
                        os.makedirs(os.path.join(BASE_DIR, 'captures'), exist_ok=True)
                        cv2.imwrite(local_path, alert_frame)
                        print(f"Saved to local path instead: {local_path}")
                    
                    if self.knife_detector.capture_counter == 0:
                        self.state.alert_active = True
                        self.state.alert_start_time = current_time
            except Exception as e:
                print(f"Error handling knife detection: {e}")

    def _draw_knife_detections(self, frame):
        """Draw knife detection boxes on frame"""
        if hasattr(self.knife_detector, 'current_detection') and self.knife_detector.current_detection:
            x1, y1, x2, y2, conf = self.knife_detector.current_detection
            box_height = y2 - y1
            y_offset = int(box_height * 0.2)
            y1 += y_offset
            y2 += y_offset
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'Knife {conf:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _draw_alert_status(self, frame, current_time):
        """Draw alert status on frame"""
        if self.state.alert_active:
            if current_time - self.state.alert_start_time < Config.ALERT_DURATION:
                alert_text = "ALERT: Knife Detected!"
                text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = frame.shape[1] - text_size[0] - 20
                cv2.putText(frame, alert_text, (text_x, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                self.state.alert_active = False

    def _update_state(self, current_time):
        """Update system state machine"""
        if self.state.current == 'idle':
            if current_time - self.state.last_person_time > 1:
                if self.person_detector and self.person_detector.detect(self.video_stream.get_frame()):
                    self.state.current = 'active'
                    self.state.last_person_time = current_time
        else:
            if current_time - self.state.last_person_time > 10:
                self.state.current = 'idle'

    def _recognize_faces(self, frame):
        """Perform face recognition on frame"""
        name, box = self.face_recognizer.recognize(frame)
        if name and box:
            color = (255, 0, 0) if name == 'Karim' else (0, 255, 255)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _draw_status_overlay(self, frame):
        """Draw system status overlay on frame"""
        cv2.putText(frame, f"State: {self.state.current}", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, datetime.now().strftime('%H:%M:%S'), 
                  (frame.shape[1] - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def initialize(self):
        """Initialize all system components"""
        try:
            # Initialize models
            knife_detector_model = ObjectDetector(
                os.path.join(BASE_DIR, 'yolov5', 'runs', 'train', 'knife_detector', 'weights', 'best.pt'),
                conf_thresh=0.25,
                iou_thresh=0.45
            )
            
            self.person_detector = ObjectDetector(
                os.path.join(BASE_DIR, 'yolov5', 'runs', 'train', 'person_detector', 'weights', 'best.pt'),
                conf_thresh=0.4,
                iou_thresh=0.45
            )
            
            self.face_recognizer = FaceRecognizer()
            
            # Initialize video stream and detectors
            self.video_stream = VideoStream(0)
            self.knife_detector = KnifeDetector(knife_detector_model)
            self.knife_detector.start()
            
            return True
        except Exception as e:
            print(f"Initialization error: {e}")
            return False

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources...")
        if self.video_stream:
            self.video_stream.stop()
        if self.knife_detector:
            self.knife_detector.running = False
            if self.knife_detector.is_alive():
                self.knife_detector.join()
        cv2.destroyAllWindows()

    def run(self):
        """Run the application"""
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

# Global variables
device = select_device('0')
alert_queue = Queue()

if __name__ == '__main__':
    security_system = SecuritySystem()
    if not security_system.initialize():
        print("Failed to initialize security system", file=sys.stderr)
        sys.exit(1)
        
    if not security_system.run():
        sys.exit(1)