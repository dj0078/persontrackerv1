import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Paths setup
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, 'data', 'SampleVideo1.mp4')
print("Using absolute path:", video_path)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Error: Could not open video at {video_path}")

# Load YOLO face‐detection model
# Make sure you've downloaded yolov8n-face.pt (Ultralytics face model)
model = YOLO("yolov8n-face.pt")
device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
model.to(device)

# Class names will be ['face']
FACE_CLASS = 0
CONF_THRESHOLD = 0.6

# DeepSort tracker
tracker = DeepSort(
    max_age=50,
    n_init=5,
    embedder='mobilenet',
    max_cosine_distance=0.3,
    nn_budget=200
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Run face detection
    results = model(frame)[0]
    face_detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Draw face bounding box + confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"face {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Collect for tracking if above threshold
        if cls_id == FACE_CLASS and conf > CONF_THRESHOLD:
            face_detections.append(((x1, y1, x2, y2), conf, 'face'))

    # 2) Update DeepSORT with current detections
    tracks = tracker.update_tracks(face_detections, frame=frame)

    # 3) Draw only ID labels (no extra boxes)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        text_x = x1
        text_y = y1 - 15 if y1 - 15 > 15 else y1 + 15
        cv2.putText(frame, f"ID:{track_id}", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 4) Display
    cv2.imshow('Face Tracking with YOLO + DeepSORT', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
