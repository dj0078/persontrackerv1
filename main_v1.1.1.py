import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Paths setup
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, 'data', 'SampleVideo1.mp4')
print("Using absolute path:", video_path)

# Prepare output directory and video writer
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, 'output.mp4')

# Open input video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Error: Could not open video at {video_path}")

# Video writer properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# Load YOLO model
model = YOLO("yolov8n.pt")
device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
model.to(device)

# COCO class labels
COCO_CLASSES = model.names

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

    # 1) Detect and draw bounding boxes + confidence
    detections = model(frame)[0]
    person_detections = []
    for box in detections.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Draw detection bounding box and confidence
        label = COCO_CLASSES[cls_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Register person detections for tracking
        if cls_id == 0 and conf > 0.6:
            person_detections.append(((x1, y1, x2, y2), conf, 'person'))

    # 2) Update tracker
    clean_frame = frame.copy()
    tracks = tracker.update_tracks(person_detections, frame=clean_frame)

    # 3) Overlay ID labels without boxes
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        text_x = x1
        text_y = y1 - 15 if y1 - 15 > 15 else y1 + 15
        cv2.putText(frame, f"ID:{track_id}", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 4) Write to output and show
    writer.write(frame)
    cv2.imshow('YOLO + DeepSORT Tracking', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Saved output video to {out_path}")
