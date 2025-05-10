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

# Load YOLO model
model = YOLO("yolov8n.pt")
model.to('cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')

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

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes
    person_detections = []

    # For tracking — unmodified frame
    clean_frame = frame.copy()

    # Process detections
    for box in boxes:
        # FIX: Use tolist() to avoid bad box shape
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = COCO_CLASSES[cls_id]

        # Draw ALL detections in green
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Register person detection for tracking
        if cls_id == 0 and conf > 0.6:
            person_detections.append(((x1, y1, x2, y2), conf, 'person'))

    # Update tracker with unannotated frame
    tracks = tracker.update_tracks(person_detections, frame=clean_frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Match tracker with YOLO box for accuracy
        best_iou = 0
        best_box = None
        for ((bx1, by1, bx2, by2), _, _) in person_detections:
            inter_x1 = max(x1, bx1)
            inter_y1 = max(y1, by1)
            inter_x2 = min(x2, bx2)
            inter_y2 = min(y2, by2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area_a = (x2 - x1) * (y2 - y1)
            area_b = (bx2 - bx1) * (by2 - by1)
            union_area = area_a + area_b - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_box = (bx1, by1, bx2, by2)

        # Use tighter box if available
        if best_iou > 0.3 and best_box:
            x1, y1, x2, y2 = best_box

        # Only draw the ID label, NOT another box
        cv2.putText(frame, f'ID: {track_id}', (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show result
    cv2.imshow('YOLO + DeepSORT Tracking', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
