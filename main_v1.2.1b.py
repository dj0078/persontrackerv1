import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# — Paths setup —
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, 'data', 'SampleVideo1.mp4')
print("Using absolute path:", video_path)

# — Open video & get FPS for dwell-time —
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Error: Could not open video at {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS)
dwell_frame_counts = {}  # track_id -> frames seen

# — Load YOLO person-only model —
model = YOLO("yolov8n.pt")
device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
model.to(device)

# — DeepSORT tracker —
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

    # — 1) Detect persons only (class 0) —
    results = model(frame, classes=[0], conf=0.6)[0]

    person_detections = []
    for box in results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # draw green box + confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,
                    f"person {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        # register for tracking
        person_detections.append(((x1, y1, x2, y2), conf, 'person'))

    # — 2) Update DeepSORT —
    tracks = tracker.update_tracks(person_detections, frame=frame)

    # — 3) Draw ID + dwell time (red text only) —
    for track in tracks:
        if not track.is_confirmed():
            continue

        tid = track.track_id
        dwell_frame_counts[tid] = dwell_frame_counts.get(tid, 0) + 1
        deci = round(dwell_frame_counts[tid] * 10 / fps)
        dwell_s = deci / 10.0

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        tx, ty = x1, (y1 - 15 if y1 - 15 > 15 else y1 + 15)

        cv2.putText(frame,
                    f"ID:{tid} {dwell_s:.1f}s",
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

    # — 4) Show frame —
    cv2.imshow('Person Tracking Only', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
