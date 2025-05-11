import os
import cv2
import csv
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Paths setup
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, 'data', 'SampleVideo1.mp4')
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
out_video_path = os.path.join(output_dir, 'output.mp4')
csv_path = os.path.join(output_dir, 'dwell_times.csv')

print("Using absolute path:", video_path)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Error: Could not open video at {video_path}")

# Get FPS and frame size
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

# Dwell counting
dwell_frame_counts = {}  # track_id -> frames seen

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

    # 1) Detect
    detections = model(frame)[0]
    person_dets = []
    for box in detections.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # draw detection box + conf
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{COCO_CLASSES[cls_id]} {conf:.2f}",
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        if cls_id == 0 and conf > 0.6:
            person_dets.append(((x1, y1, x2, y2), conf, 'person'))

    # 2) Track
    tracks = tracker.update_tracks(person_dets, frame=frame)

    # 3) Overlay ID + dwell time
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        dwell_frame_counts[tid] = dwell_frame_counts.get(tid, 0) + 1

        # nearest decisecond
        deci = round(dwell_frame_counts[tid] * 10 / fps)
        dwell_s = deci / 10.0

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        ty = y1-15 if y1-15 > 15 else y1+15
        cv2.putText(frame,
                    f"ID:{tid} {dwell_s:.1f}s",
                    (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,255),
                    2)

    # 4) Write & display
    writer.write(frame)
    cv2.imshow('YOLO + DeepSORT Tracking', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# Clean up video
cap.release()
writer.release()
cv2.destroyAllWindows()

# Write CSV of final dwell times
with open(csv_path, 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['ID', 'DwellTime_s'])
    for tid, frames in dwell_frame_counts.items():
        deci = round(frames * 10 / fps)
        dwell_s = deci / 10.0
        csv_writer.writerow([tid, f"{dwell_s:.1f}"])

print(f"Saved output video to {out_video_path}")
print(f"Saved dwell times CSV to {csv_path}")