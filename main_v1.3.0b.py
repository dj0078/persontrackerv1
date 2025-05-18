import os
import cv2
import csv
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# — Paths setup —
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, 'data', 'SampleVideo1.mp4')
output_dir = os.path.join(script_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
out_video_path = os.path.join(output_dir, 'output_combined.mp4')
csv_path = os.path.join(output_dir, 'dwell_times_person.csv')

print("Using absolute path:", video_path)

# — Open video & get FPS for dwell-time calculation —
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Error: Could not open video at {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS)
dwell_frame_counts = {}  # person_track_id → number of frames seen

# — Prepare video writer (from v1.2.1a) —
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

# — Load YOLO models —
device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'

# Person detector (COCO)
model_person = YOLO("yolov8n.pt")
model_person.to(device)

# Face detector
model_face = YOLO("yolov8n-face.pt")
model_face.to(device)

# — DeepSort trackers —
person_tracker = DeepSort(
    max_age=50,
    n_init=5,
    embedder='mobilenet',
    max_cosine_distance=0.3,
    nn_budget=200
)
face_tracker = DeepSort(
    max_age=50,
    n_init=5,
    embedder='mobilenet',
    max_cosine_distance=0.3,
    nn_budget=200
)

# — Detection thresholds —
PERSON_CLASS = 0     # in COCO: 'person'
FACE_CLASS   = 0     # in face model: 'face'
PERSON_CONF_THRESHOLD = 0.6
FACE_CONF_THRESHOLD   = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # — 1) Person detection —
    per_results = model_person(frame, classes=[PERSON_CLASS], conf=PERSON_CONF_THRESHOLD)[0]
    person_detections = []
    for box in per_results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,
                    f"person {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        person_detections.append(((x1, y1, x2, y2), conf, 'person'))

    # — 2) Face detection —
    face_results = model_face(frame, conf=FACE_CONF_THRESHOLD)[0]
    face_detections = []
    for box in face_results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,
                    f"face {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2)
        if conf > FACE_CONF_THRESHOLD:
            face_detections.append(((x1, y1, x2, y2), conf, 'face'))

    # — 3) Update trackers —
    person_tracks = person_tracker.update_tracks(person_detections, frame=frame)
    face_tracks   = face_tracker.update_tracks(face_detections,   frame=frame)

    # — 4) Draw person IDs + dwell time —
    for track in person_tracks:
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

    # — 5) Draw face FIDs —
    for track in face_tracks:
        if not track.is_confirmed():
            continue
        fid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        tx, ty = x1, (y1 - 15 if y1 - 15 > 15 else y1 + 15)
        cv2.putText(frame,
                    f"FID:{fid}",
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0), 2)

    # — 6) Write frame to output video —
    writer.write(frame)

    # — 7) Show combined result —
    cv2.imshow('Person & Face Tracking', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# — Clean up —
cap.release()
writer.release()
cv2.destroyAllWindows()

# — Write CSV of person dwell times —
with open(csv_path, 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['ID', 'DwellTime_s'])
    for tid, frames in dwell_frame_counts.items():
        deci = round(frames * 10 / fps)
        dwell_s = deci / 10.0
        csv_writer.writerow([tid, f"{dwell_s:.1f}"])

print(f"Saved combined output video to {out_video_path}")
print(f"Saved person dwell times CSV to {csv_path}")