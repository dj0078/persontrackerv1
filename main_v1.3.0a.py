import os
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# — Paths setup —
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, 'data', 'SampleVideo1.mp4')
print("Using absolute path:", video_path)

# — Open video & get FPS for dwell-time calculation —
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Error: Could not open video at {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS)
dwell_frame_counts = {}  # person_track_id → number of frames seen

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
PERSON_CLASS          = 0     # in COCO: 'person'
FACE_CLASS            = 0     # in face model: 'face'
PERSON_CONF_THRESHOLD = 0.6
FACE_CONF_THRESHOLD   = 0.5   # lowered from 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # — 1) Person detection (class 0 only) —
    per_results = model_person(frame, classes=[PERSON_CLASS], conf=PERSON_CONF_THRESHOLD)[0]
    person_detections = []
    for box in per_results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # draw green person box + confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,
                    f"person {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        # collect for tracking
        person_detections.append(((x1, y1, x2, y2), conf, 'person'))

    # — 2) Face detection (with new 0.5 confidence) —
    face_results = model_face(frame, conf=FACE_CONF_THRESHOLD)[0]
    face_detections = []
    for box in face_results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # draw green face box + confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,
                    f"face {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2)

        # only register if above FACE_CONF_THRESHOLD
        if conf > FACE_CONF_THRESHOLD:
            face_detections.append(((x1, y1, x2, y2), conf, 'face'))

    # — 3) Update both trackers —
    person_tracks = person_tracker.update_tracks(person_detections, frame=frame)
    face_tracks   = face_tracker.update_tracks(face_detections,   frame=frame)

    # — 4) Draw person IDs + dwell time (red) —
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

    # — 5) Draw face FIDs (blue) —
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

    # — 6) Show combined result —
    cv2.imshow('Person & Face Tracking', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
