import os
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# — Paths setup —
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, 'data', 'SampleVideo1.mp4')
print("Using absolute path:", video_path)

# — Open video & get FPS — 
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Error: Could not open video at {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS)

# — Load YOLO models (don’t chain .to()) —
device       = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount()>0 else 'cpu'
model_person = YOLO("yolov8n.pt")
model_person.to(device)
model_face   = YOLO("yolov8n-face.pt")
model_face.to(device)

# — DeepSort trackers —
person_tracker = DeepSort(max_age=50, n_init=5, embedder='mobilenet',
                          max_cosine_distance=0.3, nn_budget=200)
face_tracker   = DeepSort(max_age=50, n_init=5, embedder='mobilenet',
                          max_cosine_distance=0.3, nn_budget=200)

# — Detection thresholds —
PERSON_CLASS         = 0   # 'person'
PERSON_CONF_THRESHOLD = 0.6
FACE_CONF_THRESHOLD   = 0.5

# — State holders —
dwell_person       = defaultdict(int)    # tid → frames seen
dwell_face         = defaultdict(int)    # fid → frames seen
association_counts = defaultdict(int)    # (tid, fid) → frames overlapping

# — Utility: IoU for mapping detections to tracks —
def iou(boxA, boxB):
    x1,y1,x2,y2     = boxA
    x1p,y1p,x2p,y2p = boxB
    xx1, yy1 = max(x1,x1p), max(y1,y1p)
    xx2, yy2 = min(x2,x2p), min(y2,y2p)
    w, h     = max(0, xx2-xx1), max(0, yy2-yy1)
    inter    = w*h
    areaA    = (x2-x1)*(y2-y1)
    areaB    = (x2p-x1p)*(y2p-y1p)
    union    = areaA + areaB - inter
    return inter/union if union>0 else 0.0

# — Processing loop —
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Person detection + draw tight bbox + conf
    per_results      = model_person(frame, classes=[PERSON_CLASS], conf=PERSON_CONF_THRESHOLD)[0]
    person_detections = []
    for box in per_results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # draw green person box + confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame,
                    f"person {conf:.2f}",
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,0), 2)
        person_detections.append(((x1,y1,x2,y2), conf, 'person'))

    # 2) Face detection + draw tight bbox + conf
    face_results      = model_face(frame, conf=FACE_CONF_THRESHOLD)[0]
    face_detections = []
    for box in face_results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # draw green face box + confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame,
                    f"face {conf:.2f}",
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,0,255), 2)
        if conf > FACE_CONF_THRESHOLD:
            face_detections.append(((x1,y1,x2,y2), conf, 'face'))

    # 3) Update trackers
    p_tracks = person_tracker.update_tracks(person_detections, frame=frame)
    f_tracks = face_tracker.update_tracks(face_detections,   frame=frame)

    # 4) Update raw dwell counts
    for p in p_tracks:
        if not p.is_confirmed(): continue
        dwell_person[p.track_id] += 1
    for f in f_tracks:
        if not f.is_confirmed(): continue
        dwell_face[f.track_id] += 1

    # 5) Associate faces → persons by center‐in‐box
    for f in f_tracks:
        if not f.is_confirmed(): continue
        fid = f.track_id
        fx1, fy1, fx2, fy2 = map(int, f.to_ltrb())
        cx, cy = (fx1+fx2)//2, (fy1+fy2)//2
        # find the person whose bbox contains the face center
        for p in p_tracks:
            if not p.is_confirmed(): continue
            tid = p.track_id
            px1, py1, px2, py2 = map(int, p.to_ltrb())
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                association_counts[(tid, fid)] += 1
                break

    # 6) Map each detection bbox → its track ID (to get conf per track)
    person_conf = {}
    for p in p_tracks:
        if not p.is_confirmed(): continue
        best_iou, best_conf = 0, None
        tb = tuple(map(int, p.to_ltrb()))
        for (db, conf, _cls) in person_detections:
            i = iou(tb, db)
            if i > best_iou:
                best_iou, best_conf = i, conf
        if best_iou > 0.0:
            person_conf[p.track_id] = best_conf

    face_conf = {}
    for f in f_tracks:
        if not f.is_confirmed(): continue
        best_iou, best_conf = 0, None
        tb = tuple(map(int, f.to_ltrb()))
        for (db, conf, _cls) in face_detections:
            i = iou(tb, db)
            if i > best_iou:
                best_iou, best_conf = i, conf
        if best_iou > 0.0:
            face_conf[f.track_id] = best_conf

    # 7) Draw ID + conf + dwell (green) on person tracks
    for p in p_tracks:
        if not p.is_confirmed(): continue
        tid   = p.track_id
        conf  = person_conf.get(tid, 0.0)
        dwell = round(dwell_person[tid] * 10 / fps) / 10.0
        x1,y1,x2,y2 = map(int, p.to_ltrb())
        tx, ty = x1, (y1-15 if y1>15 else y1+15)
        cv2.putText(frame,
                    f"ID:{tid} {conf:.2f} {dwell:.1f}s",
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0), 2)

    # 8) Draw FID + conf + dwell (blue) on face tracks
    for f in f_tracks:
        if not f.is_confirmed(): continue
        fid   = f.track_id
        conf  = face_conf.get(fid, 0.0)
        dwell = round(dwell_face[fid] * 10 / fps) / 10.0
        x1,y1,x2,y2 = map(int, f.to_ltrb())
        tx, ty = x1, (y1-15 if y1>15 else y1+15)
        cv2.putText(frame,
                    f"FID:{fid} {conf:.2f} {dwell:.1f}s",
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,0,0), 2)

    # 9) Display
    cv2.imshow('Person & Face Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()