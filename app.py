import cv2
import os
import math
import json
from collections import deque
from datetime import datetime, timezone
import torch
import time
import numpy as np
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db

VIDEO_PATH = "E:/system/google drive/shihabaaqil22242224@gmail.com/My Drive/PR (E )/source/study/asemester/FYP-19-22/Supun-Rashee E18 Project/Codes/Videos Src/videos/sdf.mp4"
CAMERA_ID = os.getenv("CAMERA_ID", "CAM01")
FIREBASE_CRED = "E:/system/google drive/shihabaaqil22242224@gmail.com/My Drive/PR (E )/source/study/asemester/FYP-19-22/Supun-Rashee E18 Project/Codes/Videos Src/html/cctv-monitor-f3e56-firebase-adminsdk-fbsvc-e6ae0d1f7e.json"
DATABASE_URL = "https://cctv-monitor-f3e56-default-rtdb.asia-southeast1.firebasedatabase.app"
TARGET_RESOLUTION = (360, 240)
POLYGON_SOURCE_RESOLUTION = (960, 540)
SKIP_FRAMES = 3
FRAME_SKIP_FIREBASE = 150
KNOWN_CAR_WIDTH_M = 1.83
MIN_BOX_WIDTH_PX = 5
BREAKDOWN_VELOCITY_THRESHOLD = 0.5
MAX_FRAMES_SINCE_SEEN = 5
TRAJ_LENGTH = 15
HEADING_CHANGE_DEG = 90
MIN_FRAMES_FOR_ANOMALY = 3
PRINT_EVERY = 20
JITTER_DIST_M = 0.05
EMA_ALPHA = 0.6
WINDOW_NAME = f"Camera {CAMERA_ID} Live Tracking"

'''x1 = [795,576,492,454,456,481,536,559,551,576,627,694,823,1108,1357,1576,1760,1209]
y1 = [1080,603,411,274,219,179,156,154,186,230,278,335,430,627,804,945,1080,1080]

x2 = [1920,1200,943,717,655,644,655,580,568,616,711,865,1094,1354,1627,1920,1920]
y2 = [676,400,303,212,178,158,133,153,181,229,302,398,527,680,848,1041,832]'''

#960,640
x1_base = [409,313,266,236,229,228,232,242,258,280,275,302,367,546,901,626]
y1_base = [540,351,248,167,138,118,101,90,82,74,92,124,178,299,540,540]
x2_base = [698,503,367,339,322,324,307,289,284,292,339,536,801,960,960]
y2_base = [236,163,109,97,86,68,70,79,89,99,140,263,417,516,340]

#x1_base = [4,391,475,539,581,595,614,692,705,707,699,690,671,645,600,295,0]
#y1_base = [409,263,227,196,169,157,139,159,185,220,266,307,360,426,540,540,540]
#x2_base = [0]
#y2_base = [0]

def scale_polygon_points(x_points, y_points, source_resolution, target_resolution):
    src_w, src_h = source_resolution
    dst_w, dst_h = target_resolution
    x_scale = dst_w / src_w
    y_scale = dst_h / src_h
    scaled_x = [round(x * x_scale) for x in x_points]
    scaled_y = [round(y * y_scale) for y in y_points]
    return scaled_x, scaled_y

x1, y1 = scale_polygon_points(
    x1_base,
    y1_base,
    POLYGON_SOURCE_RESOLUTION,
    TARGET_RESOLUTION,
)
x2, y2 = scale_polygon_points(
    x2_base,
    y2_base,
    POLYGON_SOURCE_RESOLUTION,
    TARGET_RESOLUTION,
)

LEFT_LANE_POLY = np.array([[a, b] for a, b in zip(x1, y1)], dtype=np.int32)
RIGHT_LANE_POLY = np.array([[c, d] for c, d in zip(x2, y2)], dtype=np.int32)

ALLOWED_POLYGONS = [LEFT_LANE_POLY, RIGHT_LANE_POLY]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE.upper()}")

cred = credentials.Certificate(FIREBASE_CRED)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})
    print("Firebase initialized.")
else:
    print("Firebase app already initialized. Reusing.")

cam_ref = db.reference(f"cameras/{CAMERA_ID}")
alarms_ref = db.reference("alarms")

def heading_diff(a, b):
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d

def draw_box_and_label(frame, box, label, is_anomaly):
    x1, y1, x2, y2 = box
    color = (0, 0, 255) if is_anomaly else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_y = y1 - 10 if y1 > 20 else y1 + 20
    cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

class TrackedObject:
    __slots__ = ("id", "initial_frame", "last_seen",
                 "travel_distance", "frame_count", "avg_velocity",
                 "centroids", "headings", "rotations",
                 "smooth_cx", "smooth_cy", "anomaly_reason")
    def __init__(self, tid: int):
        self.id = tid
        self.initial_frame = self.last_seen = 0
        self.travel_distance = self.frame_count = 0.0
        self.avg_velocity = 0.0
        self.centroids = deque(maxlen=TRAJ_LENGTH)
        self.headings = deque(maxlen=10)
        self.rotations = deque(maxlen=10)
        self.smooth_cx = self.smooth_cy = None
        self.anomaly_reason = ""

def run_live_tracking_firebase():
    model = YOLO('best.pt').to(DEVICE)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
        print("Warning: FPS not readable -> using 30 FPS")
    frame_cnt = 0
    start = time.time()
    tracked: dict[int, TrackedObject] = {}
    last_results = []
    anomaly_flag = False
    live_anomalies: dict[str, dict] = {}
    last_seen_cleanup: dict[int, int] = {}
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_cnt += 1
            frame = cv2.resize(frame, TARGET_RESOLUTION)
            for poly in ALLOWED_POLYGONS:
                cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            if frame_cnt % SKIP_FRAMES == 0:
                results = model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    conf=0.25,
                    verbose=False,
                    device=DEVICE
                )
                last_results = results
            else:
                results = last_results
            frame_anom = False
            should_push = (frame_cnt % FRAME_SKIP_FIREBASE == 0)
            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.int().cpu().numpy()
                cls = results[0].boxes.cls.int().cpu().numpy()
                for box, tid, cidx in zip(boxes, ids, cls):
                    x1, y1, x2, y2 = box
                    w_px = x2 - x1
                    if w_px < MIN_BOX_WIDTH_PX:
                        continue
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    meters_per_px = KNOWN_CAR_WIDTH_M / w_px
                    obj = tracked.get(tid)
                    if obj is None:
                        obj = TrackedObject(tid)
                        obj.initial_frame = frame_cnt
                        tracked[tid] = obj
                    obj.last_seen = frame_cnt
                    if obj.smooth_cx is not None:
                        cx = EMA_ALPHA * cx + (1 - EMA_ALPHA) * obj.smooth_cx
                        cy = EMA_ALPHA * cy + (1 - EMA_ALPHA) * obj.smooth_cy
                    obj.smooth_cx, obj.smooth_cy = cx, cy
                    obj.centroids.append((cx, cy, frame_cnt))
                    dx_m = dy_m = dist_m = 0.0
                    rotation = 0.0
                    heading_deg = 0.0
                    if len(obj.centroids) >= 2:
                        (px, py, _) = obj.centroids[-2]
                        dx_m = (cx - px) * meters_per_px
                        dy_m = (cy - py) * meters_per_px
                        dist_m = math.hypot(dx_m, dy_m)
                        obj.travel_distance += dist_m
                        obj.frame_count += 1
                        elapsed_sec = (frame_cnt - obj.initial_frame) / fps
                        obj.avg_velocity = obj.travel_distance / max(elapsed_sec, 1e-3)
                        if dist_m >= JITTER_DIST_M:
                            heading_deg = math.degrees(math.atan2(cy - py, cx - px))
                            if heading_deg < 0:
                                heading_deg += 360
                            obj.headings.append(heading_deg)
                            if len(obj.headings) >= 2:
                                rotation = heading_diff(obj.headings[-1], obj.headings[0])
                                obj.rotations.append(rotation)
                        else:
                            rotation = obj.rotations[-1] if obj.rotations else 0.0
                    centroid_pt = (float(cx), float(cy))
                    in_allowed_zone = False
                    for poly in ALLOWED_POLYGONS:
                        if cv2.pointPolygonTest(poly, centroid_pt, False) >= 0:
                            in_allowed_zone = True
                            break
                    reason = {}
                    if len(obj.centroids) >= MIN_FRAMES_FOR_ANOMALY:
                        if obj.avg_velocity < BREAKDOWN_VELOCITY_THRESHOLD:
                            reason["breakdown"] = True
                        if rotation >= HEADING_CHANGE_DEG:
                            reason["sharp_turn"] = True
                        if not in_allowed_zone:
                            reason["off_road"] = True
                    is_anom = bool(reason)
                    if is_anom:
                        frame_anom = True
                        anomaly_flag = True
                        obj.anomaly_reason = " ".join(reason.keys())
                        if should_push:
                            alarm_payload = {
                                "camera": CAMERA_ID,
                                "vehicle_id": str(tid),
                                "reason": obj.anomaly_reason,
                                "frame": frame_cnt,
                                "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
                            }
                            try:
                                alarms_ref.push(alarm_payload)
                            except Exception as e:
                                print(f"[ALARM ERROR] {e}")
                    else:
                        obj.anomaly_reason = ""
                    label = f"ID:{tid} {model.names[int(cidx)]} {obj.avg_velocity:.2f}m/s"
                    if is_anom:
                        label += f" [{obj.anomaly_reason.replace(' ', ',')}]"
                    draw_box_and_label(frame, (x1, y1, x2, y2), label, is_anom)
                    cv2.circle(frame, (int(cx), int(cy)), 4, (255, 255, 0), -1)
                    if len(obj.centroids) >= 2:
                        pts = np.array([(int(px), int(py)) for px, py, _ in obj.centroids], dtype=np.int32)
                        cv2.polylines(frame, [pts], False, (255, 255, 0), 2)
                    if is_anom:
                        live_anomalies[str(tid)] = {
                            "reason": obj.anomaly_reason,
                            "last_seen": frame_cnt
                        }
                        last_seen_cleanup[tid] = frame_cnt
                    else:
                        live_anomalies.pop(str(tid), None)
                lost = [t for t in tracked if frame_cnt - tracked[t].last_seen > MAX_FRAMES_SINCE_SEEN]
                for t in lost:
                    del tracked[t]
                    live_anomalies.pop(str(t), None)
                    last_seen_cleanup.pop(t, None)
            if should_push:
                updates = {
                    "status/anomaly": frame_anom,
                    "status/last_frame": frame_cnt,
                    "status/timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "status/total_vehicles": len(tracked),
                    "status/total_anomalies": len(live_anomalies)
                }
                for vid, data in live_anomalies.items():
                    updates[f"vehicles/{vid}"] = {
                        "reason": data["reason"],
                        "last_seen": data["last_seen"]
                    }
                expired = [int(vid) for vid, f in last_seen_cleanup.items() if frame_cnt - f > 15]
                for vid in expired:
                    vid_str = str(vid)
                    updates[f"vehicles/{vid_str}"] = None
                    live_anomalies.pop(vid_str, None)
                    last_seen_cleanup.pop(vid, None)
                try:
                    cam_ref.update(updates)
                except Exception as e:
                    print(f"[FIREBASE ERROR] {e}")
            status_text = f"{CAMERA_ID} | Frame:{frame_cnt} | Tracks:{len(tracked)} | Anomalies:{len(live_anomalies)}"
            status_color = (0, 0, 255) if frame_anom else (0, 255, 255)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break
            if frame_cnt % PRINT_EVERY == 0:
                print(f'fps: {frame_cnt/(time.time()-start)}')
    finally:
        cap.release()
        cv2.destroyAllWindows()
    total_time = time.time() - start
    print(f"\nDone in {total_time:.1f}s -> {frame_cnt} frames @ {frame_cnt/total_time:.1f} FPS")
    print("ANOMALIES DETECTED!" if anomaly_flag else "All clear")
if __name__ == "__main__":
    run_live_tracking_firebase()
