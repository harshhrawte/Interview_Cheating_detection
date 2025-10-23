# attention_monitor_strict.py
# pip install opencv-python mediapipe pandas numpy

import cv2
import mediapipe as mp
import time
import csv
from datetime import datetime
import numpy as np
import math

# ---------------------- CONFIG (tune these) ----------------------
LOOK_AWAY_SECONDS_THRESHOLD = 2.0    # seconds allowed to look away before alert
NO_FACE_SECONDS_THRESHOLD = 1.5     # seconds allowed with no face before alert
PARTIAL_FACE_SECONDS_THRESHOLD = 1.5
MIN_FACE_BOX_WIDTH = 0.22           # normalized width required for "full face visible"
EDGE_MARGIN = 0.03                  # normalized margin from edges to consider "partial/out"

GAZE_LEFT_THRESH = 0.35             # horizontal gaze thresholds (normalized within eye)
GAZE_RIGHT_THRESH = 0.65

EYE_DOWN_THRESH = 0.70              # vertical iris ratio > this => eyes looking down
EYE_UP_THRESH = 0.30                # vertical iris ratio < this => eyes looking up (optional)

HEAD_PITCH_DOWN_DEG = 15.0          # head pitch (in degrees) greater than this => looking down

LOG_FILENAME = "attention_log_strict.csv"
# -----------------------------------------------------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Landmark indices (Mediapipe FaceMesh)
LEFT_EYE_CORNERS = [33, 133]    # left eye outer/inner x-corners
RIGHT_EYE_CORNERS = [362, 263]  # right eye outer/inner x-corners
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

# For vertical eye position (eyelid top/bottom) — these are commonly used indices in FaceMesh
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# For head pose (solvePnP) use these landmarks:
# nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
HP_N = 1      # nose tip
HP_CHIN = 152
HP_LEFT_EYE = 33
HP_RIGHT_EYE = 263
HP_LEFT_MOUTH = 61
HP_RIGHT_MOUTH = 291

# Basic 3D model points of a generic face (millimetres) - common approximate values
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -63.6, -12.5),         # Chin
    (-43.3, 32.7, -26.0),        # Left eye left corner
    (43.3, 32.7, -26.0),         # Right eye right corner
    (-28.9, -28.9, -20.0),       # Left Mouth corner
    (28.9, -28.9, -20.0)         # Right mouth corner
], dtype=np.float64)

# helper: write header for log file
with open(LOG_FILENAME, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp_utc", "event", "detail"])

def log_event(event: str, detail: str = ""):
    ts = datetime.utcnow().isoformat()
    print(f"[{ts}] {event} - {detail}")
    with open(LOG_FILENAME, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, event, detail])

# timers and state
last_face_seen_time = time.time()
last_full_face_time = time.time()
last_looking_center_time = time.time()
current_alert = None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Run on a machine with a webcam.")

# main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    now = time.time()
    status_text = "OK"
    alert_text = ""
    alert_on = False

    if not results.multi_face_landmarks:
        # No face
        if now - last_face_seen_time > NO_FACE_SECONDS_THRESHOLD:
            alert_text = "No face detected!"
            alert_on = True
            if current_alert != "no_face":
                log_event("ALERT_NO_FACE", f"No face for {now - last_face_seen_time:.1f}s")
                current_alert = "no_face"
        else:
            status_text = "No face (waiting)"
    else:
        # face present
        last_face_seen_time = now
        current_alert = None

        lm = results.multi_face_landmarks[0].landmark

        # bounding box
        xs = np.array([l.x for l in lm])
        ys = np.array([l.y for l in lm])
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        box_w = max_x - min_x
        box_h = max_y - min_y
        touching_edge = (min_x < EDGE_MARGIN) or (max_x > 1.0 - EDGE_MARGIN) or (min_y < EDGE_MARGIN) or (max_y > 1.0 - EDGE_MARGIN)

        if box_w < MIN_FACE_BOX_WIDTH or touching_edge:
            if now - last_full_face_time > PARTIAL_FACE_SECONDS_THRESHOLD:
                alert_text = "Partial / face not fully visible!"
                alert_on = True
                if current_alert != "partial_face":
                    log_event("ALERT_PARTIAL_FACE", f"bbox_w={box_w:.3f}, touching_edge={touching_edge}")
                    current_alert = "partial_face"
            else:
                status_text = "Face partial (waiting)"
        else:
            last_full_face_time = now
            current_alert = None

            # --------- horizontal gaze (as before) ----------
            left_eye_x = (lm[LEFT_EYE_CORNERS[0]].x + lm[LEFT_EYE_CORNERS[1]].x) / 2.0
            right_eye_x = (lm[RIGHT_EYE_CORNERS[0]].x + lm[RIGHT_EYE_CORNERS[1]].x) / 2.0
            center_x = (left_eye_x + right_eye_x) / 2.0

            # iris centers
            left_iris_x = np.mean([lm[i].x for i in LEFT_IRIS])
            left_iris_y = np.mean([lm[i].y for i in LEFT_IRIS])
            right_iris_x = np.mean([lm[i].x for i in RIGHT_IRIS])
            right_iris_y = np.mean([lm[i].y for i in RIGHT_IRIS])

            # horizontal gaze ratio inside each eye
            left_eye_left_x = lm[LEFT_EYE_CORNERS[0]].x
            left_eye_right_x = lm[LEFT_EYE_CORNERS[1]].x
            right_eye_left_x = lm[RIGHT_EYE_CORNERS[0]].x
            right_eye_right_x = lm[RIGHT_EYE_CORNERS[1]].x

            left_eye_width = max(left_eye_right_x - left_eye_left_x, 1e-6)
            right_eye_width = max(right_eye_right_x - right_eye_left_x, 1e-6)

            left_h_ratio = (left_iris_x - left_eye_left_x) / left_eye_width
            right_h_ratio = (right_iris_x - right_eye_left_x) / right_eye_width
            gaze_h_ratio = (left_h_ratio + right_h_ratio) / 2.0

            # --------- vertical iris ratio inside each eye ----------
            # Using eyelid top/bottom landmarks to compute vertical position (normalized 0..1)
            left_top_y = lm[LEFT_EYE_TOP].y
            left_bottom_y = lm[LEFT_EYE_BOTTOM].y
            right_top_y = lm[RIGHT_EYE_TOP].y
            right_bottom_y = lm[RIGHT_EYE_BOTTOM].y

            left_eye_height = max(left_bottom_y - left_top_y, 1e-6)
            right_eye_height = max(right_bottom_y - right_top_y, 1e-6)

            left_v_ratio = (left_iris_y - left_top_y) / left_eye_height   # 0 = top, 1 = bottom
            right_v_ratio = (right_iris_y - right_top_y) / right_eye_height
            gaze_v_ratio = (left_v_ratio + right_v_ratio) / 2.0

            # Draw debug points
            cv2.circle(frame, (int(left_iris_x*w), int(left_iris_y*h)), 3, (0,255,255), -1)
            cv2.circle(frame, (int(right_iris_x*w), int(right_iris_y*h)), 3, (0,255,255), -1)
            cv2.rectangle(frame, (int(min_x*w), int(min_y*h)), (int(max_x*w), int(max_y*h)), (200,200,200), 1)

            # Horizontal gaze state
            if gaze_h_ratio < GAZE_LEFT_THRESH:
                gaze_state_h = "right"   # subject looking right
                gaze_text_h = "Looking Right"
            elif gaze_h_ratio > GAZE_RIGHT_THRESH:
                gaze_state_h = "left"
                gaze_text_h = "Looking Left"
            else:
                gaze_state_h = "center"
                gaze_text_h = "Looking Center"

            # Vertical gaze state (0=top, 1=bottom)
            if gaze_v_ratio > EYE_DOWN_THRESH:
                gaze_state_v = "down"
                gaze_text_v = "Eyes Down"
            elif gaze_v_ratio < EYE_UP_THRESH:
                gaze_state_v = "up"
                gaze_text_v = "Eyes Up"
            else:
                gaze_state_v = "center_v"
                gaze_text_v = "Eyes Center"

            # --------- Head pose estimation (solvePnP) ----------
            # image points
            image_points = np.array([
                (lm[HP_N].x * w, lm[HP_N].y * h),         # nose tip
                (lm[HP_CHIN].x * w, lm[HP_CHIN].y * h),   # chin
                (lm[HP_LEFT_EYE].x * w, lm[HP_LEFT_EYE].y * h), # left eye corner
                (lm[HP_RIGHT_EYE].x * w, lm[HP_RIGHT_EYE].y * h),# right eye corner
                (lm[HP_LEFT_MOUTH].x * w, lm[HP_LEFT_MOUTH].y * h),# left mouth
                (lm[HP_RIGHT_MOUTH].x * w, lm[HP_RIGHT_MOUTH].y * h) # right mouth
            ], dtype=np.float64)

            # camera internals
            focal_length = w
            center = (w/2, h/2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4,1))  # assume no lens distortion

            # solvePnP
            success_pnp, rotation_vector, translation_vector = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            pitch = 0.0
            if success_pnp:
                # Convert rotation vector to rotation matrix
                rmat, _ = cv2.Rodrigues(rotation_vector)
                proj_matrix = np.hstack((rmat, translation_vector))
                eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]  # returns (pitch, yaw, roll)
                pitch = float(eulerAngles[0])  # degrees
                yaw = float(eulerAngles[1])
                roll = float(eulerAngles[2])
            else:
                pitch = 0.0
                yaw = 0.0
                roll = 0.0

            # Determine overall attention using combined checks
            attention_ok = True
            reason = []

            # Horizontal away check (same as before)
            if gaze_state_h != "center":
                away_duration = now - last_looking_center_time
                # only update timer if truly not center horizontally
                if gaze_state_h == "center":
                    last_looking_center_time = now
                else:
                    # if previously center, this is new away start
                    if now - last_looking_center_time > LOOK_AWAY_SECONDS_THRESHOLD:
                        attention_ok = False
                        reason.append(f"Horizontal gaze away ({gaze_state_h}) {now - last_looking_center_time:.1f}s")
                        if current_alert != "look_away":
                            log_event("ALERT_LOOK_AWAY", f"{gaze_state_h} for {now - last_looking_center_time:.1f}s, gaze_h_ratio={gaze_h_ratio:.2f}")
                            current_alert = "look_away"
            else:
                last_looking_center_time = now

            # Vertical eye down check (even if horizontal center)
            if gaze_state_v == "down" or pitch > HEAD_PITCH_DOWN_DEG:
                attention_ok = False
                if gaze_state_v == "down":
                    reason.append(f"Eyes looking down (v_ratio={gaze_v_ratio:.2f})")
                    if current_alert != "eyes_down":
                        log_event("ALERT_EYES_DOWN", f"v_ratio={gaze_v_ratio:.2f}")
                        current_alert = "eyes_down"
                if pitch > HEAD_PITCH_DOWN_DEG:
                    reason.append(f"Head pitch down ({pitch:.1f} deg)")
                    if current_alert != "head_down":
                        log_event("ALERT_HEAD_DOWN", f"pitch={pitch:.1f}")
                        current_alert = "head_down"

            # Optionally: embed object detection here to detect phone/other objects in frame.
            # If you want phone detection: use a small object detection model (SSD/MobileNet or YOLO)
            # and run it on 'frame'. If it detects class 'cell phone' or 'mobile phone' -> alert and log.
            # Placeholder:
            # detected_phone = False
            # if detected_phone:
            #     attention_ok = False
            #     reason.append("Phone detected in frame")
            #     log_event("ALERT_PHONE_DETECTED", "phone bounding box ...")

            # Finalize status/alerts
            if not attention_ok:
                alert_on = True
                alert_text = "; ".join(reason)
            else:
                status_text = "Attention: OK"

            # draw debug info
            cv2.putText(frame, f"Hg:{gaze_h_ratio:.2f} Vg:{gaze_v_ratio:.2f}", (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Pitch:{pitch:.1f}", (20, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, gaze_text_h + " | " + gaze_text_v, (20, h-100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Render overlays
    if alert_on:
        overlay = frame.copy()
        alpha = 0.35
        cv2.rectangle(overlay, (0,0), (w,h), (0,0,255), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, "ALERT", (w//2 - 70, 70), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255,255,255), 3)
        cv2.putText(frame, alert_text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    else:
        cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    cv2.imshow("Strict Attention Monitor", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
