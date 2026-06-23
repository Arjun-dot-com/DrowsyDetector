import cv2
import numpy as np
import mediapipe as mp
import time

mp_face_mesh_module = mp.solutions.face_mesh
facemesh = mp_face_mesh_module.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 6 inner-lip landmark indices following the MAR point ordering:
# m1 (left corner), m2 (upper left), m3 (upper right), m4 (right corner), m5 (lower right), m6 (lower left)
# Inner lip is used (not outer) because it captures the actual air-gap between the lips accurately, independent of lip thickness.
MOUTH_IDS = [78, 82, 312, 308, 317, 87]

# Additional outer-lip indices used only for drawing the full mouth contour
OUTER_LIP_IDS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                 375, 321, 405, 314, 17, 84, 181, 91, 146]

# MAR above this is a potential yawn.
# Normal speech typically stays below 0.5 and fluctuates rapidly.
# A genuine yawn exceeds 0.6 and holds steady for several seconds.
YAWN_THRESHOLD = 0.60

# Wall-clock durations — independent of camera / inference framerate
MOUTH_OPEN_SECS = 0.5   # min continuous time above YAWN_THRESHOLD before "Mouth Wide Open" shows;
                         # filters brief wide openings during speech or laughter
YAWN_SECS       = 1.0   # continuous time above YAWN_THRESHOLD → confirmed yawn;
                         # talking never sustains a wide opening this long


def compute_mar(landmarks, mouth_ids, w, h):
    """Compute MAR for the mouth given the 6 inner-lip landmark indices."""
    pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_ids],
        dtype=np.float64
    )
    m1, m2, m3, m4, m5, m6 = pts
    mar = (np.linalg.norm(m2 - m6) + np.linalg.norm(m3 - m5)) / (2.0 * np.linalg.norm(m1 - m4))
    return mar


def mar_detect(frame, yawn_since):
    """
    Analyse a single BGR frame for yawning via MAR.

    Parameters
    ----------
    frame      : BGR image from cv2
    yawn_since : float timestamp (time.time()) when MAR first exceeded YAWN_THRESHOLD,
                 or None if MAR is currently below it

    Returns
    -------
    status     : str         - 'Normal', 'Mouth Wide Open', 'YAWNING', or 'No Face'
    mar        : float       - current MAR value (0.0 when no face)
    yawn_since : float|None  - updated timestamp
    color      : tuple       - BGR colour matching the status
    frame      : annotated frame
    """
    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = facemesh.process(img_rgb)

    status = "No Face"
    color  = (128, 128, 128)
    mar    = 0.0

    if not results.multi_face_landmarks:
        return status, mar, None, color, frame

    lms = results.multi_face_landmarks[0].landmark
    mar = compute_mar(lms, MOUTH_IDS, w, h)
    now = time.time()

    if mar > YAWN_THRESHOLD:
        if yawn_since is None:
            yawn_since = now
        elapsed = now - yawn_since

        if elapsed >= YAWN_SECS:
            status = "YAWNING"
            color  = (0, 0, 255)
        elif elapsed >= MOUTH_OPEN_SECS:
            status = "Mouth Wide Open"
            color  = (0, 165, 255)
        else:
            status = "Normal"
            color  = (0, 255, 0)
    else:
        yawn_since = None
        status = "Normal"
        color  = (0, 255, 0)

    # Outer lip contour
    outer_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h)) for i in OUTER_LIP_IDS],
        dtype=np.int32
    )
    cv2.polylines(frame, [outer_pts], isClosed=True, color=(200, 200, 200), thickness=1)

    # 6 inner MAR landmark dots + connecting polygon
    inner_pts = np.array(
        [(int(lms[i].x * w), int(lms[i].y * h)) for i in MOUTH_IDS],
        dtype=np.int32
    )
    cv2.polylines(frame, [inner_pts], isClosed=True, color=color, thickness=1)
    for pt in inner_pts:
        cv2.circle(frame, tuple(pt), 3, (0, 255, 255), -1)

    return status, mar, yawn_since, color, frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("MAR Yawn Detection running. Press 'q' to quit.")
    print(f"  MAR > {YAWN_THRESHOLD}  → potential yawn")
    print(f"  Mouth Wide Open after {MOUTH_OPEN_SECS}s continuous — suppresses brief speech openings")
    print(f"  YAWNING confirmed after {YAWN_SECS}s continuous\n")

    yawn_since      = None
    prev_status     = None
    last_print_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        status, mar, yawn_since, color, annotated = mar_detect(frame, yawn_since)

        now          = time.time()
        yawn_elapsed = round(now - yawn_since, 1) if yawn_since else 0.0

        if status != prev_status:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}]  *** {status} ***  |  MAR: {mar:.3f}  |  elapsed: {yawn_elapsed}s")
            prev_status     = status
            last_print_time = now
        elif now - last_print_time >= 1.0 and status != "No Face":
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}]  {status:<16}  |  MAR: {mar:.3f}  |  elapsed: {yawn_elapsed}s")
            last_print_time = now

        cv2.putText(annotated, f"Status: {status}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(annotated, f"MAR: {mar:.3f}   (yawn threshold: > {YAWN_THRESHOLD})",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f"elapsed: {yawn_elapsed}s / {YAWN_SECS}s",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("MAR Yawn Detection - Phase 2 Aspect 2", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    facemesh.close()


if __name__ == "__main__":
    main()
