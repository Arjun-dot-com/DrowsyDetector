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

# 6 landmark indices per eye following the EAR point ordering:
# p1 (outer corner), p2 (upper outer), p3 (upper inner),
# p4 (inner corner), p5 (lower inner), p6 (lower outer)
RIGHT_EYE_IDS = [33,  160, 158, 133, 153, 144]
LEFT_EYE_IDS  = [362, 385, 387, 263, 373, 380]

AWAKE_THRESHOLD = 0.25   # above this → Awake
SLEEP_THRESHOLD = 0.15   # below this → Sleeping zone

# Wall-clock durations — independent of camera / inference framerate
EYES_CLOSING_SECS = 0.67  # min continuous time below AWAKE_THRESHOLD before "Eyes Closing" shows;
                           # filters slow blinks which typically clear in < 0.5 s
DROWSY_SECS       = 2.0   # continuous time below AWAKE_THRESHOLD → Drowsy
SLEEP_SECS        = 1.0   # continuous time below SLEEP_THRESHOLD → Sleeping
                           # shorter because sub-0.15 closure is a stronger signal


def compute_ear(landmarks, eye_ids, w, h):
    """Compute EAR for one eye given its 6 landmark indices."""
    pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_ids],
        dtype=np.float64
    )
    p1, p2, p3, p4, p5, p6 = pts
    ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
    return ear


def ear_detect(frame, drowsy_since, sleep_since):
    """
    Analyse a single BGR frame for drowsiness / sleep via EAR.

    Parameters
    ----------
    frame        : BGR image from cv2
    drowsy_since : float timestamp (time.time()) when EAR first dropped below
                   AWAKE_THRESHOLD, or None if EAR is currently above it
    sleep_since  : float timestamp when EAR first dropped below SLEEP_THRESHOLD,
                   or None if EAR is not currently in the sleep zone

    Returns
    -------
    status       : str   - 'Awake', 'Eyes Closing', 'Drowsy', 'Sleeping', or 'No Face'
    avg_ear      : float - averaged EAR of both eyes (0.0 when no face)
    drowsy_since : float or None - updated timestamp
    sleep_since  : float or None - updated timestamp
    color        : tuple - BGR colour matching the status
    frame        : annotated frame
    """
    h, w = frame.shape[:2]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = facemesh.process(img_rgb)

    status  = "No Face"
    color   = (128, 128, 128)
    avg_ear = 0.0

    if not results.multi_face_landmarks:
        return status, avg_ear, None, None, color, frame

    lms = results.multi_face_landmarks[0].landmark

    left_ear  = compute_ear(lms, LEFT_EYE_IDS,  w, h)
    right_ear = compute_ear(lms, RIGHT_EYE_IDS, w, h)
    avg_ear   = (left_ear + right_ear) / 2.0
    now       = time.time()

    if avg_ear > AWAKE_THRESHOLD:
        drowsy_since = None
        sleep_since  = None
        status = "Awake"
        color  = (0, 255, 0)

    elif avg_ear < SLEEP_THRESHOLD:
        if drowsy_since is None:
            drowsy_since = now
        if sleep_since is None:
            sleep_since = now

        drowsy_elapsed = now - drowsy_since
        sleep_elapsed  = now - sleep_since

        if sleep_elapsed >= SLEEP_SECS:
            status = "Sleeping"
            color  = (0, 0, 180)
        elif drowsy_elapsed >= DROWSY_SECS:
            status = "Drowsy"
            color  = (0, 0, 255)
        elif drowsy_elapsed >= EYES_CLOSING_SECS:
            status = "Eyes Closing"
            color  = (0, 165, 255)
        else:
            status = "Awake"
            color  = (0, 255, 0)

    else:
        # Drowsy zone (0.15 – 0.25): drowsy timer runs, sleep timer resets
        if drowsy_since is None:
            drowsy_since = now
        sleep_since = None

        drowsy_elapsed = now - drowsy_since

        if drowsy_elapsed >= DROWSY_SECS:
            status = "Drowsy"
            color  = (0, 0, 255)
        elif drowsy_elapsed >= EYES_CLOSING_SECS:
            status = "Eyes Closing"
            color  = (0, 165, 255)
        else:
            status = "Awake"
            color  = (0, 255, 0)

    # Draw eye contours coloured by current status
    for eye_ids in (LEFT_EYE_IDS, RIGHT_EYE_IDS):
        pts = np.array(
            [(int(lms[i].x * w), int(lms[i].y * h)) for i in eye_ids],
            dtype=np.int32
        )
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)
        for pt in pts:
            cv2.circle(frame, tuple(pt), 2, (0, 255, 255), -1)

    return status, avg_ear, drowsy_since, sleep_since, color, frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("EAR Drowsiness Detection running. Press 'q' to quit.")
    print(f"  > {AWAKE_THRESHOLD}         → Awake")
    print(f"  {SLEEP_THRESHOLD} – {AWAKE_THRESHOLD} → Drowsy  (after {DROWSY_SECS}s continuous)")
    print(f"  < {SLEEP_THRESHOLD}         → Sleeping (after {SLEEP_SECS}s continuous)")
    print(f"  Eyes Closing shown after {EYES_CLOSING_SECS}s to suppress slow blinks\n")

    drowsy_since    = None
    sleep_since     = None
    prev_status     = None
    last_print_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        status, avg_ear, drowsy_since, sleep_since, color, annotated = ear_detect(
            frame, drowsy_since, sleep_since
        )

        now = time.time()
        drowsy_elapsed = round(now - drowsy_since, 1) if drowsy_since else 0.0
        sleep_elapsed  = round(now - sleep_since,  1) if sleep_since  else 0.0

        if status != prev_status:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}]  *** {status} ***  |  EAR: {avg_ear:.3f}  "
                  f"|  drowsy: {drowsy_elapsed}s  sleep: {sleep_elapsed}s")
            prev_status     = status
            last_print_time = now
        elif now - last_print_time >= 1.0 and status != "No Face":
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}]  {status:<14}  |  EAR: {avg_ear:.3f}  "
                  f"|  drowsy: {drowsy_elapsed}s  sleep: {sleep_elapsed}s")
            last_print_time = now

        cv2.putText(annotated, f"Status: {status}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(annotated, f"EAR: {avg_ear:.3f}   (awake>{AWAKE_THRESHOLD}  sleep<{SLEEP_THRESHOLD})",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(annotated, f"drowsy: {drowsy_elapsed}s/{DROWSY_SECS}s   "
                               f"sleep: {sleep_elapsed}s/{SLEEP_SECS}s",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("EAR Drowsiness Detection - Phase 2 Aspect 1", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    facemesh.close()


if __name__ == "__main__":
    main()
