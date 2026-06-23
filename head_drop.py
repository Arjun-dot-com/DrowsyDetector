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

# MediaPipe landmark indices for the 6 key points used in solvePnP
# Nose tip, Chin, Left eye outer corner, Right eye outer corner, Left mouth corner, Right mouth corner
POSE_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# Corresponding 3D coordinates from a normalized rigid human face model (in mm)
MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),       # Nose tip
    (0.0,  -330.0,  -65.0),      # Chin
    (-225.0, 170.0, -135.0),     # Left eye outer corner
    (225.0,  170.0, -135.0),     # Right eye outer corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0,  -150.0, -125.0),    # Right mouth corner
], dtype=np.float64)

# A head drop is confirmed when pitch dips below this value (degrees)
PITCH_DROP_THRESHOLD = -12.0


def get_head_angles(rvec):
    # Return (pitch, yaw, roll) in degrees from a solvePnP rotation vector.
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles
    return pitch, yaw, roll


def head_drop_detect(frame):
    """
    Analyse a single BGR frame for head-drop events.

    Returns
    -------
    status : str   - 'Awake', 'HEAD DROP', or 'No Face'
    pitch  : float - pitch angle in degrees (0 when no face)
    yaw    : float
    roll   : float
    color  : tuple - BGR colour matching the status
    frame  : the annotated frame
    """
    h, w = frame.shape[:2]
    # OpenCV takes the colors in BGR but mediapipe takes the colors in RGB, so we convert the color
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = facemesh.process(img_rgb)

    status = "No Face"
    color  = (128, 128, 128)
    pitch = yaw = roll = 0.0

    if not results.multi_face_landmarks:
        return status, pitch, yaw, roll, color, frame

    lms = results.multi_face_landmarks[0].landmark

    # 2D image coordinates of the six key landmarks
    image_points_2d = np.array(
        [(lms[i].x * w, lms[i].y * h) for i in POSE_LANDMARK_IDS],
        dtype=np.float64
    )

    # Approximate pinhole camera matrix (focal ≈ frame width)
    focal = float(w)
    camera_matrix = np.array([
        [focal, 0,     w / 2.0],
        [0,     focal, h / 2.0],
        [0,     0,     1.0    ]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok:
        return status, pitch, yaw, roll, color, frame

    pitch, yaw, roll = get_head_angles(rvec)

    if pitch < PITCH_DROP_THRESHOLD or pitch > ((-1 * PITCH_DROP_THRESHOLD) + 5):
        status = "HEAD DROP"
        color  = (0, 0, 255)
    else:
        status = "Awake"
        color  = (0, 255, 0)

    # --- Visual overlays ---
    # Draw the six key landmark dots
    for i in POSE_LANDMARK_IDS:
        pt = (int(lms[i].x * w), int(lms[i].y * h))
        cv2.circle(frame, pt, 3, (0, 255, 255), -1)

    # Project 3-axis pose arrows from the nose tip
    axis_len = 80.0
    axis_3d = np.float64([
        [0, 0, 0],
        [axis_len, 0, 0],   # X - Red
        [0, -axis_len, 0],  # Y - Green  (negated so arrow points up on screen)
        [0, 0, -axis_len],  # Z - Blue
    ])
    projected, _ = cv2.projectPoints(
        axis_3d.reshape(-1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    origin = tuple(projected[0].ravel().astype(int))
    cv2.line(frame, origin, tuple(projected[1].ravel().astype(int)), (0,   0,   255), 2)
    cv2.line(frame, origin, tuple(projected[2].ravel().astype(int)), (0,   255, 0  ), 2)
    cv2.line(frame, origin, tuple(projected[3].ravel().astype(int)), (255, 0,   0  ), 2)

    return status, pitch, yaw, roll, color, frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Head Drop Detection running. Press 'q' to quit.")
    print(f"Alert threshold: Pitch < {PITCH_DROP_THRESHOLD}°\n")

    prev_status = None
    last_print_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        status, pitch, yaw, roll, color, annotated = head_drop_detect(frame)

        now = time.time()

        # Print to terminal on status change OR every second (for angle monitoring)
        if status != prev_status:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}]  *** {status} ***  |  Pitch: {pitch:+.1f}°  Yaw: {yaw:+.1f}°  Roll: {roll:+.1f}°")
            prev_status = status
            last_print_time = now
        elif now - last_print_time >= 1.0 and status != "No Face":
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}]  {status:<12}  |  Pitch: {pitch:+.1f}°  Yaw: {yaw:+.1f}°  Roll: {roll:+.1f}°")
            last_print_time = now

        # On-screen overlays
        cv2.putText(annotated, f"Status: {status}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(annotated, f"Pitch {pitch:+.1f}  Yaw {yaw:+.1f}  Roll {roll:+.1f}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f"Drop threshold: Pitch < {PITCH_DROP_THRESHOLD}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Head Drop Detection - Phase 2 Aspect 3", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    facemesh.close()


if __name__ == "__main__":
    main()
