import cv2
import numpy as np
import mediapipe as mp
import time

from ear import ear_detect
from mar import mar_detect
from head_drop import head_drop_detect

# Separate face mesh instance used solely for drawing the full tessellation
# on the landmark window — kept independent of the three detector modules.
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh_mod  = mp.solutions.face_mesh

facemesh_viz = mp_face_mesh_mod.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def combine_status(hd_status, ear_status, mar_status):
    """
    Merge the three aspect statuses into one overall driver alert level.

    MAR yawn only escalates to a hard alert when EAR also shows eye closure,
    matching the EAR-precondition rule defined in the project spec.
    A standalone yawn (eyes fully open) is only a warning — it could be talking.
    """
    if hd_status == "HEAD DROP":
        return "ALERT: HEAD DROP",        (0, 0, 255)
    if ear_status == "Sleeping":
        return "ALERT: SLEEPING",         (0, 0, 180)
    if mar_status == "YAWNING" and ear_status in ("Drowsy", "Eyes Closing"):
        return "ALERT: YAWNING + DROWSY", (0, 0, 255)
    if ear_status == "Drowsy":
        return "ALERT: DROWSY",           (0, 0, 255)
    if mar_status == "YAWNING":
        return "WARNING: YAWNING",        (0, 100, 255)
    if ear_status == "Eyes Closing" or mar_status == "Mouth Wide Open":
        return "WARNING",                 (0, 165, 255)
    return "Awake",                       (0, 255, 0)


def draw_text_overlay(frame,
                      overall_status, overall_color,
                      hd_status,  hd_color,  pitch, yaw, roll,
                      ear_status, ear_color, avg_ear,
                      mar_status, mar_color, mar_val):
    """Add status text to a frame. Called for both the landmark and clean windows."""
    h, w = frame.shape[:2]

    # Dark banner at the top for the overall status
    cv2.rectangle(frame, (0, 0), (w, 44), (30, 30, 30), -1)
    cv2.putText(frame, f"Overall: {overall_status}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, overall_color, 2)

    # Per-aspect lines near the bottom
    y = h - 78
    cv2.putText(frame, f"EAR : {ear_status:<16}  ({avg_ear:.3f})",
                (10, y),      cv2.FONT_HERSHEY_SIMPLEX, 0.52, ear_color, 1)
    cv2.putText(frame, f"MAR : {mar_status:<16}  ({mar_val:.3f})",
                (10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, mar_color, 1)
    cv2.putText(frame, f"HEAD: {hd_status:<16}  P:{pitch:+.1f} Y:{yaw:+.1f} R:{roll:+.1f}",
                (10, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.52, hd_color, 1)


def phase2():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Phase 2 – Combined Detection running. Press 'q' to quit.\n")
    print("  Window 1 (left)  : with face mesh tessellation + detector landmarks")
    print("  Window 2 (right) : clean feed — status text only\n")

    drowsy_since    = None
    sleep_since     = None
    yawn_since      = None
    prev_overall    = None
    last_print_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # ── Window 2 (clean) ────────────────────────────────────────────────
        # Take a raw copy right away — nothing will be drawn on this except text.
        clean = frame.copy()

        # ── Run the three detectors ──────────────────────────────────────────
        # Each detector gets its own clean frame.copy() so its internal
        # facemesh always processes unmodified pixels — chaining them on the
        # same frame caused the axis arrows / eye contours drawn by the first
        # detector to corrupt landmark detection for the ones that follow.
        hd_frame  = frame.copy()
        ear_frame = frame.copy()
        mar_frame = frame.copy()

        hd_status,  pitch, yaw, roll, hd_color,  hd_frame  = head_drop_detect(hd_frame)
        ear_status, avg_ear, drowsy_since, sleep_since, ear_color, ear_frame = ear_detect(
            ear_frame, drowsy_since, sleep_since
        )
        mar_status, mar_val, yawn_since, mar_color, mar_frame = mar_detect(mar_frame, yawn_since)

        # Composite the three sets of drawings onto one frame.
        # For every pixel where a detector changed the colour versus the clean
        # original, copy that pixel into annotated — so all drawings coexist.
        annotated = frame.copy()
        for det_frame in (hd_frame, ear_frame, mar_frame):
            changed = np.any(det_frame != frame, axis=2)
            annotated[changed] = det_frame[changed]

        # ── Window 1 (landmarks): add full face mesh on top of detector drawings ──
        landmarks = annotated.copy()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # always process the clean original
        viz_results = facemesh_viz.process(img_rgb)
        if viz_results.multi_face_landmarks:
            for face_lms in viz_results.multi_face_landmarks:
                # Fine grey tessellation
                mp_drawing.draw_landmarks(
                    landmarks,
                    face_lms,
                    mp_face_mesh_mod.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # Stronger coloured contour lines
                mp_drawing.draw_landmarks(
                    landmarks,
                    face_lms,
                    mp_face_mesh_mod.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

        # ── Overall combined status ──────────────────────────────────────────
        overall_status, overall_color = combine_status(hd_status, ear_status, mar_status)

        # ── Terminal output ──────────────────────────────────────────────────
        now = time.time()
        if overall_status != prev_overall:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}]  *** {overall_status} ***")
            print(f"        EAR: {ear_status} ({avg_ear:.3f})  |  "
                  f"MAR: {mar_status} ({mar_val:.3f})  |  "
                  f"HEAD: {hd_status} (Pitch {pitch:+.1f}°)")
            prev_overall    = overall_status
            last_print_time = now
        elif now - last_print_time >= 1.0:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}]  {overall_status:<28}  "
                  f"EAR: {avg_ear:.3f}  MAR: {mar_val:.3f}  Pitch: {pitch:+.1f}°")
            last_print_time = now

        # ── Text overlay on both windows ─────────────────────────────────────
        draw_text_overlay(landmarks, overall_status, overall_color,
                          hd_status,  hd_color,  pitch, yaw, roll,
                          ear_status, ear_color, avg_ear,
                          mar_status, mar_color, mar_val)
        draw_text_overlay(clean, overall_status, overall_color,
                          hd_status,  hd_color,  pitch, yaw, roll,
                          ear_status, ear_color, avg_ear,
                          mar_status, mar_color, mar_val)

        cv2.imshow("Phase 2 - With Landmarks", landmarks)
        cv2.imshow("Phase 2 - Clean Feed",     clean)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    facemesh_viz.close()


if __name__ == "__main__":
    phase2()
