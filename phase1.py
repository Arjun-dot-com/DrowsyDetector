"""
Phase 1 - Pre-Ignition Face Verification
==========================================
Model  : MTCNN (face detection) + InceptionResnetV1 pretrained on VGGFace2
Library: facenet-pytorch  →  pip install facenet-pytorch

First run downloads ~90 MB of model weights to the torch hub cache
(%USERPROFILE%\.cache\torch\hub\checkpoints\ on Windows).
Every subsequent run is FULLY OFFLINE — no internet required.

Flow
----
  Authorised   → green box, "AUTHORISED [name]", proceed to Phase 2
  Unauthorised → red box, "UNAUTHORISED", owner notification spawns in
                 background thread; owner approves → face enrolled & allowed
                                                      owner denies  → blocked

Controls
--------
  e  - manually enrol a new authorised face
  q  - quit
"""

import cv2
import numpy as np
import torch
import pickle
import os
import threading
import queue
import time
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# ── Model setup ────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Phase 1] Using device: {DEVICE}")

# MTCNN detects the face and returns a 160×160 aligned crop ready for ResNet
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,        # only the largest / most confident face
    select_largest=True,
    post_process=True,     # applies fixed_image_standardisation before returning
    device=DEVICE
)

# InceptionResnetV1 pretrained on VGGFace2 — outputs a 512-d embedding vector
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# ── Database paths ─────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
DB_DIR  = os.path.join(_HERE, 'face_db')
DB_PATH = os.path.join(DB_DIR, 'embeddings.pkl')

# Cosine similarity must exceed this for a face to be considered a match.
# Range 0-1; raise toward 1 to be stricter, lower to be more lenient.
SIMILARITY_THRESHOLD = 0.75

# Number of sample frames captured during enrolment (more = more robust)
N_ENROL_SAMPLES = 10


# ── Database I/O ───────────────────────────────────────────────────────────────

def load_db() -> dict:
    """Load the face database from disk.  Returns {} if it does not exist yet."""
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'rb') as f:
            return pickle.load(f)
    return {}


def save_db(db: dict) -> None:
    os.makedirs(DB_DIR, exist_ok=True)
    with open(DB_PATH, 'wb') as f:
        pickle.dump(db, f)
    total = sum(len(v) for v in db.values())
    print(f"[DB] Saved — {len(db)} person(s), {total} embeddings total.")


# ── Face processing ────────────────────────────────────────────────────────────

def get_face_data(frame_bgr):
    """
    Run MTCNN + InceptionResnetV1 on one BGR frame.

    Returns
    -------
    embedding : np.ndarray shape (512,), or None if no face found
    box       : [x1, y1, x2, y2] ints, or None
    """
    img_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    # Detect returns all bounding boxes; mtcnn() returns the aligned face tensor
    boxes, _     = mtcnn.detect(img_rgb)
    face_tensor  = mtcnn(img_rgb)

    if face_tensor is None or boxes is None:
        return None, None

    with torch.no_grad():
        emb = resnet(face_tensor.unsqueeze(0).to(DEVICE))

    box = [int(v) for v in boxes[0]]   # boxes[0] = largest face
    return emb.cpu().numpy()[0], box


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-8))


def recognize(emb: np.ndarray, db: dict):
    """
    Compare embedding against every stored embedding in the database.

    Returns
    -------
    name  : str or None   — matched name, or None if below threshold
    score : float         — best cosine similarity found
    """
    best_name, best_score = None, -1.0
    for name, stored_embs in db.items():
        for stored in stored_embs:
            sc = cosine_sim(emb, stored)
            if sc > best_score:
                best_score, best_name = sc, name
    if best_score >= SIMILARITY_THRESHOLD:
        return best_name, best_score
    return None, best_score


# ── Enrolment ──────────────────────────────────────────────────────────────────

def enrol_face(cap, db: dict) -> dict:
    """
    Capture N_ENROL_SAMPLES frames from the camera, extract embeddings,
    and store them under the given name.  Multiple embeddings per person
    improve accuracy across lighting conditions and head angles.
    """
    name = input("\nEnter name for this person: ").strip()
    if not name:
        print("Enrolment cancelled.\n")
        return db

    print(f"Enrolling '{name}' — look at the camera.")
    print(f"Capturing {N_ENROL_SAMPLES} samples (hold reasonably still)...\n")

    collected = []
    while len(collected) < N_ENROL_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break

        emb, box = get_face_data(frame)

        if emb is not None:
            collected.append(emb)
            if box:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                              (0, 255, 0), 2)
            label      = f"Captured {len(collected)} / {N_ENROL_SAMPLES}"
            label_color = (0, 255, 0)
        else:
            label      = "No face detected — adjust position"
            label_color = (0, 0, 255)

        cv2.putText(frame, label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, label_color, 2)
        cv2.imshow("Enrolling", frame)
        cv2.waitKey(300)   # ~3 fps during enrolment for frame variety

    cv2.destroyWindow("Enrolling")

    if collected:
        # Merge with any existing samples (re-enrolment just adds more)
        db.setdefault(name, []).extend(collected)
        save_db(db)
        print(f"Enrolled '{name}' — {len(collected)} new sample(s) "
              f"({len(db[name])} total).\n")
    else:
        print("No face detected during enrolment — try again.\n")

    return db


# ── Owner notification (background thread) ────────────────────────────────────

def _owner_notification(approval_q: queue.Queue) -> None:
    """
    Simulates sending the owner a notification and waiting for their response.
    Runs in a daemon thread so the camera feed continues unblocked.
    """
    print("\n" + "=" * 55)
    print("  [OWNER NOTIFICATION]")
    print("  An unrecognised person is in the driver seat.")
    print("=" * 55)
    ans = input("  Grant access to this person? (y / n): ").strip().lower()
    approval_q.put(ans == 'y')


# ── Main Phase 1 loop ─────────────────────────────────────────────────────────

def phase1():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    db = load_db()
    print(f"\nPhase 1 — Face Verification  |  {len(db)} person(s) in database")
    print("Controls :  e = enrol new face   |   q = quit\n")
    if not db:
        print("  [!] Database is empty.")
        print("      Press 'e' to enrol authorised drivers before running.\n")

    approval_q          = queue.Queue()
    notification_active = False
    prev_status         = None
    last_print_t        = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        emb, box = get_face_data(frame)
        h, w     = frame.shape[:2]

        # ── Determine status ─────────────────────────────────────────────────
        if emb is None:
            status = "No Face Detected"
            color  = (128, 128, 128)
            score  = 0.0
            name   = ""

        elif not db:
            status = "No Database — Press 'e' to Enrol"
            color  = (0, 165, 255)
            score  = 0.0
            name   = ""

        else:
            name, score = recognize(emb, db)
            if name:
                status              = f"AUTHORISED  [ {name} ]"
                color               = (0, 255, 0)
                notification_active = False   # clear pending notification on authorised face
            else:
                status = "UNAUTHORISED"
                color  = (0, 0, 255)
                # Spawn the owner notification thread once per unauthorised session
                if not notification_active and approval_q.empty():
                    notification_active = True
                    threading.Thread(target=_owner_notification,
                                     args=(approval_q,), daemon=True).start()

        # ── Handle owner response ─────────────────────────────────────────────
        if not approval_q.empty():
            approved            = approval_q.get()
            notification_active = False
            if approved:
                print("\n[OWNER] Access granted. Starting enrolment for new person...")
                db = enrol_face(cap, db)
            else:
                print("[OWNER] Access denied — person remains blocked.\n")

        # ── Terminal output ───────────────────────────────────────────────────
        now = time.time()
        if status != prev_status:
            ts      = time.strftime("%H:%M:%S")
            sim_str = f"  (sim: {score:.3f})" if db and emb is not None else ""
            print(f"[{ts}]  *** {status} ***{sim_str}")
            prev_status  = status
            last_print_t = now
        elif now - last_print_t >= 1.0 and "No" not in status:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}]  {status:<36}  sim: {score:.3f}")
            last_print_t = now

        # ── On-screen overlay ─────────────────────────────────────────────────
        # Bounding box around detected face
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Status banner at top
        cv2.rectangle(frame, (0, 0), (w, 48), (30, 30, 30), -1)
        cv2.putText(frame, status, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

        # Similarity score at bottom
        if db and emb is not None:
            cv2.putText(frame,
                        f"Similarity: {score:.3f}  (threshold: {SIMILARITY_THRESHOLD})",
                        (10, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Owner notification indicator
        if notification_active:
            cv2.putText(frame, "Owner notified — awaiting response...",
                        (10, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        cv2.imshow("Phase 1 — Pre-Ignition Face Verification", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e') and not notification_active:
            db = enrol_face(cap, db)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    phase1()
