"""
frontend.py  -  Driver Monitor Dashboard
==========================================
Left  : plain, unmodified camera feed
Right : authorization status + Phase 2 metrics (visible only when authorised)
        OR enrollment controls (visible only when unauthorised)

Run:  python frontend.py
"""

import tkinter as tk
from tkinter import messagebox
import threading
import cv2
import time
from PIL import Image, ImageTk

from phase1 import (get_face_data, recognize, load_db, save_db,
                    SIMILARITY_THRESHOLD, N_ENROL_SAMPLES)
from ear       import ear_detect, AWAKE_THRESHOLD, SLEEP_THRESHOLD
from mar       import mar_detect, YAWN_THRESHOLD
from head_drop import head_drop_detect, PITCH_DROP_THRESHOLD

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = '#111111'
PANEL     = '#1a1a1a'
CARD      = '#222222'
SEP       = '#2c2c2c'
T_PRI     = '#f0f0f0'
T_SEC     = '#6e6e6e'
T_HEAD    = '#6faee8'
C_OK      = '#4caf78'
C_WARN    = '#d4943a'
C_ALERT   = '#c94040'
C_NEUT    = '#4e6470'
FONT      = 'Segoe UI'

CAM_W, CAM_H = 640, 480
RIGHT_W      = 390


# ── Status → colour ───────────────────────────────────────────────────────────
def _colour(status: str) -> str:
    s = status.lower()
    if any(w in s for w in ('authoris', 'awake', 'normal')):
        return C_OK
    if any(w in s for w in ('closing', 'wide', 'warning')):
        return C_WARN
    if any(w in s for w in ('unauthoris', 'drop', 'drowsy', 'sleeping', 'yawning')):
        return C_ALERT
    return C_NEUT


# ── Shared state between detection thread and UI thread ───────────────────────
def _make_state() -> dict:
    return {
        'running'       : True,
        'lock'          : threading.Lock(),
        'frame'         : None,          # latest raw camera frame (BGR)

        # Phase 1
        'auth'          : 'no_face',     # 'no_face' | 'no_db' | 'authorized' | 'unauthorized'
        'auth_name'     : '',
        'auth_score'    : 0.0,

        # Phase 2  (only meaningful when auth == 'authorized')
        'ear_val'       : 0.0,
        'ear_status'    : '-',
        'mar_val'       : 0.0,
        'mar_status'    : '-',
        'pitch'         : 0.0,
        'hd_status'     : '-',

        # Enrollment control  (written by UI, read by detection thread)
        'enrol_trigger' : False,
        'enrol_name'    : '',
        'enrol_progress': 0,             # 0 … N_ENROL_SAMPLES
        'enrol_done'    : False,
    }


# ── Detection thread ──────────────────────────────────────────────────────────
def _detection_loop(state: dict, db_ref: list, cap) -> None:
    """
    Runs all model inference in a background daemon thread.
    db_ref is a one-element list so the main thread can hot-swap the
    database after enrollment without touching thread internals.
    """
    drowsy_since = sleep_since = yawn_since = None

    while state['running']:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        with state['lock']:
            state['frame'] = frame.copy()

        # ── Enrollment mode ───────────────────────────────────────────────────
        with state['lock']:
            do_enrol   = state['enrol_trigger']
            enrol_name = state['enrol_name']

        if do_enrol and enrol_name.strip():
            with state['lock']:
                state['enrol_trigger']  = False
                state['enrol_progress'] = 0
                state['enrol_done']     = False

            collected = []
            while len(collected) < N_ENROL_SAMPLES and state['running']:
                ret2, f2 = cap.read()
                if not ret2:
                    continue
                with state['lock']:
                    state['frame'] = f2.copy()
                emb, _ = get_face_data(f2)
                if emb is not None:
                    collected.append(emb)
                    with state['lock']:
                        state['enrol_progress'] = len(collected)
                time.sleep(0.3)   # ~3 fps for sample variety

            if collected:
                db_ref[0].setdefault(enrol_name.strip(), []).extend(collected)
                save_db(db_ref[0])
            with state['lock']:
                state['enrol_done'] = True
            continue

        # ── Phase 1: face verification ────────────────────────────────────────
        emb, _ = get_face_data(frame)

        if emb is None:
            with state['lock']:
                state['auth'] = 'no_face'
            drowsy_since = sleep_since = yawn_since = None
            continue

        db = db_ref[0]
        if not db:
            with state['lock']:
                state['auth'] = 'no_db'
            continue

        name, score = recognize(emb, db)

        if name:
            # ── Phase 2: live metrics ─────────────────────────────────────────
            ear_st, ear_val, drowsy_since, sleep_since, _, _ = ear_detect(
                frame.copy(), drowsy_since, sleep_since)
            mar_st, mar_val, yawn_since, _, _ = mar_detect(
                frame.copy(), yawn_since)
            hd_st, pitch, _, _, _, _ = head_drop_detect(frame.copy())

            with state['lock']:
                state['auth']       = 'authorized'
                state['auth_name']  = name
                state['auth_score'] = round(score, 3)
                state['ear_val']    = round(ear_val, 3)
                state['ear_status'] = ear_st
                state['mar_val']    = round(mar_val, 3)
                state['mar_status'] = mar_st
                state['pitch']      = round(pitch, 1)
                state['hd_status']  = hd_st
        else:
            with state['lock']:
                state['auth']       = 'unauthorized'
                state['auth_name']  = ''
                state['auth_score'] = round(score, 3)
            drowsy_since = sleep_since = yawn_since = None


# ── Dashboard window ──────────────────────────────────────────────────────────
class Dashboard(tk.Tk):

    def __init__(self, state: dict, db_ref: list):
        super().__init__()
        self.state    = state
        self.db_ref   = db_ref
        self._approved = False       # owner approval flag for enrolment

        self.title('Driver Monitor System')
        self.configure(bg=BG)
        self.resizable(False, False)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

        self._build()
        self.after(50, self._tick)

    # ── Widget construction ───────────────────────────────────────────────────
    def _build(self):
        root_pad = tk.Frame(self, bg=BG, padx=14, pady=14)
        root_pad.pack(fill='both', expand=True)

        # Title
        tk.Label(root_pad, text='DRIVER MONITOR SYSTEM',
                 font=(FONT, 13, 'bold'), bg=BG, fg=T_HEAD
                 ).pack(anchor='w', pady=(0, 10))

        body = tk.Frame(root_pad, bg=BG)
        body.pack(fill='both', expand=True)

        # Left – camera
        cam_wrap = tk.Frame(body, bg=SEP, padx=1, pady=1)
        cam_wrap.pack(side='left', anchor='n')
        self.cam_lbl = tk.Label(cam_wrap, bg='black',
                                width=CAM_W, height=CAM_H)
        self.cam_lbl.pack()

        # Right – info
        right = tk.Frame(body, bg=PANEL, width=RIGHT_W)
        right.pack(side='left', fill='y', padx=(14, 0))
        right.pack_propagate(False)
        self._build_right(right)

    def _build_right(self, p):
        # ── Authorization ─────────────────────────────────────────────────────
        self._sec_hdr(p, 'AUTHORIZATION')

        self.lbl_auth = self._kv_row(p, 'Status', '-', T_PRI)[1]
        self.lbl_score = self._kv_row(p, 'Match Score', '-', T_SEC)[1]
        self._sep(p)

        # ── Metrics block (visible when authorised) ───────────────────────────
        self.f_metrics = tk.Frame(p, bg=PANEL)
        self._build_metrics(self.f_metrics)

        # ── Enrolment block (visible when unauthorised) ───────────────────────
        self.f_enrol = tk.Frame(p, bg=PANEL)
        self._build_enrol(self.f_enrol)

    def _build_metrics(self, p):
        self._sec_hdr(p, 'LIVE METRICS')

        # EAR
        self._metric_hdr(p, 'Eye Aspect Ratio  (EAR)')
        self.lbl_ear_val    = self._kv_row(p, 'Value',     '-', T_PRI)[1]
        self.lbl_ear_status = self._kv_row(p, 'Status',    '-', T_PRI)[1]
        self._kv_row(p, 'Awake when',  f'EAR > {AWAKE_THRESHOLD}', T_SEC)
        self._kv_row(p, 'Sleep when',  f'EAR < {SLEEP_THRESHOLD}', T_SEC)
        self._sep(p)

        # MAR
        self._metric_hdr(p, 'Mouth Aperture Ratio  (MAR)')
        self.lbl_mar_val    = self._kv_row(p, 'Value',     '-', T_PRI)[1]
        self.lbl_mar_status = self._kv_row(p, 'Status',    '-', T_PRI)[1]
        self._kv_row(p, 'Yawn when', f'MAR > {YAWN_THRESHOLD}', T_SEC)
        self._sep(p)

        # Head pose
        self._metric_hdr(p, 'Head Pose')
        self.lbl_pitch      = self._kv_row(p, 'Pitch',     '-', T_PRI)[1]
        self.lbl_hd_status  = self._kv_row(p, 'Status',    '-', T_PRI)[1]
        self._kv_row(p, 'Drop when', f'Pitch < {PITCH_DROP_THRESHOLD} deg', T_SEC)

    def _build_enrol(self, p):
        self._sec_hdr(p, 'ENROLMENT')

        tk.Label(p, text='Unrecognised driver detected.\nEnter a name and request owner approval\nto add this person to the database.',
                 font=(FONT, 9), bg=PANEL, fg=T_SEC,
                 justify='left'
                 ).pack(anchor='w', padx=18, pady=(2, 10))

        # Name field
        name_row = tk.Frame(p, bg=PANEL)
        name_row.pack(fill='x', padx=18, pady=(0, 8))
        tk.Label(name_row, text='Name', width=6, anchor='w',
                 font=(FONT, 9), bg=PANEL, fg=T_SEC).pack(side='left')
        self.enrol_entry = tk.Entry(
            name_row, font=(FONT, 10), bg=CARD, fg=T_PRI,
            insertbackground=T_PRI, relief='flat', bd=5)
        self.enrol_entry.pack(side='left', fill='x', expand=True, padx=(6, 0))

        # Buttons
        tk.Button(p, text='Request Owner Approval',
                  font=(FONT, 9), bg=CARD, fg=T_HEAD,
                  activebackground=SEP, relief='flat', pady=7,
                  cursor='hand2', bd=0,
                  command=self._request_approval
                  ).pack(fill='x', padx=18, pady=(0, 6))

        self.btn_enrol = tk.Button(
            p, text='Enrol',
            font=(FONT, 9, 'bold'), bg=C_NEUT, fg=T_PRI,
            activebackground=SEP, relief='flat', pady=7,
            cursor='hand2', bd=0, state='disabled',
            command=self._start_enrol)
        self.btn_enrol.pack(fill='x', padx=18, pady=(0, 8))

        self.lbl_enrol_msg = tk.Label(
            p, text='', font=(FONT, 9), bg=PANEL, fg=C_OK,
            wraplength=RIGHT_W - 36, justify='left')
        self.lbl_enrol_msg.pack(anchor='w', padx=18, pady=(0, 10))

    # ── Small layout helpers ──────────────────────────────────────────────────
    def _sec_hdr(self, parent, text):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill='x', padx=14, pady=(12, 4))
        tk.Label(f, text=text, font=(FONT, 9, 'bold'),
                 bg=PANEL, fg=T_HEAD).pack(anchor='w')
        tk.Frame(f, bg=SEP, height=1).pack(fill='x', pady=(3, 0))

    def _metric_hdr(self, parent, text):
        tk.Label(parent, text=text, font=(FONT, 9, 'bold'),
                 bg=PANEL, fg=T_PRI
                 ).pack(anchor='w', padx=18, pady=(6, 2))

    def _sep(self, parent):
        tk.Frame(parent, bg=SEP, height=1).pack(fill='x', padx=18, pady=8)

    def _kv_row(self, parent, key, value, val_fg):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill='x', padx=28, pady=2)
        tk.Label(f, text=key, width=12, anchor='w',
                 font=(FONT, 9), bg=PANEL, fg=T_SEC).pack(side='left')
        lbl = tk.Label(f, text=value, font=(FONT, 9, 'bold'),
                       bg=PANEL, fg=val_fg, anchor='w')
        lbl.pack(side='left', fill='x')
        return f, lbl

    # ── Button callbacks ──────────────────────────────────────────────────────
    def _request_approval(self):
        granted = messagebox.askyesno(
            'Owner Approval',
            'An unrecognised person is in the driver seat.\n\n'
            'Grant access and allow enrolment?'
        )
        if granted:
            self._approved = True
            self.btn_enrol.configure(state='normal', bg=C_OK, fg='black')
            self.lbl_enrol_msg.configure(
                text='Approval granted. Enter a name and click Enrol.',
                fg=C_OK)
        else:
            self._approved = False
            self.lbl_enrol_msg.configure(text='Owner denied access.', fg=C_ALERT)

    def _start_enrol(self):
        name = self.enrol_entry.get().strip()
        if not name:
            self.lbl_enrol_msg.configure(text='Enter a name first.', fg=C_WARN)
            return
        if not self._approved:
            self.lbl_enrol_msg.configure(
                text='Request owner approval first.', fg=C_WARN)
            return
        with self.state['lock']:
            self.state['enrol_name']    = name
            self.state['enrol_trigger'] = True
            self.state['enrol_progress'] = 0
            self.state['enrol_done']    = False
        self.btn_enrol.configure(state='disabled', bg=C_NEUT, fg=T_PRI)
        self._approved = False
        self.lbl_enrol_msg.configure(
            text=f'Capturing samples for "{name}"...', fg=C_WARN)

    def _on_close(self):
        self.state['running'] = False
        self.destroy()

    # ── Main update tick (runs in UI thread every 50 ms) ──────────────────────
    def _tick(self):
        with self.state['lock']:
            frame        = self.state['frame']
            auth         = self.state['auth']
            auth_name    = self.state['auth_name']
            auth_score   = self.state['auth_score']
            ear_val      = self.state['ear_val']
            ear_st       = self.state['ear_status']
            mar_val      = self.state['mar_val']
            mar_st       = self.state['mar_status']
            pitch        = self.state['pitch']
            hd_st        = self.state['hd_status']
            enrol_prog   = self.state['enrol_progress']
            enrol_done   = self.state['enrol_done']

        # Camera display – raw, no overlays
        if frame is not None:
            img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img   = img.resize((CAM_W, CAM_H), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.cam_lbl.configure(image=photo)
            self.cam_lbl.image = photo     # hold reference to prevent GC

        # Authorization labels
        if auth == 'no_face':
            self.lbl_auth.configure( text='No Face Detected',         fg=C_NEUT)
            self.lbl_score.configure(text='-')
        elif auth == 'no_db':
            self.lbl_auth.configure( text='Database Empty',           fg=C_WARN)
            self.lbl_score.configure(text='No drivers enrolled yet')
        elif auth == 'authorized':
            self.lbl_auth.configure( text=f'AUTHORISED  [ {auth_name} ]', fg=C_OK)
            self.lbl_score.configure(text=f'{auth_score:.3f}  (min {SIMILARITY_THRESHOLD})')
        elif auth == 'unauthorized':
            self.lbl_auth.configure( text='UNAUTHORISED',             fg=C_ALERT)
            self.lbl_score.configure(text=f'{auth_score:.3f}  (min {SIMILARITY_THRESHOLD})')

        # Show/hide right-panel sections
        if auth == 'authorized':
            self.f_enrol.pack_forget()
            self.f_metrics.pack(fill='x')
            self._refresh_metrics(ear_val, ear_st, mar_val, mar_st, pitch, hd_st)
        elif auth in ('unauthorized', 'no_db'):
            self.f_metrics.pack_forget()
            self.f_enrol.pack(fill='x')
            self._blank_metrics()
        else:
            self.f_metrics.pack_forget()
            self.f_enrol.pack_forget()
            self._blank_metrics()

        # Enrolment progress / completion
        if enrol_done:
            self.lbl_enrol_msg.configure(
                text='Enrolled successfully. Database updated.', fg=C_OK)
            with self.state['lock']:
                self.state['enrol_done'] = False
            self.enrol_entry.delete(0, 'end')
            self.btn_enrol.configure(state='disabled', bg=C_NEUT, fg=T_PRI)
            self.db_ref[0] = load_db()
        elif enrol_prog > 0:
            self.lbl_enrol_msg.configure(
                text=f'Capturing sample {enrol_prog} / {N_ENROL_SAMPLES}...',
                fg=C_WARN)

        self.after(50, self._tick)

    def _refresh_metrics(self, ear_val, ear_st, mar_val, mar_st, pitch, hd_st):
        self.lbl_ear_val.configure(   text=f'{ear_val:.3f}')
        self.lbl_ear_status.configure(text=ear_st,  fg=_colour(ear_st))
        self.lbl_mar_val.configure(   text=f'{mar_val:.3f}')
        self.lbl_mar_status.configure(text=mar_st,  fg=_colour(mar_st))
        self.lbl_pitch.configure(     text=f'{pitch:+.1f} deg')
        self.lbl_hd_status.configure( text=hd_st,   fg=_colour(hd_st))

    def _blank_metrics(self):
        for lbl in (self.lbl_ear_val, self.lbl_ear_status,
                    self.lbl_mar_val, self.lbl_mar_status,
                    self.lbl_pitch,   self.lbl_hd_status):
            lbl.configure(text='-', fg=T_PRI)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Could not open camera.')
        return

    db_ref = [load_db()]
    print(f'[Frontend] {len(db_ref[0])} person(s) in database.')

    state = _make_state()

    t = threading.Thread(target=_detection_loop,
                         args=(state, db_ref, cap),
                         daemon=True)
    t.start()

    Dashboard(state, db_ref).mainloop()

    state['running'] = False
    cap.release()


if __name__ == '__main__':
    main()
