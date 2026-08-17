"""
STEP 3: SMART ATTENDANCE MONITORING (GUI)
-------------------------------------------
A desktop window (Tkinter) showing the live webcam feed side-by-side
with a status table.

Attendance logic:
  - First sighting of the day -> Check-In time is recorded.
    Marked "Late" if check-in happens after LATE_AFTER_TIME, else "Present".
  - Check-Out only gets recorded once at least MIN_CHECKOUT_GAP_MINUTES
    have passed since Check-In (avoids a checkout equal to checkin from
    a brief walk-by). After that, it keeps updating on every later sighting.
  - Everyone in the trained roster starts the day as "Absent" and only
    flips to Present/Late once actually seen by the camera.

Auto idle/resume:
  - If no face has been seen for IDLE_TIMEOUT_SECONDS, the app switches
    to a low-power "paused" state: video stops updating and it only
    checks for a face every few seconds instead of continuously.
  - The moment a face reappears, it automatically resumes full live
    monitoring. (The camera hardware itself stays open the whole time,
    since it has to keep watching in order to notice someone return —
    but resource use drops a lot while idle.)

Each day's raw attendance is saved to its own CSV in attendance/
(e.g. attendance/attendance_2026-08-16.csv), same as before. On top of
that, every time you click Stop Camera (or close the window),
attendance_report.xlsx is automatically regenerated so the Excel
dashboard always reflects the latest session -- no need to run
AI_Attendance_4_Generate_Report.py separately unless you want to.

Requirements (in addition to opencv-contrib-python and numpy):
    pip install pillow openpyxl

Usage:
    python AI_Attendance_3_Mark_Attendance.py
"""

import cv2
import os
import json
import csv
import time
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox

try:
    from PIL import Image, ImageTk, ImageDraw
except ImportError:
    raise SystemExit(
        "Missing dependency 'pillow'. Install it with:\n"
        "    pip install pillow"
    )

try:
    from AI_Attendance_4_Generate_Report import generate_report
except ImportError:
    generate_report = None  # Excel report auto-generation just gets skipped if the file/openpyxl is missing

TRAINER_DIR = "trainer"
MODEL_PATH = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_PATH = os.path.join(TRAINER_DIR, "labels.json")
ATTENDANCE_DIR = "attendance"

FACE_DETECTOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")

# LBPH confidence: LOWER value = more confident match. Tune as needed.
CONFIDENCE_THRESHOLD = 65

# Check-ins recorded after this time (24hr HH:MM:SS, local clock) are marked "Late".
LATE_AFTER_TIME = "09:30:00"

# Minimum gap required between Check-In and Check-Out being recorded.
MIN_CHECKOUT_GAP_MINUTES = 60

# How often (seconds) to flush the CSV to disk / refresh the on-screen table while active.
CSV_WRITE_INTERVAL_SECONDS = 2

# Auto-pause after this many seconds with nobody in frame.
IDLE_TIMEOUT_SECONDS = 180

# Frame polling speed: fast while actively watching, slow while paused/idle.
ACTIVE_POLL_INTERVAL_MS = 15
IDLE_POLL_INTERVAL_MS = 2000


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Attendance System")

        self._load_model()          # may raise RuntimeError -> caller handles it
        self._init_session()
        self._build_ui()

        self.cam = None
        self.running = False
        self.last_write_time = 0
        self.last_face_seen = time.monotonic()
        self.idle = False
        self.frame_size = (640, 480)

    # ---------- model / data setup ----------

    def _load_model(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
            raise RuntimeError(
                "Trained model not found. Run AI_Attendance_1_Register_Faces.py "
                "then AI_Attendance_2_Train_Model.py first."
            )

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(MODEL_PATH)

        with open(LABELS_PATH, "r") as f:
            raw_labels = json.load(f)
        self.id_to_name = {int(k): v for k, v in raw_labels.items()}

        self.detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
        if self.detector.empty():
            raise RuntimeError(
                f"Could not load face detector from: {FACE_DETECTOR_PATH}\n"
                "Make sure haarcascade_frontalface_default.xml is in the same folder as this script."
            )

    def _init_session(self):
        os.makedirs(ATTENDANCE_DIR, exist_ok=True)
        self.today_str = datetime.now().strftime("%Y-%m-%d")
        self.csv_path = os.path.join(ATTENDANCE_DIR, f"attendance_{self.today_str}.csv")

        # Every known person starts Absent; flips to Present/Late once seen.
        # checkin_dt is kept internally (not written to CSV) to enforce the min gap.
        self.records = {
            name: {"status": "Absent", "checkin": "", "checkout": "", "checkin_dt": None}
            for name in self.id_to_name.values()
        }
        self._load_existing_csv()

    def _load_existing_csv(self):
        """Resume today's progress if the app was closed and reopened."""
        if not os.path.exists(self.csv_path):
            return
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Name")
                if name not in self.records:
                    continue
                checkin_str = row.get("Check-In", "")
                checkin_dt = None
                if checkin_str:
                    try:
                        checkin_dt = datetime.strptime(f"{self.today_str} {checkin_str}", "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        checkin_dt = None
                self.records[name] = {
                    "status": row.get("Status", "Absent"),
                    "checkin": checkin_str,
                    "checkout": row.get("Check-Out", ""),
                    "checkin_dt": checkin_dt,
                }

    # ---------- UI ----------

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(main)
        left.grid(row=0, column=0, padx=(0, 10), sticky="n")
        self.video_label = ttk.Label(left)
        self.video_label.grid(row=0, column=0)

        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=1, column=0, pady=8, sticky="ew")
        self.start_btn = ttk.Button(btn_frame, text="Start Camera", command=self.start)
        self.start_btn.grid(row=0, column=0, padx=4)
        self.stop_btn = ttk.Button(btn_frame, text="Stop Camera", command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=4)

        self.status_var = tk.StringVar(value=f"Date: {self.today_str}  |  Camera stopped")
        ttk.Label(left, textvariable=self.status_var).grid(row=2, column=0, pady=(4, 0), sticky="w")

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="n")
        ttk.Label(right, text="Live Attendance", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")

        columns = ("name", "status", "checkin", "checkout")
        self.tree = ttk.Treeview(right, columns=columns, show="headings", height=15)
        self.tree.heading("name", text="Name")
        self.tree.heading("status", text="Status")
        self.tree.heading("checkin", text="Check-In")
        self.tree.heading("checkout", text="Check-Out")
        self.tree.column("name", width=140)
        self.tree.column("status", width=80, anchor="center")
        self.tree.column("checkin", width=90, anchor="center")
        self.tree.column("checkout", width=90, anchor="center")
        self.tree.grid(row=1, column=0, pady=6)

        self.tree.tag_configure("Present", background="#d9f2d9")
        self.tree.tag_configure("Late", background="#fff3cd")
        self.tree.tag_configure("Absent", background="#f8d7da")

        self._refresh_table()

    # ---------- camera control ----------

    def start(self):
        if self.running:
            return
        self.cam = cv2.VideoCapture(0)
        if not self.cam.isOpened():
            messagebox.showerror("Camera Error", "Could not access the webcam.")
            self.cam = None
            return
        self.running = True
        self.idle = False
        self.last_face_seen = time.monotonic()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set(f"Date: {self.today_str}  |  Camera running")
        self._update_frame()

    def stop(self):
        self.running = False
        self.idle = False
        if self.cam is not None:
            self.cam.release()
            self.cam = None
        self._write_csv()
        self._refresh_table()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set(f"Date: {self.today_str}  |  Camera stopped")
        self._auto_generate_report()

    def _auto_generate_report(self):
        """Regenerate attendance_report.xlsx so it always reflects the latest session."""
        if generate_report is None:
            return
        try:
            generate_report(silent=True)
        except Exception as e:
            print(f"[WARNING] Could not auto-update attendance_report.xlsx: {e}")

    # ---------- attendance logic ----------

    def _mark_person(self, name):
        now = datetime.now()
        now_str = now.strftime("%H:%M:%S")
        rec = self.records[name]

        if rec["checkin_dt"] is None:
            late_cutoff = datetime.strptime(LATE_AFTER_TIME, "%H:%M:%S").time()
            rec["status"] = "Late" if now.time() > late_cutoff else "Present"
            rec["checkin"] = now_str
            rec["checkin_dt"] = now
            return  # checkout only starts counting from here onward

        # Only record/update Check-Out once the minimum gap has passed.
        if now - rec["checkin_dt"] >= timedelta(minutes=MIN_CHECKOUT_GAP_MINUTES):
            rec["checkout"] = now_str

    def _write_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Status", "Check-In", "Check-Out"])
            for name in sorted(self.records):
                rec = self.records[name]
                writer.writerow([name, rec["status"], rec["checkin"], rec["checkout"]])

    def _refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        for name in sorted(self.records):
            rec = self.records[name]
            self.tree.insert(
                "", "end",
                values=(name, rec["status"], rec["checkin"], rec["checkout"]),
                tags=(rec["status"],)
            )

    # ---------- idle placeholder ----------

    def _idle_placeholder_image(self):
        w, h = self.frame_size
        img = Image.new("RGB", (w, h), color=(40, 40, 40))
        draw = ImageDraw.Draw(img)
        text = "No one detected\nCamera paused - will resume automatically"
        draw.multiline_text((w // 2, h // 2), text, fill=(200, 200, 200), anchor="mm", align="center")
        return img

    # ---------- main video loop ----------

    def _update_frame(self):
        if not self.running or self.cam is None:
            return

        ret, frame = self.cam.read()
        if ret:
            h, w = frame.shape[:2]
            self.frame_size = (w, h)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

            if len(faces) > 0:
                self.last_face_seen = time.monotonic()
                if self.idle:
                    self.idle = False
                    self.status_var.set(f"Date: {self.today_str}  |  Camera running")
            else:
                idle_for = time.monotonic() - self.last_face_seen
                if not self.idle and idle_for > IDLE_TIMEOUT_SECONDS:
                    self.idle = True
                    self.status_var.set(
                        f"Date: {self.today_str}  |  Paused (no one detected for {IDLE_TIMEOUT_SECONDS // 60} min)"
                    )

            for (x, y, w_, h_) in faces:
                face_roi = gray[y:y + h_, x:x + w_]
                person_id, confidence = self.recognizer.predict(face_roi)

                if confidence < CONFIDENCE_THRESHOLD:
                    name = self.id_to_name.get(person_id, "Unknown")
                    color = (0, 255, 0)
                    label = f"{name} ({round(100 - confidence)}%)"
                    if name != "Unknown":
                        self._mark_person(name)
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    label = "Unknown"

                cv2.rectangle(frame, (x, y), (x + w_, y + h_), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            now_ts = datetime.now().timestamp()
            if now_ts - self.last_write_time > CSV_WRITE_INTERVAL_SECONDS:
                self._write_csv()
                self._refresh_table()
                self.last_write_time = now_ts

            if self.idle:
                img = self._idle_placeholder_image()
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk  # keep a reference so it isn't garbage-collected
            self.video_label.configure(image=imgtk)

        delay = IDLE_POLL_INTERVAL_MS if self.idle else ACTIVE_POLL_INTERVAL_MS
        self.root.after(delay, self._update_frame)

    def on_close(self):
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = AttendanceApp(root)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
    else:
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        root.mainloop()
