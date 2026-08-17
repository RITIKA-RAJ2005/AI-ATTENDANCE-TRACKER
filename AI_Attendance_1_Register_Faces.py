"""
STEP 1: FACE REGISTRATION
--------------------------
Captures a set of face images for a new person using the webcam and
stores them in dataset/<person_id>_<person_name>/ for later training.

Usage:
    python 1_register_faces.py
"""

import cv2
import os

DATASET_DIR = "dataset"
SAMPLES_TO_CAPTURE = 180          # number of face images to capture per person
FACE_DETECTOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")


def get_next_id():
    """Auto-increments numeric ID based on existing folders in dataset/."""
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    existing = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    ids = [int(d.split("_")[0]) for d in existing if d.split("_")[0].isdigit()]
    return max(ids, default=0) + 1


def register_face():
    name = input("Enter person's full name: ").strip().replace(" ", "-")
    if not name:
        print("Name cannot be empty.")
        return

    person_id = get_next_id()
    person_dir = os.path.join(DATASET_DIR, f"{person_id}_{name}")
    os.makedirs(person_dir, exist_ok=True)

    detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    if detector.empty():
        print(f"[ERROR] Could not load face detector from: {FACE_DETECTOR_PATH}")
        print("[ERROR] Make sure haarcascade_frontalface_default.xml is in the same folder as this script.")
        return
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not access the webcam.")
        return

    print(f"\n[INFO] Registering '{name}' with ID {person_id}.")
    print("[INFO] Look at the camera. Capturing images... Press 'q' to stop early.\n")

    count = 0
    while count < SAMPLES_TO_CAPTURE:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            file_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(file_path, face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {count}/{SAMPLES_TO_CAPTURE}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            break  # only take one face per frame

        cv2.imshow("Registering Face - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if count > 0:
        print(f"\n[SUCCESS] Captured {count} images for '{name}' (ID: {person_id}).")
        print("[INFO] Run 2_train_model.py next to train the recognizer.")
    else:
        print("[WARNING] No face captured. Try again with better lighting.")


if __name__ == "__main__":
    register_face()
