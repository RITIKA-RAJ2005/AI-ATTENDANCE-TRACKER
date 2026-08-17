"""
STEP 2: TRAIN RECOGNITION MODEL
---------------------------------
Reads all face images from dataset/ and trains an LBPH (Local Binary
Patterns Histograms) face recognizer. Saves the trained model and a
label map (id -> name) for use during attendance marking.

Usage:
    python 2_train_model.py
"""

import cv2
import os
import json
import numpy as np

DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"
MODEL_PATH = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_PATH = os.path.join(TRAINER_DIR, "labels.json")

FACE_DETECTOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")


def load_training_data():
    detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
    if detector.empty():
        raise RuntimeError(
            f"Could not load face detector from: {FACE_DETECTOR_PATH}\n"
            "Make sure haarcascade_frontalface_default.xml is in the same folder as this script."
        )
    face_samples = []
    ids = []
    id_to_name = {}

    person_folders = sorted(os.listdir(DATASET_DIR))
    if not person_folders:
        raise RuntimeError("No registered faces found. Run 1_register_faces.py first.")

    for folder in person_folders:
        folder_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        try:
            person_id_str, name = folder.split("_", 1)
            person_id = int(person_id_str)
        except ValueError:
            print(f"[WARNING] Skipping malformed folder name: {folder}")
            continue

        id_to_name[person_id] = name.replace("-", " ")

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                continue

            # Images were already cropped to faces during registration,
            # but we re-detect to keep training robust to stray files.
            faces = detector.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
            if len(faces) == 0:
                # assume the stored image is already a tight face crop
                face_samples.append(gray_img)
                ids.append(person_id)
            else:
                for (x, y, w, h) in faces:
                    face_samples.append(gray_img[y:y + h, x:x + w])
                    ids.append(person_id)

    return face_samples, ids, id_to_name


def train():
    os.makedirs(TRAINER_DIR, exist_ok=True)

    print("[INFO] Loading dataset...")
    face_samples, ids, id_to_name = load_training_data()
    print(f"[INFO] Loaded {len(face_samples)} face samples for {len(id_to_name)} people.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_samples, np.array(ids))
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "w") as f:
        json.dump(id_to_name, f, indent=2)

    print(f"[SUCCESS] Model trained and saved to '{MODEL_PATH}'.")
    print(f"[SUCCESS] Labels saved to '{LABELS_PATH}'.")
    print("[INFO] Run 3_mark_attendance.py next to start attendance monitoring.")


if __name__ == "__main__":
    train()
