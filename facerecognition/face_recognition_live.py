import face_recognition
import cv2
import numpy as np
import os
import re
import sys
import time
from collections import defaultdict

# ==============================
# CONFIG
# ==============================
USE_CNN = False            # True = 'cnn' detector (more accurate on profiles, slower). False = 'hog' (faster).
FRAME_RESIZE = 0.25        # Downscale factor for processing (e.g., 0.25 = 1/4 size). Increase for speed, decrease for accuracy.
PROCESS_EVERY_N_FRAMES = 2 # Process every Nth frame (1 = process all frames).
TOLERANCE = 0.55           # Matching tolerance (lower = stricter).
NUM_JITTERS = 0            # 0 or 1 is typical. >0 = more robust but slower.
CAMERA_INDEX = 0           # Change if you have multiple cameras.
TRAINING_DIR = "."         # Directory to scan for .jpg training images.

DETECTION_MODEL = "cnn" if USE_CNN else "hog"
SCALE_BACK = 1.0 / FRAME_RESIZE

# ==============================
# Helpers
# ==============================
def extract_person_name(filename: str) -> str:
    """
    Map 'otto.jpg', 'otto1.jpg', 'otto2.jpg' -> 'otto'
    'sandra3.jpg' -> 'sandra'
    Only strips trailing digits; keeps digits in the middle of names intact.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r'\d+$', '', base)

# ==============================
# Load training images
# ==============================
def load_known_faces(directory=TRAINING_DIR, detector_model=DETECTION_MODEL):
    encodings_by_person = defaultdict(list)
    files = [f for f in os.listdir(directory) if f.lower().endswith(".jpg")]

    if not files:
        print(f"[ERROR] No .jpg files found in '{directory}'.")
        return {}, 0

    for file in sorted(files):
        person = extract_person_name(file)
        path = os.path.join(directory, file)

        try:
            image = face_recognition.load_image_file(path)
            # detect in training too (use same detector for consistency)
            locations = face_recognition.face_locations(image, model=detector_model)
            if not locations:
                # Try HOG as a fallback if CNN missed it (or vice versa)
                alt_model = "hog" if detector_model == "cnn" else "cnn"
                locations = face_recognition.face_locations(image, model=alt_model)

            if not locations:
                print(f"[WARN] No face found in {file}, skipping.")
                continue

            encs = face_recognition.face_encodings(image, locations, num_jitters=NUM_JITTERS)
            if encs:
                # If multiple faces in one training image, take the first
                encodings_by_person[person].append(encs[0])
                print(f"[INFO] Loaded face for '{person}' from {file}")
            else:
                print(f"[WARN] Could not compute encoding for {file}, skipping.")
        except Exception as e:
            print(f"[WARN] Failed to process {file}: {e}")

    total_encs = sum(len(v) for v in encodings_by_person.values())
    print(f"[INFO] Loaded {total_encs} encodings for {len(encodings_by_person)} people.")
    return encodings_by_person, total_encs

# ==============================
# Identify face with majority voting (+ distance tiebreak)
# ==============================
def identify_face(face_encoding, encodings_by_person, tolerance=TOLERANCE):
    best_name = "Unknown"
    best_votes = 0
    best_min_dist = float("inf")

    for name, enc_list in encodings_by_person.items():
        # Vectorized distances to all encodings for this person
        dists = face_recognition.face_distance(enc_list, face_encoding)
        votes = int(np.sum(dists <= tolerance))  # how many of their encodings consider it a match
        if votes > 0:
            min_dist = float(np.min(dists))
            # Pick the person with most votes; break ties by smaller min distance
            if (votes > best_votes) or (votes == best_votes and min_dist < best_min_dist):
                best_votes = votes
                best_min_dist = min_dist
                best_name = name

    return best_name

# ==============================
# Main
# ==============================
def main():
    encodings_by_person, total = load_known_faces(TRAINING_DIR, DETECTION_MODEL)
    if total == 0:
        print("[ERROR] No usable training encodings found. Exiting.")
        sys.exit(1)

    print(f"[INFO] Detection model: {DETECTION_MODEL.upper()}  "
          f"(toggle USE_CNN at top).")
    print(f"[INFO] Starting camera (index {CAMERA_INDEX})... press 'q' to quit.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    # Warm-up (some webcams need a moment)
    time.sleep(0.5)

    frame_idx = 0
    last_face_locations = []
    last_face_names = []

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Failed to grab frame from camera.")
            break

        # Downscale for speed
        small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Process only every Nth frame
        if frame_idx % PROCESS_EVERY_N_FRAMES == 0:
            face_locations = face_recognition.face_locations(rgb_small, model=DETECTION_MODEL)
            # If chosen detector misses, you can optionally try the other model as fallback
            # if not face_locations:
            #     alt_model = "hog" if DETECTION_MODEL == "cnn" else "cnn"
            #     face_locations = face_recognition.face_locations(rgb_small, model=alt_model)

            face_encodings = face_recognition.face_encodings(rgb_small, face_locations, num_jitters=NUM_JITTERS)

            names = []
            for enc in face_encodings:
                name = identify_face(enc, encodings_by_person, tolerance=TOLERANCE)
                names.append(name)

            # Cache results for frames we skip
            last_face_locations = face_locations
            last_face_names = names

        # Draw using the most recent results
        for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
            top = int(top * SCALE_BACK)
            right = int(right * SCALE_BACK)
            bottom = int(bottom * SCALE_BACK)
            left = int(left * SCALE_BACK)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 28), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow("Face Recognition (press 'q' to quit)", frame)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
