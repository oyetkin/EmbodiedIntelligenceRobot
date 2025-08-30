import cv2
import sys
import os
import re

def get_next_filename(base_filename):
    """
    Given a base filename like 'sandra.jpg' or 'sandra2.jpg',
    return the next available numbered filename in the sequence.
    """
    name, ext = os.path.splitext(base_filename)

    # If filename already ends with a number, extract base and start from there
    match = re.match(r"^(.*?)(\d+)$", name)
    if match:
        base = match.group(1)
        start_num = int(match.group(2)) + 1
    else:
        base = name
        start_num = 1

    # Scan directory for existing files
    existing = set(os.listdir("."))

    # Find first available filename
    num = start_num
    while True:
        candidate = f"{base}{num}{ext}"
        if candidate not in existing:
            return candidate
        num += 1

def preview_and_capture(initial_filename, camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    filename = initial_filename
    first_save = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Show instructions on the preview
        display_text = f"Press 's' to save {filename} | Press ESC to quit"
        preview = frame.copy()
        cv2.putText(preview, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Camera Preview", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            break
        elif key == ord('s'):
            # Save current frame
            cv2.imwrite(filename, frame)
            print(f"Saved frame to {filename}")

            if first_save:
                first_save = False
            # Generate next filename
            filename = get_next_filename(initial_filename)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    filename = "output.jpg"
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    preview_and_capture(filename)
