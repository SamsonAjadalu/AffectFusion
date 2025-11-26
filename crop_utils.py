# crop_utils.py
import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_face(frame_bgr):
    """
    Detect the largest face and return a padded crop (BGR).
    If no face is found, return None so the caller can treat it as 'uncertain'.
    """
    h, w = frame_bgr.shape[:2]
    faces = face_cascade.detectMultiScale(
        frame_bgr,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return None  # <--- changed: don't return the whole frame

    # pick largest face
    x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
    cx = x + fw // 2
    cy = y + fh // 2
    side = int(max(fw, fh) * 1.5)

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, cx + side // 2)
    y2 = min(h, cy + side // 2)

    return frame_bgr[y1:y2, x1:x2]
