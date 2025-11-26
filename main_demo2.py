# main_demo1.py
import cv2
import time
import threading
from collections import deque

from crop_utils import crop_face
from inference_dfew_2cls import predict_dfew_valence
import engagement_detector_lib as eng_det


# ---------- Helpers for valence ----------

def run_valence(frame_bgr):
    """
    Runs DFEW 2-class model on a single frame.
    Returns (label, conf, display_text).
    label can be: 'Positive', 'Negative', or 'Uncertain'
    """
    face = crop_face(frame_bgr)

    # 1) no clear face -> treat as 'Uncertain' so fusion can handle it
    if face is None:
        val_label = "Uncertain"
        val_conf = 0.0
        val_text = "Uncertain (no clear face)"
        print("[DFEW]", val_text)
        return val_label, val_conf, val_text

    # 2) normal prediction
    raw_label, conf, _ = predict_dfew_valence(face)

    # 3) low-confidence -> also 'Uncertain'
    THRESH = 0.60  # you can tweak this
    if conf < THRESH:
        val_label = "Uncertain"
        val_text = f"Uncertain ({raw_label} {conf*100:.1f}%)"
    else:
        val_label = raw_label
        val_text = f"{val_label}  {conf*100:.1f}%"

    print("[DFEW]", val_text)
    return val_label, conf, val_text


# ---------- Engagement worker (runs in background) ----------

def start_engagement_worker(frame_buffer, thresholds):
    """
    Spawns a background thread that periodically reads frames from
    frame_buffer and updates a shared last_eng_label string.
    """
    state = {
        "last_eng_label": "Neutral",
        "stop": False,
    }

    def worker():
        # small warm-up delay so buffer gets some frames
        time.sleep(1.0)

        while not state["stop"]:
            if len(frame_buffer) > 0:
                frames = list(frame_buffer)  # snapshot
                try:
                    label = eng_det.get_engagement_label(frames, thresholds)
                    state["last_eng_label"] = label
                    print("[ENG ]", label)
                except Exception as e:
                    print("[ENG ] error:", e)
            # run periodically; ~2 Hz here
            time.sleep(0.5)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return state


# ---------- Main demo ----------

def main():
    cap = cv2.VideoCapture(0)
    # helpful to keep things lighter
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not cap.isOpened():
        print("Could not open webcam")
        return

    # Keep last 75 frames for engagement model
    frame_buffer = deque(maxlen=75)

    thresholds = {
        "Boredom": 0.32,
        "Engagement": 0.65,
        "Confusion": 0.21,
        "Frustration": 0.17,
    }

    # start background engagement worker
    eng_state = start_engagement_worker(frame_buffer, thresholds)

    last_val_label = "Uncertain"
    last_val_conf = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # work on a smaller frame for both models
        frame_proc = cv2.resize(frame, (640, 360))

        # push into buffer for engagement worker
        frame_buffer.append(frame_proc)

        # valence (runs every frame; you can also throttle if needed)
        last_val_label, last_val_conf, val_text = run_valence(frame_proc)

        # engagement label is updated asynchronously by the thread
        eng_text = f"Engagement: {eng_state['last_eng_label']}"

        # choose color based on label
        if last_val_label == "Positive":
            color = (0, 255, 0)      # green
        elif last_val_label == "Negative":
            color = (0, 0, 255)      # red
        else:
            color = (0, 255, 255)    # yellow for 'Uncertain'

        # overlay on frame
        cv2.putText(
            frame_proc,
            val_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame_proc,
            eng_text,
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Multimodal Demo (DFEW + Engagement)", frame_proc)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    eng_state["stop"] = True
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
