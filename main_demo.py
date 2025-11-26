# main_demo1.py
import cv2
from crop_utils import crop_face
from inference_dfew_2cls import predict_dfew_valence

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) crop face once (this same face can later be sent to other models)
        face = crop_face(frame)

        # 2) run your DFEW 2-class model
        label, conf, _ = predict_dfew_valence(face)
        text = f"{label}  {conf*100:.1f}%"
        print(" [DFEW] ", text)

        # 3) overlay on original frame
        cv2.putText(frame, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if label == "Positive" else (0, 0, 255),
                    2, cv2.LINE_AA)

        cv2.imshow("DFEW Valence Demo (Demo1)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
