import os
import cv2
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Configuration
SEQ_LENGTH = 75
MODEL_PATH = 'rf_daisee_model_GAMMA_53features.h5'
JSON_PATH = 'selected_features_53.json'

# Globals
MODEL = None
INDICES = None
FACE_MESH = None

# Thresholds
DEFAULT_THRESHOLDS = {
    "Boredom": 0.32,
    "Engagement": 0.65,
    "Confusion": 0.21,
    "Frustration": 0.17
}


# Function to load resources if not already loaded
def initialize():
    global MODEL, INDICES, FACE_MESH
    
    if MODEL is not None:
        return

    # Loading selected feature indices
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"Missing feature file: {JSON_PATH}")
    with open(JSON_PATH, 'r') as f:
        INDICES = np.array(json.load(f))

    # Loading model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    try:
        MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    # Initializing MediaPipe
    FACE_MESH = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


# Function to extract 53 features from each frame
def process_frame(frame):
    if FACE_MESH is None: return None
    if frame is None: return None
    
    # Ensuring RGB
    if frame.shape[-1] == 3:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        rgb_frame = frame

    results = FACE_MESH.process(rgb_frame)
    if not results.multi_face_landmarks:
        return None

    # Landmark logic
    landmarks = results.multi_face_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    nose = coords[1]
    left_eye = coords[33]
    right_eye = coords[263]
    
    eye_dist = np.linalg.norm(left_eye - right_eye)
    if eye_dist < 0.0001: eye_dist = 1.0
    
    centered = coords - nose
    normalized = centered / eye_dist
    
    try:
        return normalized.flatten()[INDICES]
    except IndexError:
        return None


def decide_label(preds, thresholds):
    scores = {
        "Boredom": preds[0],
        "Confusion": preds[1],
        "Engagement": preds[2],
        "Frustration": preds[3]
    }
    
    # Find active states (score > threshold)
    deviations = []
    for label, score in scores.items():
        thresh = thresholds.get(label, 0.5)
        deviation = score - thresh
        if deviation > 0:
            deviations.append({"label": label, "deviation": deviation})
    
    if not deviations:
        return "Neutral"
    
    # Sort by how far they passed the threshold
    deviations.sort(key=lambda x: x["deviation"], reverse=True)
    return deviations[0]["label"]


def get_engagement_label(frames, thresholds=None):
    # The function takes in a numpy array of 75 frames and pads if < 75,
    # and also takes in optional thresholds
    # It outputs the engagement label:"Engagement", "Boredom", "Confusion", "Frustration", or "Neutral" (none of the others)

    # Load model if not already loaded
    initialize()
    
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # Take only the last 75 frames in case of excess
    current_frames = frames[-SEQ_LENGTH:]
    
    processed_seq = []
    for frame in current_frames:
        feats = process_frame(frame)
        if feats is not None:
            processed_seq.append(feats)
    
    if not processed_seq:
        return "Neutral" # No faces detected in the buffer

    # Padding if there are not enough frames
    while len(processed_seq) < SEQ_LENGTH:
        # Pad the beginning with a copy of the first frame
        processed_seq.insert(0, processed_seq[0])

    # Inference
    input_tensor = np.array([processed_seq])
    predictions = MODEL.predict(input_tensor, verbose=0)[0]
    
    return decide_label(predictions, thresholds)