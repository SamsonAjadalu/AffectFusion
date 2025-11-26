# Real-Time Engagement Detection (Library Module)

This branch contains a library version of the engagement
detector, designed to be imported into other Python scripts, allowing you to
pass raw video frames and receive an engagement label.

------------------------------------------------------------------------

## 1. Repository Contents

-   **engagement_detector_lib.py**\
    The main library that handles model loading, feature
    extraction, data padding, and inference.

-   **rf_daisee_model_GAMMA_53features.h5**\
    The trained model's weights (LSTM).

-   **selected_features_53.json**\
    The specific MediaPipe face mesh indices required by the model.

------------------------------------------------------------------------

## 2. Prerequisites

1. Install **Python 3.11** alongside your current version of Python.
2. Create a virtual environment using **Python 3.11** and activate it.
3. Install dependencies:
```bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 3. How to Use

Ensure rf_daisee_model_GAMMA_53features.h5 and selected_features_53.json\
are in the same folder as engagement_detector_lib.py.\
Then, simply import the library and call `get_engagement_label`.\
The library manages the heavy lifting (loading models, processing
landmarks, and normalizing data) automatically.

### Expected Implementation
This assumes you are already gathering 75 frames to pass into the model:
``` python

'''
- frame_buffer is a list of 75 frames.
- thresholds is a dictionary such as this: 
- thresholds = {"Boredom": 0.32,
                "Engagement": 0.65,
                "Confusion": 0.21,
                "Frustration": 0.17
            }
'''

label = eng.get_engagement_label(frame_buffer, thresholds)
```

### Alternative Implementation
This implementation shows a simple example of how I continuously gathered a list of 75 frames to test the library and function call.
``` python
import cv2
import engagement_detector_lib as eng

# Setting up webcam
cap = cv2.VideoCapture(0)
frame_buffer = []

while True:
    ret, frame = cap.read()
    if not ret: break

    # Maintain a buffer of frames
    frame_buffer.append(frame)

    # Call the function
    # Note: For video smoothness, you should probably only call this every, say, 10-20 frames, and not necessarily
    # for each new frame collected.
    if len(frame_buffer) % 10 == 0:

        # Example thresholds
        thresholds = {
            "Boredom": 0.32,
            "Engagement": 0.65, # Lower value = more sensitive
            "Confusion": 0.21,
            "Frustration": 0.17
        }

        # Returns: "Engagement", "Boredom", "Confusion", "Frustration", or "Neutral"
        label = eng.get_engagement_label(frame_buffer, thresholds)

        print(f"Current affective state: {label}")
```

------------------------------------------------------------------------

## 4. API Reference

### `eng.get_engagement_label(frames, thresholds=None)`

**Parameters:**

-   **frames (list of numpy.ndarray)**\
    A list of standard OpenCV images.\
    The model expects **75 frames**.
    -   If \< 75 frames: auto-pads\
    -   If \> 75 frames: auto-trims to most recent 75
-   **thresholds (dict, optional)**\
    Override default sensitivity values.

**Returns:**\
`str` --- One of:

-   Engagement\
-   Boredom\
-   Confusion\
-   Frustration\
-   Neutral