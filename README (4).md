# AffectFusion (Demo 1 – Final Branch)

**webcam emotion demo** 

---

## 1. Folder layout

layout:

```text
parent/
├─ AffectFusion/                 
└─ models/
     └─ best_resnet101_2cls_fold1_16f_VAL.pth
```



## 2. Demo files

- `crop_utils.py`  
  - Detect largest face with OpenCV Haar cascade.  
  - Returns a padded face crop in BGR format.

- `inference_dfew_2cls.py`  
  - Wraps the **2-class DFEW ResNet-101** model.  
  - On first call:
    - chooses device (`mps` / `cuda` / `cpu`),
    - loads `../models/best_resnet101_2cls_fold1_16f_VAL.pth`,
    - sets the model to eval mode.
  - API:

    ```python
    label, conf, probs = predict_dfew_valence(face_bgr)
    # label ∈ {"Positive", "Negative"}
    # conf  = probability of label (0..1)
    # probs = [p_pos, p_neg]
    ```

- `main_demo.py`  
  - Opens the webcam (`cv2.VideoCapture(0)`),
  - Crops the face using `crop_utils.crop_face(...)`,
  - Calls `predict_dfew_valence(...)`,

---

## 3. Pretrained checkpoint

Use the 2-class DFEW model:

- `best_resnet101_2cls_fold1_16f_VAL.pth` (Positive / Negative)

Place it here relative to the repo root:

```text
../models/best_resnet101_2cls_fold1_16f_VAL.pth
```

---

## 4. Setup



```bash
pip install -r requirements.txt
```

---

## 5. Run Demo 1 (webcam valence)


```bash
python main_demo.py
```

