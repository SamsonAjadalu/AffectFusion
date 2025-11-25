# AffectFusion

A multimodal deep learning framework for emotion recognition – with **image-only baselines on DFEW**, plus tools to:

- Train 2D CNNs on DFEW with different label presets (7, 5, 4, or 2 classes).
- Evaluate on the official DFEW test splits.
- Inspect class imbalance under different mappings.
- Test trained models on your own face images (with face detection and cropping).
- Run a **live webcam demo** for the 2-class Positive/Negative model.

---

## 1. Setup

Install Python deps (GPU PyTorch wheels by default):

```bash
pip install -r requirements.txt
```

`requirements.txt` pins recent `torch`, `torchvision`, `numpy`, `scikit-learn`, etc., and uses the CUDA 12.1 wheel index for NVIDIA GPUs.

---

## 2. Project layout

All scripts expect this layout (keep **data** and **models** as siblings of the repo):

```text
parent/
├─ AffectFusion/            # code
├─ dfew/DFEW-part2/         # DFEW dataset root
└─ models/                  # checkpoints (created automatically)
```

DFEW is assumed to be **Part 2** with pre-extracted 16-frame clips and official splits:

```text
../dfew/DFEW-part2/
  ├─ Clip/
  │    └─ clip_224x224_16f/<clip_id>/*.jpg   # 16 RGB frames per clip
  └─ EmoLabel_DataSplit/
       ├─ train(single-labeled)/ set_1.csv ... set_5.csv
       └─ test(single-labeled)/  set_1.csv ... set_5.csv
```
### Pretrained checkpoint

If you just want to run the webcam demo or test on your own images without training first, you can use a pretrained 2-class (Positive/Negative) ResNet-101 checkpoint.

- Download from Google Drive:  
- Download checkpoint: [Google Drive link](https://drive.google.com/file/d/14BUmlN3PLFsbuyaPsYW4kuaVoBaW7I1o/view?usp=drive_link)

After downloading, place the file here (relative to the repo root):

```text
../models/best_resnet101_2cls_fold1_16f_VAL.pth
```
Then you can run:
```text
python3 webcam_live_2cls.py \
  --ckpt ../models/best_resnet101_2cls_fold1_16f_VAL.pth \
  --arch resnet101
```
---

## 3. Label presets (class mappings)

All training / eval scripts share the same label presets:

Base 7-class indexing (from DFEW splits):

```text
0 = Happy
1 = Sad
2 = Neutral
3 = Angry
4 = Surprise
5 = Disgust
6 = Fear
```

Presets:

- **`--labels 7`**:  
  `["Happy","Sad","Neutral","Angry","Surprise","Disgust","Fear"]` (no merge).

- **`--labels 5`**  (`5cls`):  
  `["Happy","Sad","Neutral","Angry","Surprise"]`  
  - Disgust → Angry  
  - Fear → Sad  

- **`--labels 4`**  (`4cls`):  
  `["Positive","Negative","Neutral","Surprise"]`  
  - Positive: Happy  
  - Negative: Sad, Angry, Disgust, Fear  
  - Neutral: Neutral  
  - Surprise: Surprise  

- **`--labels 2`**  (`2cls`, main setup):  
  `["Positive","Negative"]`  
  - Positive: Happy, Neutral, Surprise  
  - Negative: Sad, Angry, Disgust, Fear  

You can inspect how many clips end up in each class for each split and fold with:

```bash
python analyze_dfew_class_counts.py
```

This prints counts for 7/5/4/2-class mappings per fold (train + test).

---

## 4. 2D DFEW baseline (ResNet-101 / ResNet-50)

### 4.1 Train on DFEW (2D CNN)

Main training script: `train_only_dfew_16f.py`.

It trains a **2D ResNet** on **single frames sampled from each 16-frame clip**:

- One random frame per clip per epoch for training (`per_clip=1`).
- For validation, it aggregates logits over all 16 frames (`per_clip=16`) at evaluation time.

**Basic usage (2-class Positive/Negative, ResNet-101, fold 1):**

```bash
cd AffectFusion

python train_only_dfew_16f.py   --root ../dfew/DFEW-part2   --out  ../models   --fold 1   --labels 2
```

Key arguments:

- `--root`: DFEW root (`../dfew/DFEW-part2` by default).
- `--out`: where to save checkpoints (`../models` by default).
- `--fold`: which DFEW fold (1–5).
- `--labels {7,5,4,2}`: which label preset to use (default: `2`).
- `--epochs`: number of training epochs.
- `--batch`, `--workers`, `--val_frac`, `--seed`, `--lr`.

By default, the script uses **ResNet-101** pretrained on ImageNet:

```python
m = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
```

If you want to train with **ResNet-50** instead, just edit this block:

```python
m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
```

(You can also add an `--arch` CLI flag if you want to switch in the command line, but by default the script is clean and fixed to ResNet-101.)

The script:

- Uses `AdamW` optimizer + weight decay.
- Uses mixed precision (`torch.amp.autocast` + `GradScaler`) when available.
- Keeps `cudnn.benchmark = True` for speed.
- Tracks **WAR** and **UAR** on the validation set each epoch, and saves the **best checkpoint by VAL UAR**.

Checkpoints are saved as:

```text
../models/best_resnet101_<suffix>_fold<F>_16f_VAL.pth
```

Where `<suffix>` is `7cls`, `5cls`, `4cls`, or `2cls` depending on `--labels`.

---

### 4.2 Evaluate on DFEW test split (2D)

Use `eval_only_dfew_16f.py` to evaluate a saved checkpoint on the official **test(single-labeled)** split.

Example (2-class, fold 1):

```bash
python eval_only_dfew_16f.py   --root ../dfew/DFEW-part2   --models_dir ../models   --fold 1   --labels 2   --ckpt ../models/best_resnet101_2cls_fold1_16f_VAL.pth
```

If you omit `--ckpt`, the script will look for:

```text
../models/best_resnet101_<suffix>_fold<F>_16f_VAL.pth
```

It will:

- Load the test split, applying the same label mapping as during training.
- For each clip, run all its frames through the model and **average logits per clip**.
- Compute and print:

  - **WAR** – Weighted Accuracy Rate (normal accuracy).
  - **UAR** – Unweighted Accuracy Rate (macro recall).

---

## 5. Class distribution analysis (DFEW)

`analyze_dfew_class_counts.py` gives you a nice summary of how many clips fall into each class under different mappings (7/5/4/2), for every fold and for both train and test splits.

Run:

```bash
python analyze_dfew_class_counts.py
```

You’ll see outputs like:

```text
FOLD 1

--- TRAIN split ---
  7-class original:
    Happy    : ...
    ...
  4-class Pos/Neg/Neu/Sup:
    Positive : ...
    ...

--- TEST split ---
  ...
```

This is useful for making decisions like:

- Whether to merge rare classes.
- Whether to use 2-class vs 4-class setups.

---

## 6. Testing on your own face images

Use `test_custom_faces.py` to evaluate a checkpoint on your own images.  
It:

- Runs **OpenCV face detection**.
- Crops the largest face with some padding.
- Applies the same Resize → CenterCrop → Normalize as DFEW.
- Predicts emotions for each image.
- Optionally computes accuracy if your folder name matches a class name.

### 6.1 Folder structure

By default:

```text
custom_data/
  positive/
      img_001.jpg
      img_002.jpg
      ...
  negative/
      img_101.jpg
      ...
```

…but it’s flexible: the script will try to infer the expected label from the folder name:

- For 2-class models: `positive/`, `pos/`, `negative/`, `neg/` etc.
- For 4/5/7-class models: `happy/`, `sad/`, `neutral/`, `angry/`, `surprise/`, `disgust/`, `fear/`, etc.

If it recognizes the folder name as one of the class names, it will also compute accuracy for that folder.

### 6.2 Usage

```bash
python test_custom_faces.py   --custom_root custom_data   --ckpt ../models/best_resnet101_2cls_fold1_16f_VAL.pth   --labels 2   --arch resnet101   --debug_crops debug_crops
```

Arguments:

- `--custom_root`: root folder with one subfolder per “class”.
- `--ckpt`: path to a trained `.pth` file (any of your DFEW models).
- `--labels {7,5,4,2}`: must match how the model was trained.
- `--arch {resnet50,resnet101}`: must match the backbone you used.
- `--debug_crops`: where to save the cropped 224×224 faces for visual inspection.

The script will log predictions like:

```text
=== Folder: positive (expected: Positive) ===
  img_001.jpg -> Positive
  img_002.jpg -> Negative
  ...
  Accuracy in positive: 18/20 = 0.900
```

So you can manually collect a small custom dataset (your own selfies) and see how well the model generalizes.

---

## 7. Live webcam demo (2-class)

`webcam_live_2cls.py` lets you run the **2-class Positive/Negative model in real time** on a webcam stream.

It:

- Opens a webcam (`cv2.VideoCapture`).
- Detects the largest face in each frame, crops and normalizes it.
- Runs the 2-class model, keeps a moving window of logits, and averages them over the last N frames (temporal smoothing).
- Overlays the predicted label + confidence on the video.

Usage example:

```bash
python webcam_live_2cls.py   --ckpt ../models/best_resnet101_2cls_fold1_16f_VAL.pth   --arch resnet101   --camera 0   --window 16
```

Args:

- `--ckpt`: path to your 2-class checkpoint.
- `--arch {resnet50,resnet101}`: must match the backbone.
- `--camera`: webcam index (0 by default).
- `--window`: number of frames used for temporal averaging (e.g., 16).

Press **`q`** in the window to quit.

---

## 8. (Optional) 3D video baseline

There is also a **3D CNN training script** (`train_dfew_3d_r3d18.py`) which uses a 3D ResNet-18 (R3D-18) on fixed 16-frame clips from DFEW. The idea:

- Instead of treating frames independently, it feeds the whole 16-frame clip (T×H×W) into a 3D model.
- You can use it to compare 2D vs 3D temporal modeling.

The CLI and label handling are analogous to the 2D script:

```bash
python train_dfew_3d_r3d18.py   --root ../dfew/DFEW-part2   --out  ../models   --fold 1   --labels 2
```

Because 3D models are heavier:

- Start with smaller batch sizes.

---

## 9. CLI commands

**Train (2D, ResNet-101, Positive/Negative):**

```bash
python train_only_dfew_16f.py --fold 1 --labels 2
```

**Eval on DFEW test (2D):**

```bash
python eval_only_dfew_16f.py --fold 1 --labels 2
```

**Class counts / imbalance:**

```bash
python analyze_dfew_class_counts.py
```

**Test on your own images:**

```bash
python test_custom_faces.py --labels 2
```

**Live webcam demo (2-class):**

```bash
python webcam_live_2cls.py --labels 2   # (labels implied by the checkpoint)
```

You can switch to 4/5/7 classes by training with `--labels 4`, `--labels 5`, or `--labels 7`, and then matching that flag for evaluation and custom testing. ResNet-50 vs ResNet-101 is controlled by the `arch` you use in the scripts (or by editing the backbone line in `train_only_dfew_16f.py` if you want to change the default).
