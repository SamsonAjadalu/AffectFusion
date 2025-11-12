# AffectFusion
A multimodal deep learning framework for emotion recognition.

## What’s here
**Image-only baseline on DFEW** using ResNet‑50 on pre-extracted frames (`clip_224x224_16f`).  
Clip-level evaluation = average logits over frames. Metrics: **WAR** (accuracy) and **UAR** (macro recall).  
We **save the best model by VAL UAR** to handle class imbalance.

## Setup
```bash
pip install -r requirements.txt
```
**Keep data & models outside the repo** (so they aren’t pushed). Default paths resolve to *siblings* of this repo:
```
parent/
├─ AffectFusion/            # this repo
├─ dfew/DFEW-part2/        # dataset root (you create/extract)
└─ models/                 # checkpoints (created automatically)
```
## Dataset (DFEW)
- Homepage: [DFEW dataset](https://dfew-dataset.github.io/index.html)
- Download: follow the instructions on the DFEW page to obtain **Part 2** (clips/frames + splits).
- Place the extracted folder at `../dfew/DFEW-part2/` to match the defaults in this repo.

Expected dataset layout:
```
../dfew/DFEW-part2/
  ├─ Clip/clip_224x224_16f/<clip_id>/*.jpg
  └─ EmoLabel_DataSplit/
       ├─ train(single-labeled)/ set_1.csv ... set_5.csv
       └─ test(single-labeled)/  set_1.csv ... set_5.csv
```

## Train (defaults)
```bash
python train_only_dfew_16f.py --fold 1
# saves: ../models/best_resnet50_fold1_16f_VAL.pth
```

## Test (defaults)
```bash
python test_dfew_16f.py --fold 1
# uses: ../models/best_resnet50_fold1_16f_VAL.pth
# prints: [TEST] WAR ...  UAR ...
```

## CLI cheatsheet (override any default)
`--root ../dfew/DFEW-part2`, `--out ../models`, `--fold 1..5`,  
`--epochs`, `--batch`, `--workers`, `--seed`, `--val_frac`,  
`--lr`, `--weight_decay`.

### Examples
```bash
python train_only_dfew_16f.py --fold 1 --epochs 200 --batch 128 --out ../models_exp1
python test_dfew_16f.py        --fold 1 --ckpt ../models_exp1/best_resnet50_fold1_16f_VAL.pth
```

## Tips (quick wins)
- Sample more frames per clip for training; add stronger augmentation.
- Try class-balanced/focal loss and cosine LR for longer runs.
- Use larger input (e.g., 256) if VRAM allows; face-crop can improve UAR.

## Metrics refresher
- **WAR**: overall accuracy (higher is better).
- **UAR**: macro recall (average per-class recall; higher is better). We **select by UAR**.
- **Train loss**: optimized on training batches (lower is better on train).
