import os, glob, csv, random, shutil
from pathlib import Path

# ---------- CONFIG ----------
N_PER_CLASS = 100                      # how many clips per class
DATA_ROOT   = "dfew/DFEW-part2"        # relative to ../emotion
OUT_DIR     = "dfew_example_frames"    # will be created under ../emotion
# -----------------------------

# base = /media/.../emotion   (parent of AffectFusion)
base = Path(__file__).resolve().parent.parent
dfew_root  = (base / DATA_ROOT).resolve()
frames_root = dfew_root / "Clip" / "clip_224x224_16f"
splits_root = dfew_root / "EmoLabel_DataSplit"
out_root    = (base / OUT_DIR).resolve()

os.makedirs(out_root / "Positive", exist_ok=True)
os.makedirs(out_root / "Negative", exist_ok=True)

print("DFEW root:", dfew_root)
print("Saving samples to:", out_root)

# ---------- label helpers (same as training) ----------
BASE_EMOTIONS = ["Happy","Sad","Neutral","Angry","Surprise","Disgust","Fear"]
two_class_map = {
    0: 0,  # Happy    -> Positive
    1: 1,  # Sad      -> Negative
    2: 0,  # Neutral  -> Positive
    3: 1,  # Angry    -> Negative
    4: 0,  # Surprise -> would be Positive, but we DROP 4
    5: 1,  # Disgust  -> Negative
    6: 1,  # Fear     -> Negative
}

def parse_label(tok: str) -> int:
    t = tok.strip()
    if t.isdigit():
        idx = int(t) - 1
        return idx
    t = t.lower()
    alias = {"happiness":"happy","surprised":"surprise","neutrality":"neutral","anger":"angry"}
    t = alias.get(t, t)
    lut = {e.lower(): i for i, e in enumerate(BASE_EMOTIONS)}
    return lut[t]

def load_split(path: str):
    items = []
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            rdr = csv.reader(f)
            for row in rdr:
                if not row:
                    continue
                first = str(row[0]).strip().lower()
                last  = str(row[-1]).strip().lower()
                if first in {"video_name","video","video_id","clip","id","name"} and \
                   last  in {"label","emotion","class"}:
                    continue
                clip = str(row[0]).strip()
                if not clip:
                    continue
                base_idx = parse_label(str(row[-1]))
                if base_idx == 4:        # drop Surprise
                    continue
                lab = two_class_map[base_idx]  # 0=Positive, 1=Negative
                items.append((clip, lab))
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p for p in line.replace(",", " ").split() if p]
                base_idx = parse_label(parts[-1])
                if base_idx == 4:
                    continue
                lab = two_class_map[base_idx]
                items.append((parts[0], lab))
    return items

def find_split(base_dir, fold, split_kind):
    d = os.path.join(base_dir, f"{split_kind}(single-labeled)")
    cand = sorted(
        glob.glob(os.path.join(d, f"*{fold}*.csv")) +
        glob.glob(os.path.join(d, f"*{fold}*.txt"))
    )
    if not cand:
        raise FileNotFoundError(f"no split file for fold {fold} in {d}")
    return cand[0]

# ---------- collect clip IDs from ALL train folds ----------
pos_clips = set()
neg_clips = set()

for fold in [1,2,3,4,5]:
    path = find_split(str(splits_root), fold, "train")
    items = load_split(path)
    for cid, lab in items:
        if lab == 0:
            pos_clips.add(cid)
        else:
            neg_clips.add(cid)

pos_clips = list(pos_clips)
neg_clips = list(neg_clips)
random.shuffle(pos_clips)
random.shuffle(neg_clips)

print(f"Found {len(pos_clips)} positive clips, {len(neg_clips)} negative clips")

# ---------- helper to copy one random frame per clip ----------
def copy_examples(clip_ids, class_name, n):
    out_dir = out_root / class_name
    copied = 0
    for cid in clip_ids:
        if copied >= n:
            break
        d = frames_root / str(cid)
        if not d.is_dir():
            d2 = frames_root / str(cid).zfill(5)
            if d2.is_dir():
                d = d2
            else:
                continue
        ims = sorted(
            glob.glob(str(d / "*.jpg")) +
            glob.glob(str(d / "*.jpeg")) +
            glob.glob(str(d / "*.png"))
        )
        if not ims:
            continue
        frame_path = random.choice(ims)
        dest_name = f"{cid}_{os.path.basename(frame_path)}"
        shutil.copy(frame_path, out_dir / dest_name)
        copied += 1
    print(f"Copied {copied} {class_name} frames to {out_dir}")

copy_examples(pos_clips, "Positive", N_PER_CLASS)
copy_examples(neg_clips, "Negative", N_PER_CLASS)

print("Done.")
