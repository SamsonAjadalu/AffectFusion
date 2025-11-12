import os, glob, csv, random, argparse
from collections import defaultdict
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path


# ---------------- Labels (matches your annotation 1..7) ----------------
# 1→Happy, 2→Sad, 3→Neutral, 4→Angry, 5→Surprise, 6→Disgust, 7→Fear
EMOTIONS = ["Happy","Sad","Neutral","Angry","Surprise","Disgust","Fear"]
NUM_CLASSES = len(EMOTIONS)

def parse_label(tok: str) -> int:
    t = tok.strip()
    if t.isdigit():
        idx = int(t) - 1                # 1..7 -> 0..6
        if not (0 <= idx < NUM_CLASSES):
            raise ValueError(f"Label out of range: {tok}")
        return idx
    t = t.lower()
    alias = {"happiness":"happy","surprised":"surprise","neutrality":"neutral","anger":"angry"}
    t = alias.get(t, t)
    lut = {e.lower(): i for i, e in enumerate(EMOTIONS)}
    if t in lut: return lut[t]
    raise ValueError(f"Unrecognized label token: {tok}")

def load_split(path: str):
    """Reads CSV/TXT. For CSV, assumes first col=clip id, last col=label."""
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
                # Skip header like: video_name,label
                if first in {"video_name","video","video_id","clip","id","name"} and last in {"label","emotion","class"}:
                    continue
                clip = str(row[0]).strip()
                lab  = parse_label(str(row[-1]))
                if clip:
                    items.append((clip, lab))
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): 
                    continue
                parts = [p for p in line.replace(",", " ").split() if p]
                items.append((parts[0], parse_label(parts[-1])))
    if not items:
        raise RuntimeError(f"No items in {path}")
    return items

def find_split(base_dir, fold, split_kind):
    d = os.path.join(base_dir, f"{split_kind}(single-labeled)")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"missing dir: {d}")
    cand = sorted(
        glob.glob(os.path.join(d, f"*{fold}*.csv")) +
        glob.glob(os.path.join(d, f"*{fold}*.txt"))
    )
    if not cand:
        raise FileNotFoundError(f"no split file for fold {fold} in {d}")
    return cand[0]

# ---------------- Stratified train/val split (from TRAIN only) ----------------
def stratified_split(items, val_frac=0.1, seed=42):
    rng = random.Random(seed)
    by_lab = defaultdict(list)
    for cid, lab in items:
        by_lab[lab].append((cid, lab))
    tr, va = [], []
    for lab, lst in by_lab.items():
        rng.shuffle(lst)
        k = max(1, int(round(len(lst) * val_frac))) if len(lst) > 2 else 1
        va.extend(lst[:k]); tr.extend(lst[k:])
    rng.shuffle(tr); rng.shuffle(va)
    return tr, va

# ---------------- Dataset ----------------
class DFEWFrames(Dataset):
    """
    Train: per_clip=1 (one random frame/clip/epoch)
    Eval : per_clip=16 (cover all 16 frames)
    """
    def __init__(self, frames_root, split_items, tfm, per_clip=1):
        self.tfm = tfm
        self.per_clip = per_clip
        self.clips = []
        missing = []
        for cid, lab in split_items:
            d = os.path.join(frames_root, str(cid))
            if not os.path.isdir(d):
                d2 = os.path.join(frames_root, str(cid).zfill(5))
                d = d2 if os.path.isdir(d2) else d
            ims = sorted(
                glob.glob(os.path.join(d, "*.jpg")) +
                glob.glob(os.path.join(d, "*.jpeg")) +
                glob.glob(os.path.join(d, "*.png"))
            )
            if ims:
                self.clips.append((os.path.basename(d), lab, ims))
            else:
                missing.append(cid)
        if missing:
            print(f"[warn] missing {len(missing)} clips (e.g., {missing[:3]})")
        if not self.clips:
            raise RuntimeError("No frames found. Check --root and the 16f folder.")

    def __len__(self):
        return len(self.clips) * self.per_clip

    def __getitem__(self, i):
        ci = i // self.per_clip
        cid, lab, ims = self.clips[ci]
        idx = random.randrange(len(ims))
        img = Image.open(ims[idx]).convert("RGB")
        x = self.tfm(img)
        return x, lab, cid

# ---------------- Metrics ----------------
def metrics_war_uar(y_true, y_pred, K=7):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    war = float((y_true == y_pred).mean())
    rec = []
    for k in range(K):
        m = (y_true == k)
        if m.sum(): rec.append(float((y_pred[m] == k).mean()))
    uar = float(np.mean(rec)) if rec else 0.0
    return war, uar

def eval_loader(model, dloader, device):
    model.eval(); clip_logits = defaultdict(lambda: None); clip_lab = {}
    with torch.no_grad():
        for x, y, cid in dloader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda" if device.type=="cuda" else "cpu"):
                logits = model(x)  # (B,7)
            for i, c in enumerate(cid):
                clip_lab[c] = int(y[i])
                cur = logits[i].detach().cpu()
                clip_logits[c] = cur if clip_logits[c] is None else (clip_logits[c] + cur)
    y_true, y_pred = [], []
    for c, logit_sum in clip_logits.items():
        y_true.append(clip_lab[c]); y_pred.append(int(logit_sum.argmax().item()))
    return metrics_war_uar(y_true, y_pred, K=NUM_CLASSES)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dfew/DFEW-part2", help="path to DFEW-part2 (repo-relative default)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--val_frac", type=float, default=0.1, help="fraction of TRAIN used as validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="models", help="folder to save checkpoints")
    args = ap.parse_args()
    base = Path(__file__).resolve().parent.parent
    args.root = str((base / args.root).resolve())
    args.out  = str((base / args.out).resolve())
    os.makedirs(args.out, exist_ok=True)



    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    frames_root = os.path.join(args.root, "Clip", "clip_224x224_16f")
    splits_root = os.path.join(args.root, "EmoLabel_DataSplit")

    # Load TRAIN only, then make (train,val)
    train_file = find_split(splits_root, args.fold, "train")
    train_all  = load_split(train_file)
    train_items, val_items = stratified_split(train_all, val_frac=args.val_frac, seed=args.seed)

    # Transforms
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # Datasets / Loaders
    ds_tr = DFEWFrames(frames_root, train_items, tf_train, per_clip=1)
    ds_va = DFEWFrames(frames_root, val_items,   tf_eval,  per_clip=16)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,  num_workers=args.workers,
                       pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       pin_memory=True)

    # Model
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.to(device)

    opt    = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda" if device.type=="cuda" else "cpu")
    torch.backends.cudnn.benchmark = True

    best_uar = -1.0
    best_path = os.path.join(args.out, f"best_resnet50_fold{args.fold}_16f_VAL.pth")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        m.train(); running = 0.0; n = 0
        for x, y, _ in dl_tr:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda" if device.type=="cuda" else "cpu"):
                logits = m(x)
                loss = nn.CrossEntropyLoss()(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item() * x.size(0); n += x.size(0)

        # ---- Validate on VAL (not test) ----
        war, uar = eval_loader(m, dl_va, device)
        print(f"epoch {epoch:02d}  train_loss {running/max(n,1):.4f}  VAL_WAR {war:.4f}  VAL_UAR {uar:.4f}")

        if uar > best_uar:
            best_uar = uar
            torch.save(m.state_dict(), best_path)
            print(f"  ✅ new best VAL_UAR {best_uar:.4f} -> saved {best_path}")

    print(f"done. best VAL UAR={best_uar:.4f}")
    if os.path.isfile(best_path):
        print(f"best checkpoint: {best_path}")

if __name__ == "__main__":
    main()
