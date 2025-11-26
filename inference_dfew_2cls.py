# inference_dfew_2cls.py
import cv2
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path
from PIL import Image
from torchvision import models, transforms

# ----- globals so we only init once -----
_model = None
_device = None

_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

_classes = ["Positive", "Negative"]

def _choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def _init_model():
    """Lazy init: choose device, build ResNet101, load 2-class checkpoint."""
    global _model, _device
    if _model is not None:
        return

    _device = _choose_device()
    print("[DFEW] using device:", _device)

    m = models.resnet101(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)  # Positive / Negative

    root = Path(__file__).resolve().parent.parent  # go up from AffectFusion/ to its parent
    ckpt = root / "models" / "best_resnet101_2cls_fold1_16f_VAL.pth"

    sd = torch.load(str(ckpt), map_location=_device)
    m.load_state_dict(sd, strict=True)
    m.to(_device).eval()

    _model = m
    print("[DFEW] checkpoint loaded:", ckpt)

def predict_dfew_valence(face_bgr):
    """
    Input : face_bgr (cropped face, H,W,3, BGR from OpenCV)
    Output: (label_str, conf_float, probs_list)

      label_str  = "Positive" or "Negative"
      conf_float = probability of that class (0..1)
      probs_list = [p_pos, p_neg]
    """
    _init_model()

    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    x = _tf(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(probs.argmax())
    label = _classes[idx]
    conf = float(probs[idx])
    return label, conf, probs.tolist()
