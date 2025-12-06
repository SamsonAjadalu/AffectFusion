# audio_inference.py

import time
from io import BytesIO

import numpy as np
import sounddevice as sd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import librosa

# ------- config -------
DURATION = 5          # seconds per audio window
SAMPLE_RATE = 22050
CLASSES = ["positive", "negative", "surprised"]

# ------- globals -------
_model = None
_last_emotion = None
_last_conf = None
_last_time = None


def _load_model():
    """Lazy-load the Keras audio model once."""
    global _model
    if _model is None:
        print("[AUDIO] loading model audio_model1.keras ...")
        _model = tf.keras.models.load_model("audio_model1.keras")
        print("[AUDIO] model loaded.")


def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    print(f"[AUDIO] Recording {duration} seconds of audio...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def _waveform_to_image(audio, sample_rate=SAMPLE_RATE, n=100, dpi=100):
    """Same plotting logic you used before."""
    times = librosa.times_like(audio, sr=sample_rate)

    fig = plt.figure(figsize=(n / dpi, n / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    plt.plot(times, audio, linewidth=0.6)

    plt.xlim(times[0], times[-1])
    plt.ylim(audio.min() * 1.1, audio.max() * 1.1)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                pad_inches=0, dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("L")          # grayscale
    img = img.resize((n, n), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_emotion_from_audio(audio, sample_rate=SAMPLE_RATE):
    """Return (label, confidence, probs)."""
    _load_model()
    features = _waveform_to_image(audio, sample_rate, n=100, dpi=100)
    preds = _model.predict(features, verbose=0)[0]

    idx = int(np.argmax(preds))
    emotion = CLASSES[idx]
    conf = float(preds[idx])
    return emotion, conf, preds


def audio_loop():
    """
    Background loop: keep recording + predicting.
    Updates globals _last_emotion / _last_conf.
    """
    global _last_emotion, _last_conf, _last_time
    _load_model()

    print(f"[AUDIO] Live loop started (window = {DURATION}s). Ctrl+C stops main script.")

    while True:
        audio = record_audio(DURATION, SAMPLE_RATE)
        emotion, conf, _ = predict_emotion_from_audio(audio, SAMPLE_RATE)

        _last_emotion = emotion
        _last_conf = conf
        _last_time = time.time()

        print(f"[AUDIO] {emotion.upper()} {conf*100:.1f}%")
        print("-" * 24)


def get_last_audio_result():
    """
    Called from main_demo1 to fetch last audio prediction.

    Returns:
        (emotion:str or None, confidence:float or None, timestamp:float or None)
    """
    return _last_emotion, _last_conf, _last_time
