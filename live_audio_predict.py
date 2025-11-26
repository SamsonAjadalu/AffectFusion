import numpy as np
import sounddevice as sd
import tensorflow as tf
from datetime import datetime
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import librosa

DURATION = 5 # seconds

classes = ['positive', 'negative', 'surprised']

model = tf.keras.models.load_model("audio_model1.keras")

def record_audio(duration=4, sample_rate=22050):
    print(f"Recording {duration} seconds of audio")
    audio = sd.rec(int(duration * sample_rate), 
                   samplerate=sample_rate, 
                   channels=1, 
                   dtype='float32')
    sd.wait()
    return audio.flatten()

def preprocess_audio(audio, sample_rate=22050, n=100, dpi=100):

    times = librosa.times_like(audio, sr=sample_rate)

    figure = plt.figure(figsize=(n/dpi, n/dpi), dpi=dpi)
    figure.patch.set_facecolor('white')
    plt.plot(times, audio, color='black', linewidth=0.6)

    plt.xlim(times[0], times[-1])
    plt.ylim(audio.min()*1.1, audio.max()*1.1)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', 
                pad_inches=0, dpi=dpi, facecolor='white')
    plt.close(figure)
    buf.seek(0)
    img = Image.open(buf).convert('L')
    img = img.resize((n, n), Image.LANCZOS)

    arr = np.array(img, dtype=np.int64) / 255.0

    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    
    return arr

def predict_emotion(audio, sample_rate=22050):
    features = preprocess_audio(audio, sample_rate, n=100, dpi=100)

    predictions = model.predict(features, verbose=0)
    emotion_idx = np.argmax(predictions[0])
    confidence = predictions[0][emotion_idx]
    emotion = classes[emotion_idx]
    
    return emotion, confidence, predictions[0]


def print_results(emotion, confidence, all_probs):
    
    print(f"\n{emotion.upper()} {confidence*100:.1f}%")
    print(f"{"------------------"}\n")

def main():

    print(f"Recording interval: {DURATION} seconds")
    print("\nPress Ctrl+C to stop\n")
    print("------------------")
    
    try:
        while True:

            audio = record_audio(duration=DURATION, sample_rate=22050)
            
            # Predict emotion
            emotion, confidence, all_probs = predict_emotion(audio, 22050)

            print_results(emotion, confidence, all_probs)

    except KeyboardInterrupt:
        print("Stopping inference")

if __name__ == "__main__":
    main()