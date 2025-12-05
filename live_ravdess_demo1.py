import numpy as np
import sounddevice as sd
import tensorflow as tf
import librosa

DURATION = 5 # seconds
SR=22050

model = tf.keras.models.load_model("audio1dconv.keras")

def record_audio(dur=DURATION, sr=SR):
    print(f"Recording {dur} seconds of audio")
    audio = sd.rec(int(dur*sr), 
                   samplerate=sr, 
                   channels=1, 
                   dtype='float32')
    sd.wait()
    return audio.flatten()

def detect_silence(clip,threshold):
    rms = np.sqrt(np.mean(clip**2))
    return rms<threshold

def wav_to_wav(clip):
    clip_len=DURATION*SR
    audio, sr = librosa.load(clip, sr=SR, mono=True)
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    
    # fix length
    if len(audio) < clip_len:
        audio = np.pad(audio, (0, clip_len - len(audio)))
    else:
        audio = audio[:clip_len]
    return audio.reshape((1,clip_len,1))

def wav_predict_emotion(audio):
    features=wav_to_wav(audio)
    prediction=model.predict(features)

    # prediction >0.5 is confidence in negative
    if prediction>0.5:
        pred='negative'
        conf=prediction
    # prediction <0.5 is low confidence in negative
    # <=> opposite confidence in positive
    else:
        pred='positive'
        conf=1-prediction
    return pred,conf


def main():

    print(f"Recording interval: {DURATION} seconds")
    print("\nPress Ctrl+C to stop\n")
    print("------------------")
    
    try:
        while True:
            audio = record_audio(dur=DURATION, sr=SR)
            if detect_silence(audio,0.001):
                print("No audio detected. Skipping prediction")
                print("------------------")
            else:
                emotion, confidence = wav_predict_emotion(audio)
                k=str(confidence*100)+'% confident'
                print(emotion,k)
                print("------------------")

    except KeyboardInterrupt:
        print("Stopping inference")
    return

if __name__ == "__main__":
    main()