import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import pickle

MODEL_PATH = 'model3.keras'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    return model, le

model, le = load_artifacts()

def extract_mfcc_sequence(file_path, n_mfcc=40, max_len=200):
    y, sr = librosa.load(file_path, res_type='scipy')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T

def predict_emotion(file_path):
    mfcc_seq = extract_mfcc_sequence(file_path)
    mfcc_seq = np.expand_dims(mfcc_seq, axis=0)
    pred = model.predict(mfcc_seq)
    predicted_class = np.argmax(pred)
    return le.classes_[predicted_class]

st.title("Speech Emotion Recognition Web App")
st.write("Upload a WAV or MP3 audio file to classify its emotion.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    st.audio("temp.wav")
    emotion = predict_emotion("temp.wav")
    st.success(f"Predicted Emotion: {emotion.capitalize()}")
