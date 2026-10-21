import streamlit as st
import joblib
import librosa
import numpy as np
from scipy.signal import butter, lfilter
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Voice Gender Recognition AI", page_icon="🎙️", layout="centered")

# 1. Load trained model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("svm_gender_classifier.pkl")
    scaler = joblib.load("voice_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    st.error(f"Error loading models/scalers: {e}")
    artifacts_loaded = False

# Butterworth High-Pass Filter to eliminate low-frequency environment noise (< 80 Hz)
def highpass_filter(data, cutoff=80.0, sr=22050, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y

# 2. Extract identical 66 features with highly-optimized Yin Pitch Tracking for live recordings
def extract_rich_features(file_path):
    # Load audio signal
    y, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')

    # Apply highpass filter to eliminate live environment noise / mic hums
    y = highpass_filter(y, cutoff=80.0, sr=sr)

    # MFCCs & Deltas
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs.T, axis=0)
    mfcc_delta_mean = np.mean(librosa.feature.delta(mfccs).T, axis=0)
    mfcc_delta2_mean = np.mean(librosa.feature.delta(mfccs, order=2).T, axis=0)

    # Chroma
    chroma_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

    # Spectral
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spec_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    spec_contrast_mean = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

    # Temporal
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))

    # Highly robust Pitch Tracking (YIN algorithm restricted to vocal ranges)
    try:
        # Voice Pitch (F0) tracking optimized for live performance (80Hz to 400Hz bound)
        f0 = librosa.yin(y=y, sr=sr, fmin=80, fmax=400)
        # Mask non-voiced or noisy pitch frames
        voiced_pitches = f0[~np.isnan(f0)]
        voiced_pitches = voiced_pitches[voiced_pitches > 80]
        
        if len(voiced_pitches) > 0:
            mean_pitch = np.mean(voiced_pitches)
            std_pitch = np.std(voiced_pitches)
        else:
            # Fallback to piptrack if YIN yields no valid frames
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[pitches > 80]
            mean_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 180.0
            std_pitch = np.std(pitch_values) if len(pitch_values) > 0 else 50.0
    except Exception:
        mean_pitch = 180.0
        std_pitch = 50.0

    # Assemble features in identical dictionary structure
    features = {
        "mean_pitch": mean_pitch,
        "std_pitch": std_pitch,
        "spectral_centroid": spec_centroid,
        "spectral_bandwidth": spec_bandwidth,
        "spectral_rolloff": spec_rolloff,
        "spectral_flatness": spec_flatness,
        "zero_crossing_rate": zcr,
        "rms_energy": rms
    }

    for i in range(13):
        features[f"mfcc_{i+1}"] = mfcc_mean[i]
        features[f"mfcc_delta_{i+1}"] = mfcc_delta_mean[i]
        features[f"mfcc_delta2_{i+1}"] = mfcc_delta2_mean[i]

    for i in range(12):
        features[f"chroma_{i+1}"] = chroma_mean[i]

    for i in range(len(spec_contrast_mean)):
        features[f"spec_contrast_{i+1}"] = spec_contrast_mean[i]

    return np.array(list(features.values())).reshape(1, -1)

# 3. Streamlit Interface
st.title("🎙️ Voice Gender Recognition AI")
st.write("Upload a standard `.wav` audio file below to instantly classify the speaker's voice.")

if artifacts_loaded:
    audio_file = st.file_uploader("Choose a WAV file", type=["wav"])

    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')

        if st.button("Run Classifier", type="primary"):
            with st.spinner("Processing signal, filtering noise, and extracting 66 acoustic features..."):
                temp_filename = "temp_app_audio.wav"
                with open(temp_filename, "wb") as f:
                    f.write(audio_file.getbuffer())

                try:
                    feats = extract_rich_features(temp_filename)
                    feats_scaled = scaler.transform(feats)
                    pred = model.predict(feats_scaled)[0]
                    label = "Male ♂️" if pred == 1 else "Female ♀️"

                    st.success("Classification Finished!")
                    st.metric(label="Predicted Gender", value=label)
                except Exception as ex:
                    st.error(f"Error processing file: {ex}")
