import librosa
import numpy as np
from pathlib import Path
from scipy.io import wavfile
import soundfile as sf
import pandas as pd
import os

def extract_features(y, sr):

    # Duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Mel-frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000), ref=np.max)
    delta_mfcc = librosa.feature.delta(mfccs)

    # Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    
    # Spectral centroid : brightness of sound
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Bandwidth : Spread of frequencies around the centroid
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # Rolloff : Frequency below which a certain % of the total spectral energy lies
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # Flatness : Tonal vs. noisy quality (close to 1: noisy)
    flatness = librosa.feature.spectral_flatness(y=y)

    # Contrast : Difference in energy between peaks and valleys in spectrum
    contrast = librosa.feature.spectral_contrast(y=y)
    
    # RMS energy
    rms = librosa.feature.rms(y=y)

    # tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    

    return np.hstack([mfccs.mean(axis=1), chroma.mean(axis=1), zcr.std(), zcr.mean(), centroid.mean(), centroid.std(), rms.mean(), rms.std(),
                      bandwidth.mean(), bandwidth.std(), rolloff.mean(), rolloff.std(), flatness.mean(), flatness.std(), tempo,
                      contrast.mean(), contrast.std(), delta_mfcc.mean(axis=1), mel_spec.mean(), mel_spec.std(), duration])



def detect_audio_events(filepath, threshold=0.0065, sr=44100, min_event_duration = 0.2, min_silence_duration=0.5):
    """
    Détecte les événements audio basés sur l'amplitude et retourne leurs instants de début et fin.
    
    Paramètres :
        filepath (str) : chemin vers le fichier audio
        threshold (float) : seuil d'amplitude pour détecter un événement
        sr (int) : taux d'échantillonnage
        min_silence_duration (float) : durée minimale de silence entre deux événements (en secondes)

    Retour :
        pd.DataFrame : DataFrame avec colonnes ["onset", "offset"] exprimées en secondes
    """
    y, sr = librosa.load(filepath, sr=sr)
    amplitude = np.abs(y)

    # Détection des indices où l'amplitude dépasse le seuil
    event_indices = np.where(amplitude > threshold)[0]

    if len(event_indices) == 0:
        return pd.DataFrame(columns=["onset", "offset"])

    # Regroupement des indices en événements séparés par un silence suffisant
    events = []
    start_idx = event_indices[0]
    min_gap = int(min_silence_duration * sr)

    for i in range(1, len(event_indices)):
        if event_indices[i] - event_indices[i-1] > min_gap:
            end_idx = event_indices[i-1]
            if end_idx - start_idx >= min_event_duration * sr:
                events.append((start_idx, end_idx))
            start_idx = event_indices[i]

    end_idx = event_indices[-1]
    if end_idx - start_idx >= min_event_duration * sr:
        events.append((start_idx, end_idx))

    # Conversion en secondes et structuration en DataFrame
    event_times = [(round(start / sr, 2), round(end / sr, 2)) for start, end in events]
    df = pd.DataFrame(event_times, columns=["onset", "offset"])
    onsets = df['onset'].tolist()
    offsets = df['offset'].tolist()
    
    # Prepare DataFrame
    features_list = []
    for onset, offset in zip(onsets, offsets):
        start_sample = int(onset * sr)
        end_sample = int(offset * sr)
        segment = y[start_sample:end_sample]

        features = extract_features(segment, sr)
        features_list.append([onset, offset] + list(features))

    # columns names 
    feature_names = (
        [f"mfcc_{i+1}_mean" for i in range(13)] +
        [f"chroma_{i+1}_mean" for i in range(12)] +
        ["zcr_std", "zcr_mean"] +
        ["centroid_mean", "centroid_std"] +
        ["rms_mean", "rms_std"] +
        ["bandwidth_mean", "bandwidth_std"] +
        ["rolloff_mean", "rolloff_std"] +
        ["flatness_mean", "flatness_std"] +
        ["tempo"] +
        ["contrast_mean", "contrast_std"] +
        [f"delta_mfcc_{i+1}_mean" for i in range(13)] +
        ["mel_spec_mean", "mel_spec_std"] +
        ["duration"]
        )

    df = pd.DataFrame(features_list, columns=["onset", "offset"] + feature_names)
    print(f"✅ {len(df)} events detected and corresponding features are extracted...")

    return df