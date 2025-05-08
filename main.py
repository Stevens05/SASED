from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib, shutil, os
import tempfile
import librosa
import python_multipart
from scipy.io import wavfile
import soundfile as sf
import pandas as pd
import numpy as np
from utils.audio_features import extract_features, detect_audio_events

app = FastAPI()
model = joblib.load("model/best_multiclass_model.pkl")
scaler = joblib.load("model/scaler.pkl")
le = joblib.load("model/label_encoder.pkl")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "world"}

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        
        df = detect_audio_events(tmp_path)
        features = df.drop(['offset', 'onset'], axis=1)
        X = pd.DataFrame(scaler.transform(features), columns=features.columns)
        pred = model.predict(X)
        labels = le.inverse_transform(pred)
        
        os.remove(tmp_path)
        return {"predictions": labels.tolist(),
                "onset" : df['onset'].tolist(),
                "offset" : df['offset'].tolist()}
    
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"file {tmp_path} is not valid"
        )
