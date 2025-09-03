from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
from src.infer_model import EnsembleRF
from src.similarity import load_index, build_query_text, query as query_sim

app = FastAPI(title="Fire Prediction API")

class Incident(BaseModel):
    fire_type_nm: Optional[str] = None
    bldg_frame_nm: Optional[str] = None
    bldg_structure_nm: Optional[str] = None
    bldg_inside_nm: Optional[str] = None
    power_source_nm: Optional[str] = None
    ground_high: Optional[float] = None
    ground_low: Optional[float] = None
    ignition_start_floor: Optional[float] = None
    hr_unit_temp: Optional[float] = None
    dispatch_time: Optional[float] = None
    prpt_dam_amount: Optional[float] = None

class SimilarCase(BaseModel):
    occur_datetime: Optional[str] = None
    fire_type_nm: Optional[str] = None
    ignition_nm: Optional[str] = None
    ignition_structure_nm: Optional[str] = None
    similarity: float

class PredictResponse(BaseModel):
    cause: str
    cause_confidence: float
    ignition: str
    ignition_confidence: float
    risk_score: float
    risk_level: int
    similar: List[SimilarCase] = []

MODEL_DIR = "models/rf_v1"
INDEX_DIR = "models/index_v1"

try:
    ensemble = EnsembleRF(MODEL_DIR)
except Exception as e:
    ensemble = None
    print("[WARN] model not loaded:", e)

try:
    tfidf_vec, tfidf_X, tfidf_meta = load_index(INDEX_DIR)
except Exception as e:
    tfidf_vec = tfidf_X = tfidf_meta = None
    print("[WARN] index not loaded:", e)

def risk_and_level(c_conf: float, i_conf: float, dispatch_time: float|None):
    # simple aggregation of confidences; adjust later as needed
    risk = 0.6*c_conf + 0.4*i_conf
    if dispatch_time and dispatch_time>10:
        risk = min(1.0, risk+0.1)
    level = 1 + int(risk*4.0)  # 1~5
    return float(risk), int(level)

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(inc: Incident):
    # fallback if model missing
    if ensemble is None:
        return PredictResponse(
            cause="정보부족", cause_confidence=0.0,
            ignition="정보부족", ignition_confidence=0.0,
            risk_score=0.0, risk_level=1, similar=[]
        )

    row = pd.DataFrame([inc.model_dump()])
    pred = ensemble.predict(row)[0]
    risk, level = risk_and_level(pred["cause_confidence"], pred["ignition_confidence"], inc.dispatch_time)

    sim_list = []
    if tfidf_vec is not None:
        qtext = build_query_text(inc.model_dump(), pred)
        sim_raw = query_sim(tfidf_vec, tfidf_X, tfidf_meta, qtext, topk=5)
        for r in sim_raw:
            sim_list.append(SimilarCase(**r))

    return PredictResponse(
        cause=pred["cause"], cause_confidence=pred["cause_confidence"],
        ignition=pred["ignition"], ignition_confidence=pred["ignition_confidence"],
        risk_score=risk, risk_level=level, similar=sim_list
    )
