import pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TEXT_COLS = ['fire_type_nm','bldg_frame_nm','bldg_structure_nm','bldg_inside_nm',
             'ignition_nm','ignition_structure_nm','first_ignition_nm',
             'combusion_enlarger_nm','facility_info_nm']

def build_query_text(row: dict, predicted: dict) -> str:
    parts = [row.get('fire_type_nm'), row.get('bldg_frame_nm'), row.get('bldg_structure_nm'),
             row.get('bldg_inside_nm'), row.get('power_source_nm'),
             predicted.get('cause'), predicted.get('ignition')]
    return " ".join([str(p) for p in parts if p])

def load_index(indir: str):
    vec = joblib.load(f"{indir}/tfidf_vectorizer.joblib")
    X = joblib.load(f"{indir}/tfidf_matrix.joblib")
    meta = pd.read_parquet(f"{indir}/meta.parquet")
    return vec, X, meta

def query(vec, X, meta, qtext: str, topk=5):
    q = vec.transform([qtext])
    sims = cosine_similarity(q, X)[0]
    idx = sims.argsort()[::-1][:topk]
    out=[]
    for i in idx:
        r = meta.iloc[int(i)]
        out.append(dict(
            occur_datetime=str(r.get('occur_datetime')) if 'occur_datetime' in meta.columns else None,
            fire_type_nm=r.get('fire_type_nm') if 'fire_type_nm' in meta.columns else None,
            ignition_nm=r.get('ignition_nm') if 'ignition_nm' in meta.columns else None,
            ignition_structure_nm=r.get('ignition_structure_nm') if 'ignition_structure_nm' in meta.columns else None,
            similarity=float(sims[int(i)])
        ))
    return out
