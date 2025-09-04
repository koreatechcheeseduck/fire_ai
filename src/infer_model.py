# src/infer_model.py
from __future__ import annotations
import json, glob
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# ===== 입력 피처 =====
INPUT_FEATURES: List[str] = [
    "building_agreement_count","building_structure","building_usage_status",
    "total_floor_area","soot_area","multi_use_flag","fuel_type",
    "fire_management_target_flag","unit_temperature","unit_humidity","unit_wind_speed",
    "facility_location","forest_fire_flag","total_floor_count","vehicle_fire_flag",
    "ignition_material","special_fire_object_name","wind_direction",
]

# ===== 타깃 =====
CLASS_TARGETS = [
    "ignition_device","ignition_heat_source","ignition_cause",
    "fire_station_name","combustion_expansion_material","fire_type",
]
REG_TARGETS = ["casualty_count","property_damage_amount"]

def _pick_one(paths: List[Path]) -> Optional[Path]:
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0] if paths else None

class SimilarCaseIndex:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.nn = None
        self.feature_df = None
        self.meta = {}
        if self.index_dir.exists():
            nn_path = self.index_dir / "nn.joblib"
            feat_path = self.index_dir / "features.parquet"
            meta_path = self.index_dir / "meta.json"
            if nn_path.exists() and feat_path.exists():
                self.nn = joblib.load(nn_path)
                self.feature_df = pd.read_parquet(feat_path)
            if meta_path.exists():
                try:
                    self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    self.meta = {}

    def is_ready(self) -> bool:
        return self.nn is not None and self.feature_df is not None

    def query(self, x_vec: np.ndarray, topk: int = 10) -> List[Dict[str, Any]]:
        if not self.is_ready(): return []
        dist, idx = self.nn.kneighbors(x_vec, n_neighbors=min(topk, len(self.feature_df)))
        idx, dist = idx[0], dist[0]
        rows = self.feature_df.iloc[idx].copy()
        rows["similarity"] = (1.0 / (1.0 + dist))
        return rows.to_dict(orient="records")

class LocalEnsembleRF:
    def __init__(self, model_dir: str = "models/rf_v1", index_dir: Optional[str] = "models/index_v1"):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"model_dir not found: {self.model_dir.resolve()}")

        # 모델
        self.cls_models: Dict[str, Any] = {}
        self.reg_models: Dict[str, Any] = {}

        # 분류 모델 로딩(라벨인코더 파일과 이름이 겹치지 않도록 rf_ 접두 위주)
        for tgt in CLASS_TARGETS:
            path = _pick_one([Path(p) for p in glob.glob(str(self.model_dir / f"rf_{tgt}*.joblib"))])
            if path:
                self.cls_models[tgt] = joblib.load(path)

        # 회귀 모델
        for tgt in REG_TARGETS:
            path = _pick_one([Path(p) for p in glob.glob(str(self.model_dir / f"rf_{tgt}*.joblib"))])
            if path:
                self.reg_models[tgt] = joblib.load(path)

        # 라벨 인코더(있으면 사용)
        self.label_encoders: Dict[str, Any] = {}
        enc_dir = self.model_dir / "encoders"
        if enc_dir.exists():
            for tgt in CLASS_TARGETS:
                cand = [
                    enc_dir / f"{tgt}_le.joblib",
                    enc_dir / f"{tgt}.joblib",
                ]
                p = next((c for c in cand if c.exists()), None)
                if p:
                    self.label_encoders[tgt] = joblib.load(p)

        # 메타(백업 클래스명/스케일링 규칙)
        self.meta = {}
        meta_path = self.model_dir / "meta.json"
        if meta_path.exists():
            try:
                self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                self.meta = {}

        # 유사사례
        self.index = SimilarCaseIndex(Path(index_dir)) if index_dir else None

    # ---------- 내부 유틸 ----------
    def _to_df(self, payload: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame([{k: payload.get(k, np.nan) for k in INPUT_FEATURES}], columns=INPUT_FEATURES)

    def _inverse_label(self, target: str, idx: int) -> Optional[str]:
        # 1) 저장된 LabelEncoder 최우선
        le = self.label_encoders.get(target)
        if le is not None:
            try:
                return str(le.inverse_transform([idx])[0])
            except Exception:
                pass
        # 2) meta.json에 클래스 리스트가 있으면 사용
        classes = (self.meta.get("class_targets", {}).get(target, {}) or {}).get("classes")
        if classes and 0 <= idx < len(classes):
            return str(classes[idx])
        return None  # 복원 실패

    # ---------- 공개 API ----------
    def predict(self, payload: Dict[str, Any], return_similar: bool = True, topk: int = 10) -> Dict[str, Any]:
        X = self._to_df(payload)
        out: Dict[str, Any] = {"inputs": payload, "predictions": {}, "similar_cases": []}

        # 분류
        for tgt, model in self.cls_models.items():
            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X)[0]
                    pred_idx = int(np.argmax(prob))
                    conf = float(np.max(prob))
                    label = self._inverse_label(tgt, pred_idx)
                    # (선택) 상위 Top-5
                    topn = min(5, len(prob))
                    top_idx = np.argsort(prob)[::-1][:topn]
                    top = [
                        {"index": int(i), "label": self._inverse_label(tgt, int(i)), "prob": float(prob[i])}
                        for i in top_idx
                    ]
                    out["predictions"][tgt] = {"index": pred_idx, "label": label, "confidence": conf, "top": top}
                else:
                    pred_idx = int(model.predict(X)[0])
                    label = self._inverse_label(tgt, pred_idx)
                    out["predictions"][tgt] = {"index": pred_idx, "label": label}
            except Exception as e:
                out["predictions"][tgt] = {"error": f"{type(e).__name__}: {e}"}

        # 회귀
        for tgt, model in self.reg_models.items():
            try:
                y = float(model.predict(X)[0])
                rule = (self.meta.get("regressors", {}).get(tgt, {}) or {})
                if rule.get("exp", False):
                    y = float(np.expm1(y))
                out["predictions"][tgt] = {"value": y}
            except Exception as e:
                out["predictions"][tgt] = {"error": f"{type(e).__name__}: {e}"}

        # 유사사례
        if return_similar and self.index and self.index.is_ready():
            # 임의의 분류 파이프라인 전처리를 재사용(있는 경우)
            any_model = next(iter(self.cls_models.values()), None) or next(iter(self.reg_models.values()), None)
            vec = None
            if any_model is not None:
                try:
                    if hasattr(any_model, "named_steps") and "preprocess" in any_model.named_steps:
                        Xt = any_model.named_steps["preprocess"].transform(X)
                    else:
                        Xt = any_model.transform(X) if hasattr(any_model, "transform") else None
                    if Xt is not None:
                        vec = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
                except Exception:
                    vec = None
            if vec is not None:
                out["similar_cases"] = self.index.query(vec, topk=topk)

        return out

# --- 외부에서 쓰던 클래스명과 호환 ---
EnsembleRF = LocalEnsembleRF
