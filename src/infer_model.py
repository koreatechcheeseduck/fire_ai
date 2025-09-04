# src/infer_model.py
from __future__ import annotations
import json, glob
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# === 학습 시 정의했던 입력 피처 명 ===
INPUT_FEATURES: List[str] = [
    "building_agreement_count",
    "building_structure",
    "building_usage_status",
    "total_floor_area",
    "soot_area",
    "multi_use_flag",
    "fuel_type",
    "fire_management_target_flag",
    "unit_temperature",
    "unit_humidity",
    "unit_wind_speed",
    "facility_location",
    "forest_fire_flag",
    "total_floor_count",
    "vehicle_fire_flag",
    "ignition_material",
    "special_fire_object_name",
    "wind_direction",
]

# 학습 시 수치형/플래그/문자열 컬럼을 동일하게 맞춰주기 위한 정의
NUMERIC_COLS = {
    "building_agreement_count",
    "total_floor_area",
    "soot_area",
    "unit_temperature",
    "unit_humidity",
    "unit_wind_speed",
    "total_floor_count",
}
FLAG_COLS = {
    "multi_use_flag",
    "fire_management_target_flag",
    "forest_fire_flag",
    "vehicle_fire_flag",
}
TEXT_COLS = set(INPUT_FEATURES) - NUMERIC_COLS - FLAG_COLS

# === 타깃 헤드(분류/회귀) ===
CLASS_TARGETS = [
    "ignition_device",
    "ignition_heat_source",
    "ignition_cause",
    "fire_station_name",
    "combustion_expansion_material",
    "fire_type",
]
REG_TARGETS = [
    "casualty_count",
    "property_damage_amount",
]

def _pick_one(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def _safe_joblib_load(path: Optional[Path]) -> Any:
    return joblib.load(path) if path and path.exists() else None

# ---------------------------
# 유사사례 인덱스 (TF-IDF + NN)
# ---------------------------
class SimilarCaseIndex:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.vectorizer = None
        self.nn = None
        self.meta_df = None
        if self.index_dir.exists():
            self.vectorizer = _safe_joblib_load(self.index_dir / "tfidf_vectorizer.joblib")
            self.nn = _safe_joblib_load(self.index_dir / "nn_index.joblib")
            meta_path = self.index_dir / "meta.parquet"
            if meta_path.exists():
                self.meta_df = pd.read_parquet(meta_path)

    def is_ready(self) -> bool:
        return self.vectorizer is not None and self.nn is not None and self.meta_df is not None

    def query(self, text: str, topk: int = 10) -> List[Dict[str, Any]]:
        if not self.is_ready():
            return []
        X = self.vectorizer.transform([text])
        dist, idx = self.nn.kneighbors(X, n_neighbors=min(topk, len(self.meta_df)))
        idx = idx[0]
        dist = dist[0]
        rows = self.meta_df.iloc[idx].copy()
        rows["similarity"] = (1.0 / (1.0 + dist))
        # dict로 반환 (NaN을 JSON에 안전하게)
        return json.loads(rows.to_json(orient="records", force_ascii=False))

# ---------------------------
# 메인 모델 래퍼
# ---------------------------
class EnsembleRF:
    """
    - model_dir: 각 타깃별 파이프라인(.joblib)과 레이블(.joblib) 저장 위치
    - index_dir: scripts/build_index.py 산출물 위치 (선택)
    """
    def __init__(self, model_dir: str = "models/rf_v1", index_dir: Optional[str] = "models/index_v1"):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"model_dir not found: {self.model_dir.resolve()}")

        # 유사사례 인덱스(선택)
        self.index = SimilarCaseIndex(Path(index_dir)) if index_dir else None

        # 분류/회귀 모델 로드
        self.cls_models: Dict[str, Any] = {}
        self.reg_models: Dict[str, Any] = {}
        self.cls_labels: Dict[str, List[str]] = {}

        # 파일 네이밍 예시:
        #  - ignition_device_rf.joblib, ignition_device_labels.joblib
        #  - casualty_count_rf.joblib  (회귀는 labels 없음)
        for tgt in CLASS_TARGETS:
            model_path = _pick_one([Path(p) for p in glob.glob(str(self.model_dir / f"*{tgt}*rf.joblib"))])
            labels_path = _pick_one([Path(p) for p in glob.glob(str(self.model_dir / f"*{tgt}*labels.joblib"))])
            if model_path:
                self.cls_models[tgt] = joblib.load(model_path)
            if labels_path:
                labels_obj = joblib.load(labels_path)
                # labels_obj가 list 또는 numpy array라고 가정
                if isinstance(labels_obj, (list, tuple, np.ndarray)):
                    self.cls_labels[tgt] = [str(x) for x in list(labels_obj)]
                else:
                    # dict로 저장된 경우 classes_ 키를 우선 사용
                    self.cls_labels[tgt] = [str(x) for x in labels_obj.get("classes_", [])]

        for tgt in REG_TARGETS:
            model_path = _pick_one([Path(p) for p in glob.glob(str(self.model_dir / f"*{tgt}*rf.joblib"))])
            if model_path:
                self.reg_models[tgt] = joblib.load(model_path)

        # (선택) 공통 메타
        self.meta = {}
        meta_path = self.model_dir / "meta.json"
        if meta_path.exists():
            try:
                self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                self.meta = {}

    # ---------- 입력 정규화 ----------
    @staticmethod
    def _yn_to01(v: Any) -> Optional[int]:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        s = str(v).strip().upper()
        if s in {"Y", "YES", "1", "TRUE"}:
            return 1
        if s in {"N", "NO", "0", "FALSE"}:
            return 0
        # 숫자처럼 보이면 숫자화
        try:
            f = float(s)
            return int(f)
        except Exception:
            return None

    @staticmethod
    def _clean_text(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s != "" else None

    @staticmethod
    def _to_float(v: Any) -> Optional[float]:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except Exception:
            return None

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        norm: Dict[str, Any] = {}
        for k in INPUT_FEATURES:
            v = payload.get(k, None)

            if k in FLAG_COLS:
                norm[k] = self._yn_to01(v)
            elif k in NUMERIC_COLS:
                norm[k] = self._to_float(v)
            else:  # TEXT_COLS
                norm[k] = self._clean_text(v)

        # 빈 문자열을 NaN으로 치환
        for k, v in list(norm.items()):
            if isinstance(v, str) and v == "":
                norm[k] = None
        return norm

    def _to_dataframe(self, payload: Dict[str, Any]) -> pd.DataFrame:
        data = self._normalize_payload(payload)
        # DataFrame으로 만들고, pandas가 None -> NaN 으로 처리하도록
        df = pd.DataFrame([data], columns=INPUT_FEATURES)
        return df

    # 유사사례용 텍스트 빌드 (build_index.py에서 사용한 텍스트 구성과 유사하게)
    def _build_case_text(self, payload: Dict[str, Any]) -> str:
        p = self._normalize_payload(payload)
        # 주로 텍스트/기호성 컬럼을 이어 붙임
        keys = [
            "building_structure",
            "building_usage_status",
            "total_floor_area",
            "soot_area",
            "multi_use_flag",
            "fuel_type",
            "fire_management_target_flag",
            "unit_temperature",
            "unit_humidity",
            "unit_wind_speed",
            "facility_location",
            "forest_fire_flag",
            "total_floor_count",
            "vehicle_fire_flag",
            "ignition_material",
            "special_fire_object_name",
            "wind_direction",
        ]
        parts = []
        for k in keys:
            v = p.get(k, None)
            if v is None:
                continue
            parts.append(str(v))
        return " ".join(parts)

    # ---------- 예측 ----------
    def predict(self, payload: Dict[str, Any], return_similar: bool = True, topk: int = 10) -> Dict[str, Any]:
        X = self._to_dataframe(payload)

        out: Dict[str, Any] = {
            "inputs": payload,
            "predictions": {},
            "similar_cases": [],
        }

        # 분류
        for tgt, model in self.cls_models.items():
            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X)[0]
                    idx = int(np.argmax(prob))
                    conf = float(np.max(prob))
                else:
                    idx = int(model.predict(X)[0])
                    conf = None

                label = None
                labels = self.cls_labels.get(tgt)
                if labels and 0 <= idx < len(labels):
                    label = labels[idx]

                out["predictions"][tgt] = {
                    "index": idx,
                    "label": label,
                    "confidence": conf,
                }
            except Exception as e:
                out["predictions"][tgt] = {"error": f"{type(e).__name__}: {e}"}

        # 회귀
        for tgt, model in self.reg_models.items():
            try:
                y = float(model.predict(X)[0])
                rule = self.meta.get("regressors", {}).get(tgt, {})
                if rule.get("exp", False):
                    y = float(np.expm1(y))
                out["predictions"][tgt] = {"value": y}
            except Exception as e:
                out["predictions"][tgt] = {"error": f"{type(e).__name__}: {e}"}

        # 유사사례
        if return_similar and self.index and self.index.is_ready():
            text = self._build_case_text(payload)
            out["similar_cases"] = self.index.query(text, topk=topk)

        return out
