# src/infer_model.py
from __future__ import annotations
import json, glob
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd

# ===== 입력 피처 =====
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

# ===== 타깃 목록 =====
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
    # 필요 시 여기에 지연 분(target)을 추가
    # "arrival_delay_min",
    # "initial_extinguish_delay_min",
]


def _pick_latest(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]


class SimilarCaseIndex:
    """scripts/build_index.py 산출물 사용 (있는 경우에만 동작)"""
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir) if index_dir else None
        self.nn = None
        self.feature_df = None
        self.meta: Dict[str, Any] = {}
        if self.index_dir and self.index_dir.exists():
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
        if not self.is_ready():
            return []
        dist, idx = self.nn.kneighbors(x_vec, n_neighbors=min(topk, len(self.feature_df)))
        idx, dist = idx[0], dist[0]
        rows = self.feature_df.iloc[idx].copy()
        rows["similarity"] = 1.0 / (1.0 + dist)
        return rows.to_dict(orient="records")


class LocalEnsembleRF:
    """
    - model_dir 밑의 각 타깃별 파이프라인(.joblib) 자동 로드
    - LabelEncoder 류 artifact는 따로 분리해 저장 (모델로 착각하지 않음)
    - index_dir 이 있으면 유사사례 검색 제공
    """
    def __init__(self, model_dir: str = "models/rf_v1", index_dir: Optional[str] = "models/index_v1"):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"model_dir not found: {self.model_dir.resolve()}")

        # (선택) 유사사례 인덱스
        self.index = SimilarCaseIndex(Path(index_dir)) if index_dir else None

        # (선택) 공용 전처리
        self.preprocess = None
        for name in ("preprocess.joblib", "preprocessor.joblib"):
            pp = self.model_dir / name
            if pp.exists():
                self.preprocess = joblib.load(pp)
                break

        # (선택) 메타
        self.meta: Dict[str, Any] = {}
        meta_path = self.model_dir / "meta.json"
        if meta_path.exists():
            try:
                self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                self.meta = {}

        # (선택) 라벨 인코더(여러 타깃을 묶은 dict 형태라고 가정)
        self.encoders: Dict[str, Any] = {}
        for name in ("label_encoders.joblib", "label_encoder.joblib", "encoders.joblib"):
            p = self.model_dir / name
            if p.exists():
                try:
                    obj = joblib.load(p)
                    if isinstance(obj, dict):
                        self.encoders.update(obj)
                except Exception:
                    pass

        # 실제 모델 컨테이너
        self.cls_models: Dict[str, Any] = {}
        self.reg_models: Dict[str, Any] = {}

        # 타깃별 파일 탐색 → 최신 파일 선택 → 로드 → predict 없는 객체는 인코더로 분류
        def _load_best(pattern: str) -> Optional[Any]:
            cands = [Path(p) for p in glob.glob(str(self.model_dir / pattern))]
            cands = [p for p in cands if "encoder" not in p.name.lower()]  # 파일명으로 1차 필터
            path = _pick_latest(cands)
            if not path:
                return None
            obj = joblib.load(path)
            # 예: LabelEncoder, dict 등은 모델 아님
            if hasattr(obj, "predict") or hasattr(obj, "predict_proba") or hasattr(obj, "transform"):
                return obj
            # 그래도 인코더일 수 있으니 기록만 해두고 None 반환
            if hasattr(obj, "classes_"):
                # 파일명이 rf_* 가 아닌 경우(=인코더 파일)일 확률 높음 → 버킷에 넣지 않음
                return None
            return None

        # 분류 모델
        for tgt in CLASS_TARGETS:
            # 저장 파일명이 다양할 수 있어 패턴을 여럿 시도
            model = (_load_best(f"*{tgt}*fold*.joblib")
                     or _load_best(f"*{tgt}*.joblib")
                     or _load_best(f"rf_cls_{tgt}*.joblib"))
            if model is not None:
                self.cls_models[tgt] = model

        # 회귀 모델
        for tgt in REG_TARGETS:
            model = (_load_best(f"*{tgt}*fold*.joblib")
                     or _load_best(f"*{tgt}*.joblib")
                     or _load_best(f"rf_reg_{tgt}*.joblib"))
            if model is not None:
                self.reg_models[tgt] = model

    # -------- 내부 유틸 --------
    def _to_dataframe(self, payload: Dict[str, Any]) -> pd.DataFrame:
        data = {k: payload.get(k, np.nan) for k in INPUT_FEATURES}
        return pd.DataFrame([data], columns=INPUT_FEATURES)

    def _vector_for_index(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        # 전처리 벡터 하나 생성 (아무 분류/회귀 모델의 전처리 단을 활용)
        model_any = (next(iter(self.cls_models.values()), None)
                     or next(iter(self.reg_models.values()), None))
        if model_any is None:
            return None
        try:
            if hasattr(model_any, "named_steps") and "preprocess" in getattr(model_any, "named_steps", {}):
                Xt = model_any.named_steps["preprocess"].transform(X)
            elif hasattr(model_any, "transform"):
                Xt = model_any.transform(X)
            else:
                return None
            if hasattr(Xt, "toarray"):
                Xt = Xt.toarray()
            return np.asarray(Xt, dtype=float)
        except Exception:
            return None

    # -------- 공개 API --------
    def predict(self, payload: Dict[str, Any], return_similar: bool = True, topk: int = 10) -> Dict[str, Any]:
        X = self._to_dataframe(payload)
        results: Dict[str, Any] = {"inputs": payload, "predictions": {}, "similar_cases": []}

        # 분류 타깃
        for tgt in CLASS_TARGETS:
            model = self.cls_models.get(tgt)
            if model is None:
                results["predictions"][tgt] = {"error": "No classifier model file loaded for this target."}
                continue

            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X)[0]
                    pred_idx = int(np.argmax(prob))
                    conf = float(np.max(prob))
                else:
                    pred_idx = int(model.predict(X)[0])
                    conf = None

                # 라벨 역변환(가능하면)
                label = None
                enc = self.encoders.get(tgt)
                if enc is not None and hasattr(enc, "classes_"):
                    try:
                        label = enc.classes_[pred_idx].item() if hasattr(enc.classes_[pred_idx], "item") else enc.classes_[pred_idx]
                    except Exception:
                        label = None

                results["predictions"][tgt] = {
                    "index": pred_idx,
                    "label": label,
                    "confidence": conf,
                }
            except Exception as e:
                results["predictions"][tgt] = {"error": f"{type(e).__name__}: {e}"}

        # 회귀 타깃
        for tgt in REG_TARGETS:
            model = self.reg_models.get(tgt)
            if model is None:
                results["predictions"][tgt] = {"error": "No regressor model file loaded for this target."}
                continue
            try:
                y = float(model.predict(X)[0])
                rule = self.meta.get("regressors", {}).get(tgt, {})
                if rule.get("exp", False):
                    y = float(np.expm1(y))
                results["predictions"][tgt] = {"value": y}
            except Exception as e:
                results["predictions"][tgt] = {"error": f"{type(e).__name__}: {e}"}

        # 유사사례
        if return_similar and self.index and self.index.is_ready():
            vec = self._vector_for_index(X)
            if vec is not None:
                results["similar_cases"] = self.index.query(vec, topk=topk)

        return results


# --- 외부에서 기대하는 이름과 맞추기 ---
# API/스크립트가 from src.infer_model import EnsembleRF 를 사용해도 동작하도록 alias
EnsembleRF = LocalEnsembleRF
