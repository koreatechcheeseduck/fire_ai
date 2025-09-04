# src/train_model.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

# ===== 공통 입력 특성 =====
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

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # 필요한 컬럼만 유지 (없는 건 채움)
    for col in INPUT_FEATURES + CLASS_TARGETS + REG_TARGETS:
        if col not in df.columns:
            df[col] = np.nan
    return df

def split_num_cat(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = []
    cat_cols = []
    for c in INPUT_FEATURES:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols

def _ohe_kwargs() -> Dict[str, Any]:
    # sklearn 1.2+ uses sparse_output; 이전 버전은 sparse
    try:
        # noinspection PyArgumentList
        OneHotEncoder(sparse_output=False)
        return dict(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return dict(handle_unknown="ignore", sparse=False)

def build_preprocess(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    ONNX 변환 호환성 높은 전처리:
      - 수치: SimpleImputer(strategy="median")
      - 문자열: SimpleImputer(strategy="constant", fill_value="__NA__") + OneHotEncoder(handle_unknown="ignore")
    """
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categoric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="__NA__")),
        ("onehot", OneHotEncoder(**_ohe_kwargs())),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categoric, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,  # OHE가 dense여도 OK
    )
    return pre

def kfold_train_report(X: pd.DataFrame, y: pd.Series, pipe: Pipeline, tgt: str, k: int = 3):
    f1s, accs = [], []
    # 소수 클래스 경고 회피를 위해 최소 분포 확인
    y_nonan = y.fillna("__NA__")
    skf = StratifiedKFold(n_splits=min(k, max(2, y_nonan.nunique())), shuffle=True, random_state=42)
    for i, (tr, va) in enumerate(skf.split(X, y_nonan), 1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        yp = pipe.predict(X.iloc[va])
        f1s.append(f1_score(y.iloc[va], yp, average="macro"))
        accs.append(accuracy_score(y.iloc[va], yp))
        print(f"[{tgt}] Fold {i}/{skf.n_splits} 완료 ({int(i/skf.n_splits*100)}%) | F1={f1s[-1]:.4f}, Acc={accs[-1]:.4f}")
    return float(np.mean(f1s)), float(np.mean(accs))

def train_classifier_head(
    X: pd.DataFrame, y: pd.Series, target_name: str, outdir: Path, base_pre: ColumnTransformer, fast: bool = False
) -> Dict[str, Any]:
    y = y.astype(str).fillna("__NA__")
    model = RandomForestClassifier(
        n_estimators=200 if not fast else 60,
        max_depth=None if not fast else 12,
        n_jobs=-1,
        random_state=42,
    )
    pipe = Pipeline([
        ("preprocess", base_pre),
        ("clf", model),
    ])
    if fast:
        pipe.fit(X, y)
        f1, acc = 0.0, 0.0
    else:
        f1, acc = kfold_train_report(X, y, pipe, target_name, k=3)

    joblib.dump(pipe, outdir / f"{target_name}_rf.joblib")
    # 클래스 라벨도 별도 저장
    try:
        check_is_fitted(pipe)
        classes_ = pipe.named_steps["clf"].classes_.tolist()
    except Exception:
        classes_ = sorted(y.unique().tolist())
    joblib.dump(classes_, outdir / f"{target_name}_labels.joblib")

    return {"f1_macro": f1, "accuracy": acc}

def train_regressor_head(
    X: pd.DataFrame, y: pd.Series, target_name: str, outdir: Path, base_pre: ColumnTransformer, fast: bool = False
) -> Dict[str, Any]:
    y = pd.to_numeric(y, errors="coerce").fillna(0.0)
    model = RandomForestRegressor(
        n_estimators=300 if not fast else 80,
        max_depth=None if not fast else 16,
        n_jobs=-1,
        random_state=42,
    )
    pipe = Pipeline([
        ("preprocess", base_pre),
        ("reg", model),
    ])
    if fast:
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(Xtr, ytr)
        yp = pipe.predict(Xva)
        r2 = float(r2_score(yva, yp))
        mae = float(mean_absolute_error(yva, yp))
    else:
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(Xtr, ytr)
        yp = pipe.predict(Xva)
        r2 = float(r2_score(yva, yp))
        mae = float(mean_absolute_error(yva, yp))

    joblib.dump(pipe, outdir / f"{target_name}_rf.joblib")
    return {"r2": r2, "mae": mae}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fast", action="store_true", help="빠르게(경량) 학습")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.input)
    X = df[INPUT_FEATURES].copy()

    num_cols, cat_cols = split_num_cat(X)
    print(f"[INFO] num_cols({len(num_cols)}), cat_cols({len(cat_cols)})")
    base_pre = build_preprocess(num_cols, cat_cols)
    joblib.dump({"num_cols": num_cols, "cat_cols": cat_cols}, outdir / "preprocess_schema.json")

    metrics = {"classifiers": {}, "regressors": {}}

    # 분류
    for tgt in CLASS_TARGETS:
        print(f"[Train/CLS] {tgt}")
        metrics["classifiers"][tgt] = train_classifier_head(X, df[tgt], tgt, outdir, base_pre, fast=args.fast)

    # 회귀
    for tgt in REG_TARGETS:
        print(f"[Train/REG] {tgt}")
        metrics["regressors"][tgt] = train_regressor_head(X, df[tgt], tgt, outdir, base_pre, fast=args.fast)

    (outdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE] saved models to", outdir)

if __name__ == "__main__":
    main()
