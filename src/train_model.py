# src/train_model.py
import argparse
import os
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error

# === 입력 피처 목록 (infer_model.py와 동일하게 유지) ===
INPUT_FEATURES = [
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

# === 타깃 목록 ===
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

# === 전처리 ===
def build_preprocessor(X: pd.DataFrame):
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,  # 희소행렬 유지(원-핫 속도 ↑)
    )

def _rf_params(is_fast: bool, is_cls: bool):
    """디버그 모드일 때 가벼운 하이퍼파라미터를 사용."""
    if is_fast:
        params = dict(
            n_estimators=60,
            max_depth=12,
            n_jobs=min(8, os.cpu_count() or 1),
            random_state=42,
            verbose=1,  # ✅ 트리 빌드 진행상황 출력
        )
    else:
        params = dict(
            n_estimators=200 if is_cls else 200,
            max_depth=None,
            n_jobs=min(8, os.cpu_count() or 1),
            random_state=42,
            verbose=0,
        )
    return params

# === 학습 함수 ===
def train_classifier_head(X, y, target, outdir: Path, preprocess, fast: bool):
    print(f"[Train/CLS] {target}")
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(**_rf_params(fast, is_cls=True)))
    ])

    n_splits = 2 if fast else 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    f1s, accs = [], []
    total_folds = skf.get_n_splits()
    for i, (tr, va) in enumerate(skf.split(X, y_enc), start=1):
        pipe.fit(X.iloc[tr], y_enc[tr])
        pred = pipe.predict(X.iloc[va])
        f1 = f1_score(y_enc[va], pred, average="macro")
        acc = accuracy_score(y_enc[va], pred)
        f1s.append(f1); accs.append(acc)

        # ✅ 진행률 출력 (Fold 기준)
        percent = int(i / total_folds * 100)
        print(f"[{target}] Fold {i}/{total_folds} 완료 ({percent}%) | F1={f1:.4f}, Acc={acc:.4f}")

    # 모델 & 라벨 저장
    joblib.dump(pipe, outdir / f"{target}_rf.joblib")
    joblib.dump(le, outdir / f"{target}_labels.joblib")

    return {"f1": float(np.mean(f1s)), "acc": float(np.mean(accs))}

def train_regressor_head(X, y, target, outdir: Path, preprocess, fast: bool):
    print(f"[Train/REG] {target}")
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("reg", RandomForestRegressor(**_rf_params(fast, is_cls=False)))
    ])

    n_splits = 2 if fast else 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses = []
    total_folds = kf.get_n_splits()
    for i, (tr, va) in enumerate(kf.split(X), start=1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[va])
        rmse = mean_squared_error(y.iloc[va], pred, squared=False)
        rmses.append(rmse)

        # ✅ 진행률 출력 (Fold 기준)
        percent = int(i / total_folds * 100)
        print(f"[{target}] Fold {i}/{total_folds} 완료 ({percent}%) | RMSE={rmse:.4f}")

    joblib.dump(pipe, outdir / f"{target}_rf.joblib")
    return {"rmse": float(np.mean(rmses))}

# === 메인 ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="학습 데이터(csv/parquet)")
    ap.add_argument("--outdir", required=True, help="모델 저장 경로")
    ap.add_argument("--fast", action="store_true", help="빠른 디버그 모드(샘플링/작은 모델/2-Fold/트리 로그)")
    ap.add_argument("--sample_rows", type=int, default=None, help="임의 샘플 행 수(지정 시 우선)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    df = pd.read_csv(args.input) if args.input.endswith(".csv") else pd.read_parquet(args.input)

    # 디버그용 샘플링
    if args.sample_rows is not None and args.sample_rows > 0:
        df = df.sample(n=min(args.sample_rows, len(df)), random_state=42)
        print(f"[INFO] 샘플링 적용: {len(df):,} rows")
    elif args.fast:
        n = min(8000, len(df))
        df = df.sample(n=n, random_state=42)
        print(f"[INFO] FAST 모드 샘플링 적용: {n:,} rows")

    # 입력/타깃 분리
    X = df[INPUT_FEATURES].copy()
    preprocess = build_preprocessor(X)

    metrics = {"classifiers": {}, "regressors": {}}

    # 분류 타깃 학습
    for tgt in CLASS_TARGETS:
        if tgt not in df.columns:
            continue
        y = df[tgt].dropna()
        if y.nunique() <= 1:
            print(f"[WARN] {tgt}: 유효 클래스가 1개 이하 → 건너뜀")
            continue
        X_valid = X.loc[y.index]
        metrics["classifiers"][tgt] = train_classifier_head(X_valid, y, tgt, outdir, preprocess, args.fast)

    # 회귀 타깃 학습
    for tgt in REG_TARGETS:
        if tgt not in df.columns:
            continue
        y = df[tgt].dropna()
        if len(y) < 10:
            print(f"[WARN] {tgt}: 표본 수 부족 → 건너뜀")
            continue
        X_valid = X.loc[y.index]
        metrics["regressors"][tgt] = train_regressor_head(X_valid, y, tgt, outdir, preprocess, args.fast)

    # 지표 저장
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] 모델 저장 완료 → {outdir}")

if __name__ == "__main__":
    main()
