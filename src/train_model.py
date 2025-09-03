# src/train_model.py
# 입력 피처 고정 + 분류/회귀 멀티헤드 학습 + 메모리 절약형 파라미터
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ===== 입력/타깃 스키마 =====
INPUT_COLS = [
    "building_agreement_count","building_structure","building_usage_status",
    "total_floor_area","soot_area","multi_use_flag","fuel_type",
    "fire_management_target_flag","unit_temperature","unit_humidity","unit_wind_speed",
    "facility_location","forest_fire_flag","total_floor_count","vehicle_fire_flag",
    "ignition_material","special_fire_object_name","wind_direction"
]

CLASS_TARGETS = [
    "ignition_device","ignition_heat_source","ignition_cause",
    "fire_station_name","combustion_expansion_material","fire_type"
]
REG_TARGETS = [
    "casualty_count","property_damage_amount","arrival_delay_min","initial_extinguish_delay_min"
]

def build_preprocess(min_freq=10):
    # 수치 vs 범주 분리
    numeric_feats = [
        "building_agreement_count","total_floor_area","soot_area",
        "unit_temperature","unit_humidity","unit_wind_speed","total_floor_count"
    ]
    cat_feats = [c for c in INPUT_COLS if c not in numeric_feats]

    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_freq,
            sparse_output=True
        ))
    ])
    num = Pipeline([("imp", SimpleImputer(strategy="median"))])

    return ColumnTransformer([
        ("cat", cat, cat_feats),
        ("num", num, numeric_feats)
    ])

def make_rf_clf(seed):
    return RandomForestClassifier(
        n_estimators=150, max_depth=16, min_samples_leaf=3,
        max_features="sqrt", class_weight="balanced",
        bootstrap=True, n_jobs=2, random_state=seed
    )

def make_rf_reg(seed):
    return RandomForestRegressor(
        n_estimators=150, max_depth=16, min_samples_leaf=3,
        max_features="sqrt", bootstrap=True, n_jobs=2, random_state=seed
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="models/rf_v2")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    # 입력 피처만 유지(없으면 생성)
    for c in INPUT_COLS:
        if c not in df.columns: df[c] = np.nan
    X = df[INPUT_COLS].copy()

    # 가공본 저장(추후 인덱스/검증 용)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    save_cols = list(dict.fromkeys(INPUT_COLS + CLASS_TARGETS + REG_TARGETS))
    (df[save_cols]).to_parquet("data/processed/dataset.parquet", index=False)

    preprocess = build_preprocess(min_freq=10)

    # ===== 분류 타깃: 폴드 앙상블 =====
    for tgt in CLASS_TARGETS:
        if tgt not in df.columns or df[tgt].notna().sum() < 2:
            print(f"[SKIP] {tgt}: 라벨 부족")
            continue
        mask = df[tgt].notna()
        Xt, yt_raw = X.loc[mask], df.loc[mask, tgt].astype(str)

        le = LabelEncoder().fit(yt_raw)
        yt = pd.Series(le.transform(yt_raw))

        print(f"\n[Train/CLS] {tgt}")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        models, oof = [], np.empty(len(yt), dtype=int)
        total_folds = skf.get_n_splits()

        for i, (tr, va) in enumerate(skf.split(Xt, yt), start=1):
            pipe = Pipeline([("pre", preprocess), ("rf", make_rf_clf(42 + i))])
            pipe.fit(Xt.iloc[tr], yt.iloc[tr])
            models.append(pipe)
            pred = pipe.predict(Xt.iloc[va]);
            oof[va] = pred
            percent = (i / total_folds) * 100
            print(f"[{tgt}] Fold {i}/{total_folds} 완료 ({percent:.1f}%) | "
                  f"F1={f1_score(yt.iloc[va], pred, average='macro'):.4f}, "
                  f"Acc={accuracy_score(yt.iloc[va], pred):.4f}")

        print(classification_report(yt, oof, zero_division=0))
        for i, m in enumerate(models):
            joblib.dump(m, outdir / f"clf_{tgt}_fold{i+1}.joblib")
        joblib.dump(le, outdir / f"le_{tgt}.joblib")

    # ===== 회귀 타깃: 단일 모델 =====
    for tgt in REG_TARGETS:
        if tgt not in df.columns or df[tgt].notna().sum() < 10:
            print(f"[SKIP] {tgt}: 유효 샘플 부족")
            continue
        mask = df[tgt].notna()
        Xt, yt = X.loc[mask], df.loc[mask, tgt].astype(float)

        print(f"\n[Train/REG] {tgt}")
        pipe = Pipeline([("pre", preprocess), ("rf", make_rf_reg(42))])
        pipe.fit(Xt, yt)
        joblib.dump(pipe, outdir / f"reg_{tgt}.joblib")

    print(f"[OK] 모델 저장 완료 → {outdir}")

if __name__ == "__main__":
    main()
