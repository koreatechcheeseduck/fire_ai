import argparse
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, accuracy_score
import joblib

CATEGORICAL = [
    "fire_type_nm","bldg_frame_nm","bldg_structure_nm","bldg_inside_nm",
    "power_source_nm","ignition_heat_nm","first_ignition_nm",
    "combusion_enlarger_nm","facility_info_nm"
]
NUMERIC = [
    "total_floors","ignition_start_floor","hr_unit_temp",
    "dispatch_time","prpt_dam_amount"
]
TARGET_CAUSE = "ignition_nm"
TARGET_IGN = "ignition_structure_nm"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # numeric coercion & ensure columns exist
    for c in ['ground_high','ground_low','ignition_start_floor','hr_unit_temp',
              'dispatch_time','death_count','injry_count','human_dam_count','prpt_dam_amount']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            if c in ['ground_high','ground_low']:
                df[c] = np.nan

    # total floors safe
    if 'ground_high' in df.columns and 'ground_low' in df.columns:
        df['total_floors'] = df[['ground_high','ground_low']].sum(axis=1, min_count=1)
    else:
        df['total_floors'] = np.nan

    if "power_source_nm" in df.columns:
        df["has_multiple_powers"] = df["power_source_nm"].fillna("").str.contains(",").astype(int)

    for c in NUMERIC:
        if c not in df.columns:
            df[c] = np.nan
    for c in CATEGORICAL:
        if c not in df.columns:
            df[c] = ""

    return df

def build_preprocess():
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    num = Pipeline([("imp", SimpleImputer(strategy="median"))])
    return ColumnTransformer([("cat", cat, CATEGORICAL), ("num", num, NUMERIC)])

def train_head(X,y,preprocess,seed=42,n_estimators=400,n_splits=5):
    # Guard: need at least 2 classes for stratified split
    if len(set(y)) < 2:
        raise ValueError("Need at least 2 classes to train this head.")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models, oof = [], np.empty(len(y),dtype=int)
    for i,(tr,va) in enumerate(skf.split(X,y)):
        rf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",
                                    random_state=seed+i, n_jobs=-1)
        pipe = Pipeline([("pre",preprocess),("rf",rf)])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        models.append(pipe)
        pred = pipe.predict(X.iloc[va])
        oof[va] = pred
        print(f"[Fold {i+1}] F1={f1_score(y.iloc[va],pred,average='macro'):.4f}, Acc={accuracy_score(y.iloc[va],pred):.4f}")
    print(classification_report(y,oof))
    return models

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="models/rf_v1")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    df = pd.read_csv(args.input)
    df = engineer_features(df)
    Path("data/processed").mkdir(parents=True,exist_ok=True)
    df.to_parquet("data/processed/dataset.parquet", index=False)

    feat_cols = CATEGORICAL+NUMERIC
    preprocess = build_preprocess()

    # Cause head
    if df[TARGET_CAUSE].notna().sum()>0:
        mask = df[TARGET_CAUSE].notna()
        Xc, yc = df.loc[mask,feat_cols].reset_index(drop=True), df.loc[mask,TARGET_CAUSE].astype(str).reset_index(drop=True)
        le_cause = LabelEncoder().fit(yc)
        yc_enc = pd.Series(le_cause.transform(yc))
        if len(set(yc_enc)) >= 2:
            print("\\n[Train] Cause Head")
            models_cause = train_head(Xc,yc_enc,preprocess)
            for i,m in enumerate(models_cause):
                joblib.dump(m,outdir/f"rf_cause_fold{i+1}.joblib")
            joblib.dump(le_cause,outdir/"label_encoder_cause.joblib")
        else:
            print("[WARN] Cause head has <2 classes; skipped.")
    else:
        print("[WARN] ignition_nm labels missing; skipped cause head.")

    # Ignition head
    if df[TARGET_IGN].notna().sum()>0:
        mask = df[TARGET_IGN].notna()
        Xi, yi = df.loc[mask,feat_cols].reset_index(drop=True), df.loc[mask,TARGET_IGN].astype(str).reset_index(drop=True)
        le_ign = LabelEncoder().fit(yi)
        yi_enc = pd.Series(le_ign.transform(yi))
        if len(set(yi_enc)) >= 2:
            print("\\n[Train] Ignition Head")
            models_ign = train_head(Xi,yi_enc,preprocess)
            for i,m in enumerate(models_ign):
                joblib.dump(m,outdir/f"rf_ignition_fold{i+1}.joblib")
            joblib.dump(le_ign,outdir/"label_encoder_ign.joblib")
        else:
            print("[WARN] Ignition head has <2 classes; skipped.")
    else:
        print("[WARN] ignition_structure_nm labels missing; skipped ignition head.")

    print("[OK] Saved models to", outdir)

if __name__=="__main__":
    main()
