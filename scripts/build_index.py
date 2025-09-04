# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


TEXT_COLS_CANDIDATES: List[str] = [
    # 입력 피처들(문자/수치 모두 텍스트로 이어붙임)
    "building_structure", "building_usage_status", "total_floor_area", "soot_area",
    "multi_use_flag", "fuel_type", "fire_management_target_flag",
    "unit_temperature", "unit_humidity", "unit_wind_speed",
    "facility_location", "forest_fire_flag", "total_floor_count",
    "vehicle_fire_flag", "ignition_material", "special_fire_object_name", "wind_direction",
    # 타깃 일부(텍스트)
    "ignition_device", "ignition_heat_source", "ignition_cause", "fire_type",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Parquet/CSV (auto-detect)")
    ap.add_argument("--outdir", required=True, help="output dir")
    ap.add_argument("--id_col", default="_row_id")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ip = Path(args.input)
    print(f"[INFO] load -> {ip}")
    if ip.suffix.lower() == ".parquet":
        df = pd.read_parquet(ip)
    else:
        df = pd.read_csv(ip)

    # 행 id
    if args.id_col not in df.columns:
        df = df.reset_index(drop=False).rename(columns={"index": args.id_col})

    # 사용할 텍스트 컬럼
    cols = [c for c in TEXT_COLS_CANDIDATES if c in df.columns]
    missing = [c for c in TEXT_COLS_CANDIDATES if c not in df.columns]
    print(f"[INFO] rows={len(df):,}, cols={len(df.columns)}")
    print(f"[INFO] text columns used ({len(cols)}): {cols}")
    if missing:
        print(f"[WARN] missing columns ignored ({len(missing)}): {missing}")

    # 인덱싱용 텍스트 만들기
    def row_to_text(row):
        return " ".join(str(row.get(c, "")) for c in cols)

    texts = [row_to_text(r) for _, r in df.iterrows()]

    print("[INFO] fit TF-IDF …")
    vec = TfidfVectorizer(max_features=40000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    print(f"[INFO] TF-IDF shape = {X.shape}")

    print("[INFO] fit NearestNeighbors (metric=cosine) …")
    nn = NearestNeighbors(n_neighbors=20, metric="cosine", n_jobs=-1)
    nn.fit(X)

    # 메타(매칭 시 보여줄 최소 정보)
    meta = df[[args.id_col]].copy()
    for c in cols[:20]:
        meta[c] = df[c]
    meta_path = outdir / "meta.parquet"
    meta.to_parquet(meta_path, index=False)

    # 저장
    joblib.dump(vec, outdir / "tfidf_vectorizer.joblib")
    joblib.dump(nn, outdir / "nn_index.joblib")
    joblib.dump(X, outdir / "tfidf_csr.joblib")

    print("[OK] saved vectorizer   ->", outdir / "tfidf_vectorizer.joblib")
    print("[OK] saved NN index     ->", outdir / "nn_index.joblib")
    print("[OK] saved TF-IDF CSR   ->", outdir / "tfidf_csr.joblib")
    print("[OK] saved meta         ->", meta_path)
    print("[DONE] index build complete.")


if __name__ == "__main__":
    main()
