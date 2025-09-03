# -*- coding: utf-8 -*-
"""
유사 사례 검색용 인덱스 생성 스크립트
- 입력: Parquet 또는 CSV (정제 데이터)
- 출력: TF-IDF 벡터라이저, 최근접이웃 인덱스, 메타 정보
사용 예:
  python scripts/build_index.py --input data/processed/dataset.parquet --outdir models/index_v1
  python scripts/build_index.py --input data/raw/fire_incidents_aligned.csv --outdir models/index_v1 --sep ","
  python scripts/build_index.py --input data/processed/dataset.parquet --outdir models/index_v1 --cols building_structure ignition_material fire_type
"""

import argparse
import os
import sys
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# 새/옛 스키마를 모두 포괄하는 "후보 컬럼" 목록
# (존재하는 것만 자동 사용)
CANDIDATE_TEXT_COLS: List[str] = [
    # 새 스키마(영문)
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
    "ignition_device",
    "ignition_heat_source",
    "ignition_cause",
    "fire_type",

    # 과거 스키마(예시, *_nm 등)
    "fire_type_nm",
    "bldg_frame_nm",
    "bldg_structure_nm",
    "bldg_inside_nm",
    "ignition_nm",
    "ignition_structure_nm",
]

ID_CANDIDATES = ["id", "case_id", "incident_id", "index"]


def read_table(path: str, sep: str | None) -> pd.DataFrame:
    # parquet 우선, 실패 시 CSV
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".feather"):
        return pd.read_feather(path)
    # CSV류
    if sep is None:
        # sep 미지정 시 자동 추정 시도 → 실패하면 콤마
        try:
            return pd.read_csv(path, encoding="utf-8")
        except Exception:
            return pd.read_csv(path, sep=",", encoding="utf-8", engine="python")
    return pd.read_csv(path, sep=sep, encoding="utf-8", engine="python")


def pick_id_column(df: pd.DataFrame) -> str:
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    # 없으면 인덱스에서 생성
    df["_row_id"] = np.arange(len(df))
    return "_row_id"


def build_text(df: pd.DataFrame, use_cols: List[str]) -> pd.Series:
    # 존재하는 컬럼만 골라서 문자열 결합
    cols = [c for c in use_cols if c in df.columns]
    if not cols:
        # 후보 전체에서 존재하는 것 체크
        exist = [c for c in CANDIDATE_TEXT_COLS if c in df.columns]
        raise RuntimeError(
            "텍스트를 만들 컬럼이 없습니다. --cols 로 사용할 컬럼을 지정하거나 "
            f"데이터 내 존재 컬럼 중에서 선택하세요. (데이터 내 유사 후보: {exist[:20]})"
        )
    # 문자열로 캐스팅 + 결측치 빈문자 처리 + 공백으로 결합
    parts = []
    for c in cols:
        s = df[c].astype(str).replace("nan", "", regex=False)
        parts.append(s)
    text = parts[0]
    for s in parts[1:]:
        text = (text + " " + s)
    # 전처리(소문자/양끝 공백 제거)
    text = text.str.lower().str.strip()
    return text, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="입력 테이블 (parquet/csv)")
    ap.add_argument("--outdir", required=True, help="출력 디렉터리")
    ap.add_argument("--sep", default=None, help="CSV 구분자(선택)")
    ap.add_argument(
        "--cols",
        nargs="*",
        default=None,
        help="텍스트 구성에 사용할 컬럼들(공백 구분). 지정 안하면 후보 목록 CANDIDATE_TEXT_COLS에서 존재 컬럼 자동 선택",
    )
    ap.add_argument("--max_features", type=int, default=200_000)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--ngram_max", type=int, default=2, help="TF-IDF ngram 상한 (1~N)")
    ap.add_argument("--n_neighbors", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"[INFO] load -> {args.input}")
    df = read_table(args.input, sep=args.sep)
    print(f"[INFO] rows={len(df):,}, cols={len(df.columns)}")

    # 사용할 텍스트 컬럼 결정
    use_cols = args.cols if args.cols else CANDIDATE_TEXT_COLS
    text, used_cols = build_text(df, use_cols)
    missing = [c for c in use_cols if c not in df.columns]
    print(f"[INFO] text columns used ({len(used_cols)}): {used_cols}")
    if missing:
        print(f"[WARN] missing columns ignored ({len(missing)}): {missing}")

    # ID 컬럼 선택/생성
    id_col = pick_id_column(df)
    print(f"[INFO] id column: {id_col}")

    # TF-IDF
    print("[INFO] fit TF-IDF …")
    vect = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_range=(1, max(1, int(args.ngram_max))),
    )
    X = vect.fit_transform(text.values.tolist())
    print(f"[INFO] TF-IDF shape = {X.shape}")

    # NNeighbors (cosine)
    print("[INFO] fit NearestNeighbors (metric=cosine) …")
    nn = NearestNeighbors(n_neighbors=args.n_neighbors, metric="cosine", n_jobs=-1)
    nn.fit(X)

    # 메타(원문 텍스트/키 컬럼 일부) 저장
    meta_cols = list(dict.fromkeys([id_col] + used_cols))  # 중복 제거 & 순서 유지
    meta = df[meta_cols].copy()
    meta["__text__"] = text

    # 저장
    vect_path = os.path.join(args.outdir, "tfidf_vectorizer.joblib")
    nn_path = os.path.join(args.outdir, "nn_index.joblib")
    mat_path = os.path.join(args.outdir, "tfidf_csr.joblib")
    meta_path = os.path.join(args.outdir, "meta.parquet")

    joblib.dump(vect, vect_path)
    joblib.dump(nn, nn_path)
    joblib.dump(X, mat_path)
    meta.to_parquet(meta_path, index=False)

    print(f"[OK] saved vectorizer   -> {vect_path}")
    print(f"[OK] saved NN index     -> {nn_path}")
    print(f"[OK] saved TF-IDF CSR   -> {mat_path}")
    print(f"[OK] saved meta         -> {meta_path}")
    print("[DONE] index build complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e, file=sys.stderr)
        sys.exit(1)
