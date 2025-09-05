# -*- coding: utf-8 -*-
"""
타깃별 ONNX 내보내기 (ColumnTransformer 컬럼명 스키마 + 문자열 Imputer 제거 + CT 탐색 버그 수정)
- 파이프라인: [ preprocess(ColumnTransformer) -> (RF clf/reg) ]
- 변환 시:
  1) 정제 CSV의 dtype으로 컬럼별 initial_types 생성(컬럼명 기반)  ← DataFrame을 컬럼들로 본다
  2) cat 파이프라인의 '문자형 SimpleImputer' 제거(ONNX-ML 미지원) ← 공식 예제 권장
  3) 분류기는 zipmap=False(확률 벡터)
참고: https://onnx.ai/sklearn-onnx/auto_examples/plot_complex_pipeline.html
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
)

# === 프로젝트 입력 피처(순서 유지) ===
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


# -------------------------------
# 유틸
# -------------------------------
def _is_ct(obj: Any) -> bool:
    """ColumnTransformer 판별 (isinstance + duck-typing)."""
    try:
        from sklearn.compose import ColumnTransformer  # local import
        if isinstance(obj, ColumnTransformer):
            return True
    except Exception:
        pass
    return hasattr(obj, "transformers") and obj.__class__.__name__ == "ColumnTransformer"


def _iter_estimators(est) -> Iterable[Any]:
    """
    파이프라인/CT 내부 추정기를 깊이 순회.
    (중요) '자기 자신'도 먼저 yield 한 뒤 내부로 들어간다. ← 이전 버그 수정
    """
    # 자신 먼저
    yield est

    # 내부 순회
    if isinstance(est, Pipeline):
        for _, sub in est.steps:
            yield from _iter_estimators(sub)
    elif _is_ct(est):
        for _, trans, _ in getattr(est, "transformers", []):
            if trans is not None and trans != "drop":
                yield from _iter_estimators(trans)


def _find_ct(est) -> Optional[Any]:
    """어디든 박혀 있는 ColumnTransformer 하나를 찾아 반환."""
    for sub in _iter_estimators(est):
        if _is_ct(sub):
            return sub
    return None


def _remove_string_imputer_in_cat(ct) -> None:
    """
    ColumnTransformer의 cat 파이프라인에서 '문자형 SimpleImputer' 스텝 제거.
    (숫자 Imputer는 유지)  ← ONNX-ML이 문자열 Imputer 미지원이라 변환 시 제거 권장
    """
    from sklearn.pipeline import Pipeline as SKPipeline

    def _strip_in_pipeline(pipe: SKPipeline):
        new_steps = []
        for name, tr in pipe.steps:
            if isinstance(tr, SimpleImputer):
                # fill_value가 문자열(None 포함) => 문자형으로 간주 → 제거
                fv = getattr(tr, "fill_value", None)
                if fv is None or isinstance(fv, str):
                    continue  # drop this step
            new_steps.append((name, tr))
        pipe.steps = new_steps  # in-place

    # 원본 정의(transformers) 처리
    for i, (name, trans, cols) in enumerate(list(ct.transformers)):
        if name == "cat" and isinstance(trans, SKPipeline):
            _strip_in_pipeline(trans)

    # fit 이후 내부(transformers_) 처리
    if hasattr(ct, "transformers_"):
        new_trs_ = []
        for t in ct.transformers_:
            name = t[0]
            trans = t[1]
            if name == "cat" and isinstance(trans, SKPipeline):
                _strip_in_pipeline(trans)
            new_trs_.append(t)
        ct.transformers_ = new_trs_


def _convert_dataframe_schema(df: pd.DataFrame, keep_cols: List[str]):
    """
    DataFrame을 '컬럼들의 집합'으로 보고, 각 컬럼명/타입을 ONNX initial_types로 매핑.
    (int64 -> Int64TensorType, float64 -> FloatTensorType, 그 외 -> StringTensorType)
    """
    init = []
    for c in keep_cols:
        if c not in df.columns:
            init.append((c, StringTensorType([None, 1])))
            continue
        dt = df[c].dtype
        if pd.api.types.is_integer_dtype(dt):
            init.append((c, Int64TensorType([None, 1])))
        elif pd.api.types.is_float_dtype(dt):
            init.append((c, FloatTensorType([None, 1])))
        else:
            init.append((c, StringTensorType([None, 1])))
    return init


def _discover_targets(model_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in model_dir.glob("*.joblib"):
        name = p.stem
        if name.endswith("_rf"):
            tgt = name[:-3]
        elif name.endswith("_clf") or name.endswith("_reg"):
            tgt = name.rsplit("_", 1)[0]
        else:
            continue
        out[tgt] = p
    return out


def _is_classifier(model) -> bool:
    last = model.steps[-1][1] if isinstance(model, Pipeline) else model
    return isinstance(last, ClassifierMixin) or hasattr(last, "predict_proba")


# -------------------------------
# 메인 변환
# -------------------------------
def export_per_target(model_dir: Path, out_dir: Path, schema_csv: Optional[Path], opset: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = _discover_targets(model_dir)
    if not targets:
        print(f"[WARN] 모델을 찾지 못했습니다: {model_dir}")
        return 1

    # 스키마용 DataFrame 로드 (정제 CSV 권장)
    df_schema = None
    if schema_csv is not None and schema_csv.exists():
        df_schema_full = pd.read_csv(schema_csv, nrows=200)
        cols = [c for c in INPUT_FEATURES if c in df_schema_full.columns]
        df_schema = df_schema_full[cols].copy()
        print(f"[INFO] schema_csv loaded: {schema_csv} (cols={list(df_schema.columns)})")
    else:
        print("[INFO] schema_csv not provided; will create schema from names only.")

    # 컬럼명 기반 initial_types 생성
    if df_schema is not None and len(df_schema.columns) > 0:
        initial_types = _convert_dataframe_schema(df_schema, INPUT_FEATURES)
    else:
        initial_types = [(c, StringTensorType([None, 1])) for c in INPUT_FEATURES]

    print(f"[INFO] initial_types count = {len(initial_types)}")
    print(f"[INFO] 발견된 타깃({len(targets)}): {', '.join(sorted(targets.keys()))}")

    failed: List[Tuple[str, str]] = []

    for tgt, joblib_path in sorted(targets.items()):
        out_path = out_dir / f"{tgt}.onnx"
        print(f"[INFO] convert {tgt} -> {out_path}")

        try:
            model = joblib.load(joblib_path)

            # 1) ColumnTransformer 확보(이제 제대로 잡힘)
            ct = _find_ct(model)
            if ct is None:
                raise RuntimeError("파이프라인에 ColumnTransformer가 없습니다.")

            # 2) cat 파이프라인의 '문자형 SimpleImputer' 제거 (ONNX 호환)
            _remove_string_imputer_in_cat(ct)

            # 3) 분류기 zipmap 옵션
            options = {id(model): {"zipmap": False}} if _is_classifier(model) else None

            # 4) 변환 (컬럼명 기반 initial_types만 사용)
            onx = convert_sklearn(
                model,
                name=f"{tgt}_pipeline",
                initial_types=initial_types,
                target_opset=opset,
                options=options,
            )
            with open(out_path, "wb") as f:
                f.write(onx.SerializeToString())

        except Exception as e:
            msg = str(e)
            print(f"[ERROR] {tgt}: {msg}")
            failed.append((tgt, msg))

    if failed:
        print("\n[SUMMARY] 변환 실패 타깃:")
        for tgt, msg in failed:
            print(f"  - {tgt}: {msg}")
        return 1

    print("[DONE] 모든 타깃 ONNX 변환 완료:", out_dir)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="학습된 joblib 파이프라인 폴더")
    ap.add_argument("--out_dir", required=True, help="ONNX 출력 폴더")
    ap.add_argument("--schema_csv", default="data/raw/fire_incidents_aligned.csv",
                    help="입력 컬럼 dtype 추출용 CSV(정제본). 없으면 이름만으로 스키마 생성")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    print(f"[INFO] model_dir = {args.model_dir}")
    print(f"[INFO] out_dir   = {args.out_dir}")
    print(f"[INFO] opset     = {args.opset}")
    return export_per_target(Path(args.model_dir), Path(args.out_dir), Path(args.schema_csv), args.opset)


if __name__ == "__main__":
    raise SystemExit(main())
