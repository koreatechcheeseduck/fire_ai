# -*- coding: utf-8 -*-
"""
각 타깃별로 학습된 scikit-learn 파이프라인(.joblib)을 ONNX로 변환합니다.
- ColumnTransformer의 수치/범주 컬럼 구성을 모델에서 직접 추출
- 그에 맞춘 dtype의 샘플 DataFrame으로 to_onnx 호출 (dtype 혼동 방지)
- 분류기는 zipmap=False 로 확률을 텐서로 출력
- 각 타깃의 라벨 파일(*.labels.json)도 함께 저장

사용:
    python scripts/export_onnx_per_target.py --model_dir models/rf_v1 --out_dir models/onnx_v1 --opset 17
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from skl2onnx import to_onnx

# ---- 프로젝트 내부의 입력 피처 정의를 쓰되, 실패 시 로컬 fallback ----
DEFAULT_INPUT_FEATURES: List[str] = [
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

def _load_input_features() -> List[str]:
    try:
        # 루트에서 실행할 때 import 가능
        from src.infer_model import INPUT_FEATURES  # type: ignore
        if isinstance(INPUT_FEATURES, list) and INPUT_FEATURES:
            return INPUT_FEATURES
    except Exception:
        pass
    return DEFAULT_INPUT_FEATURES


def _find_column_transformer(pipe) -> Tuple[Optional[object], Optional[List[str]], Optional[List[str]]]:
    """
    파이프라인에서 ColumnTransformer와 그 안의 수치/범주 컬럼 목록을 찾아서 반환.
    - 우선 'preprocess'/'preprocessor' 스텝을 찾고, 없으면 첫 번째 ColumnTransformer를 탐색
    """
    ct = None
    num_cols: Optional[List[str]] = None
    cat_cols: Optional[List[str]] = None

    if hasattr(pipe, "named_steps"):
        # 흔한 이름 우선 탐색
        for key in ("preprocess", "preprocessor"):
            if key in pipe.named_steps:
                maybe = pipe.named_steps[key]
                if maybe.__class__.__name__ == "ColumnTransformer":
                    ct = maybe
                    break

        # 못 찾았으면 전체 스텝에서 ColumnTransformer 검색
        if ct is None:
            for obj in pipe.named_steps.values():
                if obj.__class__.__name__ == "ColumnTransformer":
                    ct = obj
                    break

    # column 목록 가져오기
    if ct is not None and hasattr(ct, "transformers_"):
        for name, trans, cols in ct.transformers_:
            # passthrough 도 존재할 수 있으니 검사
            if cols is None or cols == "remainder":
                continue
            # 흔히 num/cat 라는 이름을 씁니다.
            if isinstance(cols, list):
                if name.lower().startswith("num"):
                    num_cols = cols
                elif name.lower().startswith("cat"):
                    cat_cols = cols

        # 이름이 특이한 경우, 타입으로 추정 (숫자 imputer/원핫 유무)
        if (num_cols is None or cat_cols is None) and hasattr(ct, "transformers_"):
            tmp_num, tmp_cat = [], []
            for name, trans, cols in ct.transformers_:
                if cols is None or cols == "remainder":
                    continue
                if isinstance(cols, list):
                    # 간단 추정: OneHotEncoder가 들어있는 transformer는 범주
                    has_ohe = False
                    if hasattr(trans, "named_steps"):
                        for v in trans.named_steps.values():
                            if v.__class__.__name__ == "OneHotEncoder":
                                has_ohe = True
                                break
                    if has_ohe:
                        tmp_cat.extend(cols)
                    else:
                        tmp_num.extend(cols)
            if num_cols is None and tmp_num:
                num_cols = tmp_num
            if cat_cols is None and tmp_cat:
                cat_cols = tmp_cat

    return ct, num_cols, cat_cols


def _build_sample_df(all_cols: List[str], num_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    """
    변환용 더미 DataFrame 생성 (dtype 엄격히 설정).
    - 수치 컬럼: float64 (0.0)
    - 범주 컬럼: object (빈 문자열)
    - 나머지 컬럼: object (빈 문자열)
    """
    data: Dict[str, object] = {}
    num_set = set(num_cols)
    cat_set = set(cat_cols)
    for c in all_cols:
        if c in num_set:
            data[c] = np.array([0.0], dtype=np.float64)
        else:
            # 범주 + 기타 모두 object(str)
            data[c] = np.array([""], dtype=object)
    df = pd.DataFrame(data, columns=all_cols)
    # 안전하게 dtype 캐스팅(특히 object 유지)
    for c in num_cols:
        df[c] = df[c].astype(np.float64)
    for c in cat_cols:
        df[c] = df[c].astype(object)
    return df


def _is_classifier(final_estimator) -> bool:
    # 일반적으로 분류기는 predict_proba 존재
    return hasattr(final_estimator, "predict_proba")


def _export_labels_if_any(model_path: Path, out_path_noext: Path):
    """
    같은 디렉터리에 존재하는 라벨 파일을 찾아 함께 저장.
    - 규칙: <target>_labels.joblib, 또는 파이프라인 최종추정기의 classes_
    """
    # 1) *_labels.joblib 찾기
    labels_candidates = list(model_path.parent.glob("*labels.joblib"))
    by_target: Dict[str, List[str]] = {}
    for p in labels_candidates:
        try:
            obj = joblib.load(p)
            # dict[str, list] 또는 list 로 저장된 경우를 모두 지원
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (list, np.ndarray)):
                        by_target[k] = list(v)
            elif isinstance(obj, (list, np.ndarray)):
                # 파일명이 '<target>_labels.joblib' 라고 가정
                tgt_name = p.stem.replace("_labels", "")
                by_target[tgt_name] = list(obj)
        except Exception:
            pass

    # 2) 파이프라인 자체에서 classes_ 추정
    try:
        pipe = joblib.load(model_path)
        final_est = getattr(pipe, "named_steps", {}).get("model", None)
        if final_est is None:
            # 마지막 스텝이 model이라고 가정 못하는 경우, 마지막 스텝 사용
            if hasattr(pipe, "named_steps") and pipe.named_steps:
                final_est = list(pipe.named_steps.values())[-1]
        if final_est is not None and hasattr(final_est, "classes_"):
            # 어느 타깃인지 추정
            tgt = model_path.stem.replace("_rf", "")
            by_target.setdefault(tgt, list(map(lambda x: str(x), list(final_est.classes_))))
    except Exception:
        pass

    # 저장
    for tgt, labels in by_target.items():
        out_json = out_path_noext.parent / f"{tgt}.labels.json"
        try:
            out_json.write_text(json.dumps({"labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


def export_per_target(model_dir: str, out_dir: str, opset: int = 17):
    model_dir_p = Path(model_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    jobs = sorted(model_dir_p.glob("*.joblib"))
    # rf, gb, etc 모두 포함될 수 있으니 *.joblib 필터링
    for job_path in jobs:
        tgt = job_path.stem.replace("_rf", "")  # 학습 스크립트에서 *_rf.joblib로 저장했다면 이 규칙으로 타깃 추정
        out_onnx = out_dir_p / f"{tgt}.onnx"

        print(f"[INFO] convert {tgt} -> {out_onnx}")
        pipe = joblib.load(job_path)

        # ColumnTransformer & 컬럼 목록 추출
        ct, num_cols, cat_cols = _find_column_transformer(pipe)
        if ct is None or num_cols is None or cat_cols is None:
            # 최후 수단: infer_model 정의 or fallback
            all_cols = _load_input_features()
            # 단순 기준으로 나눔 (숫자형 후보)
            default_num = {
                "building_agreement_count", "total_floor_area", "soot_area",
                "unit_temperature", "unit_humidity", "unit_wind_speed", "total_floor_count"
            }
            num_cols = [c for c in all_cols if c in default_num]
            cat_cols = [c for c in all_cols if c not in default_num]
        else:
            # ColumnTransformer에서 사용되는 컬럼만 사용
            all_cols = list(dict.fromkeys([*num_cols, *cat_cols]))

        # 샘플 DF (정확한 dtype 설정)
        df = _build_sample_df(all_cols, num_cols, cat_cols)

        # 분류기/회귀기 옵션
        final_est = getattr(pipe, "named_steps", {}).get("model", None)
        if final_est is None and hasattr(pipe, "named_steps") and pipe.named_steps:
            final_est = list(pipe.named_steps.values())[-1]

        onnx_options = {}
        if final_est is not None and _is_classifier(final_est):
            # 분류기: zipmap=False 로 확률을 텐서로
            onnx_options = {id(final_est): {"zipmap": False}}

        # 변환 (DataFrame을 직접 넘겨 dtype 혼란 제거)
        onx = to_onnx(
            pipe,
            df,
            target_opset=opset,
            options=onnx_options,
        )

        with open(out_onnx, "wb") as f:
            f.write(onx.SerializeToString())

        # 라벨 저장(있으면)
        _export_labels_if_any(job_path, out_onnx.with_suffix(""))

    print("[DONE] ONNX per-target export completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="학습된 파이프라인(.joblib)들이 있는 디렉토리")
    parser.add_argument("--out_dir", required=True, help="ONNX를 저장할 디렉토리")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset (기본 17)")
    args = parser.parse_args()

    print(f"[INFO] model_dir = {args.model_dir}")
    print(f"[INFO] out_dir   = {args.out_dir}")
    print(f"[INFO] opset     = {args.opset}")

    export_per_target(args.model_dir, args.out_dir, args.opset)


if __name__ == "__main__":
    sys.exit(main())
