# -*- coding: utf-8 -*-
"""
모든 ONNX 모델 입·출력 점검 + 1건 샘플 추론(+분류 Top-3 라벨/확률)
- 파일 발견 개수/경로를 먼저 출력
- 모델별 입출력/샘플 추론 출력
- 예외는 traceback까지 상세 표시
사용:
  python scripts/verify_onnx.py --onnx_dir models/onnx_v1 --schema_csv data/raw/fire_incidents_aligned.csv --labels_dir models/rf_v1
"""
from __future__ import annotations
import argparse, traceback
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import onnxruntime as rt
import joblib

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

def load_schema_row(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, nrows=1)
    for c in INPUT_FEATURES:
        if c not in df.columns:
            df[c] = ""
    df = df[INPUT_FEATURES].copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c].dtype):
            df[c] = df[c].astype("float32")
        elif pd.api.types.is_integer_dtype(df[c].dtype):
            df[c] = df[c].astype("int64")
        else:
            df[c] = df[c].astype("object")
            if pd.isna(df.loc[0, c]):
                df.loc[0, c] = ""
    return df.iloc[0]

def build_ort_inputs(sample: pd.Series, input_names: List[str]) -> Dict[str, np.ndarray]:
    feed: Dict[str, np.ndarray] = {}
    for name in input_names:
        val = sample[name] if name in sample.index else ""
        if isinstance(val, (np.floating, float)):
            arr = np.array([[val]], dtype=np.float32)
        elif isinstance(val, (np.integer, int, np.int64)):
            arr = np.array([[int(val)]], dtype=np.int64)
        else:
            arr = np.array([[str(val)]], dtype=object)
        feed[name] = arr
    return feed

def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)

def load_labels(labels_dir: Path, target: str):
    p = labels_dir / f"{target}_labels.joblib"
    if p.exists():
        return list(joblib.load(p))
    return None

def interpret_outputs(raw: List[np.ndarray], out_names: List[str], labels):
    # 회귀: 출력 1개 float, shape 끝이 1
    if len(raw) == 1 and str(raw[0].dtype).startswith("float"):
        return {"type": "regression", "value": float(np.array(raw[0]).reshape(-1)[0])}

    # 분류: float 2D 하나를 확률로 간주
    prob = None
    for arr in raw:
        if str(arr.dtype).startswith("float") and arr.ndim == 2:
            prob = np.array(arr)
            break
    if prob is not None:
        if not np.allclose(prob.sum(axis=1), 1.0, atol=1e-4):
            prob = softmax(prob)
        topk = min(3, prob.shape[1])
        idxs = np.argsort(-prob[0])[:topk]
        top = []
        for j in idxs:
            name = labels[j] if labels and j < len(labels) else int(j)
            top.append({"label": name, "prob": float(prob[0, j])})
        return {"type": "classification", "pred": top[0]["label"], "conf": top[0]["prob"], "top3": top}

    return {"type": "unknown", "raw_shapes": [(out_names[i], np.array(raw[i]).shape, str(np.array(raw[i]).dtype)) for i in range(len(raw))]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_dir", required=True)
    ap.add_argument("--schema_csv", required=True)
    ap.add_argument("--labels_dir", default="models/rf_v1")
    args = ap.parse_args()

    onnx_dir = Path(args.onnx_dir).resolve()
    schema_csv = Path(args.schema_csv).resolve()
    labels_dir = Path(args.labels_dir).resolve()

    print(f"[INFO] onnx_dir  = {onnx_dir}")
    print(f"[INFO] schema   = {schema_csv}")
    print(f"[INFO] labels   = {labels_dir}")

    files = sorted(onnx_dir.glob("*.onnx"))
    print(f"[INFO] discovered {len(files)} ONNX files")
    if not files:
        print("[HINT] 경로가 맞는지 확인하세요. (예: models/onnx_v1)")
        return 1

    sample = load_schema_row(schema_csv)

    for p in files:
        print(f"\n=== {p.name} ===")
        try:
            sess = rt.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            in_names = [i.name for i in sess.get_inputs()]
            out_names = [o.name for o in sess.get_outputs()]
            print("inputs :", [(i.name, i.type, i.shape) for i in sess.get_inputs()])
            print("outputs:", [(o.name, o.type, o.shape) for o in sess.get_outputs()])

            feed = build_ort_inputs(sample, in_names)
            raw = sess.run(None, feed)

            target = p.stem
            labels = load_labels(labels_dir, target)
            parsed = interpret_outputs(raw, out_names, labels)

            if parsed["type"] == "regression":
                print("prediction:", parsed["value"])
            elif parsed["type"] == "classification":
                print("pred:", parsed["pred"], " conf:", round(parsed["conf"], 4))
                print("top3:", parsed["top3"])
            else:
                print("unknown outputs:", parsed["raw_shapes"])

        except Exception as e:
            print("[ERROR] failed:", p.name)
            traceback.print_exc()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
