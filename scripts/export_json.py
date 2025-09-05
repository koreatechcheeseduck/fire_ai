# -*- coding: utf-8 -*-
"""
fire_all.onnx 메타정보를 '하나의 JSON'으로 내보내기
- inputs/outputs 시그니처
- 타깃 유형(회귀/분류) 추론 및 분류 라벨(classes_)
- 샘플 요청 본문
사용:
  python scripts/export_one_json.py \
    --onnx models/onnx_merged/fire_all.onnx \
    --model_dir models/rf_v1 \
    --out models/onnx_merged/fire_all.meta.json
"""
from __future__ import annotations
import argparse, json, datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import onnx
import joblib
from sklearn.base import ClassifierMixin

# 학습에 사용한 18개 입력 컬럼 (보기 좋게 정렬 용)
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

def _dtype_str(vi) -> str:
    if vi.type.WhichOneof("value") != "tensor_type":
        return "unknown"
    dt = vi.type.tensor_type.elem_type
    name = onnx.TensorProto.DataType.Name(dt)  # "FLOAT","INT64","STRING", ...
    return {"FLOAT": "float", "INT64": "int64", "STRING": "string"}.get(name, name.lower())

def _shape_list(vi) -> List[Optional[int]]:
    if vi.type.WhichOneof("value") != "tensor_type":
        return []
    dims = vi.type.tensor_type.shape.dim
    out = []
    for d in dims:
        if d.HasField("dim_value"):
            out.append(int(d.dim_value))
        else:
            out.append(None)
    return out

def _read_signature(onnx_path: Path):
    m = onnx.load(str(onnx_path))
    # opset/ir 버전 정보
    opsets = {imp.domain or "ai.onnx": imp.version for imp in m.opset_import}
    ir_version = getattr(m, "ir_version", None)

    inputs = [{"name": vi.name, "dtype": _dtype_str(vi), "shape": _shape_list(vi)} for vi in m.graph.input]
    outputs = [{"name": vi.name, "dtype": _dtype_str(vi), "shape": _shape_list(vi)} for vi in m.graph.output]

    # 입력은 학습 피처 순으로 정렬
    idx = {n: i for i, n in enumerate(INPUT_FEATURES)}
    inputs.sort(key=lambda x: idx.get(x["name"], 1_000))

    return m, inputs, outputs, opsets, ir_version

def _guess_targets(outputs: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    *_variable -> 회귀, *_label / *_probabilities -> 분류
    """
    info: Dict[str, str] = {}
    for o in outputs:
        n = o["name"]
        if n.endswith("_variable"):
            info[n.replace("_variable","")] = "regression"
        elif n.endswith("_label") or n.endswith("_probabilities"):
            info[n.rsplit("_", 1)[0]] = "classification"
    return info

def _choose_model_path(model_dir: Path, target: str) -> Optional[Path]:
    cand = model_dir / f"{target}_rf.joblib"
    if cand.exists(): return cand
    cands = sorted(model_dir.glob(f"{target}_clf*.joblib"))
    return cands[0] if cands else None

def _extract_classes(model_path: Optional[Path]) -> Optional[List[str]]:
    if not model_path: return None
    try:
        pipe = joblib.load(model_path)
    except Exception:
        return None
    last = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe
    if isinstance(last, ClassifierMixin) or hasattr(last, "classes_"):
        labels = getattr(last, "classes_", None)
        if labels is None: return None
        return ["" if x is None else str(x) for x in list(labels)]
    return None

def _build_sample(inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    sample = {}
    for it in inputs:
        dt = it["dtype"]
        if dt == "float":
            sample[it["name"]] = 0.0
        elif dt == "int64":
            sample[it["name"]] = 0
        else:
            sample[it["name"]] = ""
    return sample

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="병합된 ONNX 경로 (fire_all.onnx)")
    ap.add_argument("--model_dir", required=True, help="학습 파이프라인(.joblib) 폴더 (분류 라벨용)")
    ap.add_argument("--out", required=True, help="메타 JSON 저장 경로 (예: models/onnx_merged/fire_all.meta.json)")
    args = ap.parse_args()

    onnx_path = Path(args.onnx).resolve()
    model_dir = Path(args.model_dir).resolve()
    out_path  = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    m, inputs, outputs, opsets, ir_version = _read_signature(onnx_path)
    targets = _guess_targets(outputs)

    classes_map: Dict[str, List[str]] = {}
    for tgt, typ in targets.items():
        if typ != "classification":
            continue
        mp = _choose_model_path(model_dir, tgt)
        cls = _extract_classes(mp)
        if cls:
            classes_map[tgt] = cls

    meta = {
        "model": {
            "path": str(onnx_path),
            "ir_version": ir_version,
            "opsets": opsets,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        },
        "inputs": inputs,
        "outputs": outputs,
        "targets": targets,        # {target: "classification"|"regression"}
        "classes": classes_map,    # 분류 타깃만 {target: [label0, ...]} (probabilities 인덱스와 동일 순서)
        "sample_request": _build_sample(inputs)
    }

    out_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE] meta written:", out_path)

if __name__ == "__main__":
    main()
