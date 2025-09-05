# -*- coding: utf-8 -*-
"""
여러 타깃 ONNX(컬럼별 입력, 동일 스키마)를 '하나의 ONNX'로 병합 (tap 방식)
- export_onnx_per_target.py 와 동일한 INPUT_FEATURES를 기준으로 시그니처 검사
- --src 아래를 재귀(rglob)로 *.onnx 수집
- 베이스 그래프에 각 입력을 N개(타깃 수)로 'Identity' 노드로 복제한 tap 출력 생성:
    __tap{idx}_{feature}
  → 매 병합에서 서로 다른 tap을 g2의 입력에 연결(소모 문제 해결)
- 병합 완료 후 tap 출력은 제거하고, 타깃별 출력만 남긴다.

사용:
  python scripts/merge_onnx.py --src models/onnx_v1 --dst models/onnx_merged --out fire_all.onnx
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import onnx
from onnx import helper, checker
from onnx.compose import merge_models

# === export_onnx_per_target.py 와 정확히 동일해야 함 ===
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

TARGET_ORDER = [
    "casualty_count",
    "property_damage_amount",
    "fire_type",
    "ignition_cause",
    "ignition_device",
    "ignition_heat_source",
    "combustion_expansion_material",
    "fire_station_name",
]

def _discover_models(src: Path) -> List[Path]:
    return sorted(src.rglob("*.onnx"))

def _tensor_type_str(vi: onnx.ValueInfoProto) -> str:
    if vi.type.WhichOneof("value") == "tensor_type":
        return onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)
    return "NON_TENSOR"

def _tensor_shape_tuple(vi: onnx.ValueInfoProto) -> Tuple:
    if vi.type.WhichOneof("value") != "tensor_type":
        return tuple()
    dims = vi.type.tensor_type.shape.dim
    return tuple([d.dim_value if d.HasField("dim_value") else None for d in dims])

def _signature_of_inputs(m: onnx.ModelProto) -> Dict[str, Tuple[str, Tuple]]:
    return {vi.name: (_tensor_type_str(vi), _tensor_shape_tuple(vi)) for vi in m.graph.input}

def _make_base_with_taps(first: onnx.ModelProto, n_models: int) -> onnx.ModelProto:
    """
    첫 모델의 입력을 그대로 복제해 베이스 그래프 구성.
    + 각 입력마다 n_models개의 Identity 노드(tap)를 만들어
      __tap{idx}_{feature} 이름으로 출력에 노출한다.
    """
    inputs: List[onnx.ValueInfoProto] = []
    outputs: List[onnx.ValueInfoProto] = []
    nodes: List[onnx.NodeProto] = []

    # 입력 복제
    for vi in first.graph.input:
        new_in = helper.make_tensor_value_info(vi.name, 0, None)
        new_in.type.CopyFrom(vi.type)
        inputs.append(new_in)

    # tap 노드/출력 생성
    for idx in range(n_models):
        for vi in first.graph.input:
            tap_name = f"__tap{idx}_{vi.name}"
            # Identity 노드: in -> tap
            nodes.append(helper.make_node("Identity", inputs=[vi.name], outputs=[tap_name],
                                          name=f"Identity_tap{idx}_{vi.name}"))
            # tap을 출력으로도 노출 (merge에서 io_map의 g1 출력으로 사용)
            tap_out = helper.make_tensor_value_info(tap_name, 0, None)
            tap_out.type.CopyFrom(vi.type)
            outputs.append(tap_out)

    graph = helper.make_graph(
        nodes=nodes,
        name="fire_all_base",
        inputs=inputs,
        outputs=outputs,   # tap 출력만 노출
        initializer=[],
        value_info=[],
    )
    base = helper.make_model(graph)
    # opset / ir 버전 복사
    del base.opset_import[:]
    for op in first.opset_import:
        imp = base.opset_import.add()
        imp.domain = op.domain
        imp.version = op.version
    base.ir_version = first.ir_version
    return base

def _merge_all(models_named: List[Tuple[str, onnx.ModelProto]]) -> onnx.ModelProto:
    """
    models_named: [(target_name, model_proto), ...]
    - 각 타깃 i에 대해 io_map = [(__tap{i}_{feat}, feat)]  (★ 접두사 없이 원본 이름)
    - prefix2는 g2 전체에 자동 적용됨(출력/내부 이름 충돌 방지)
    """
    tgt0, m0 = models_named[0]
    merged = _make_base_with_taps(m0, n_models=len(models_named))

    for idx, (tgt, mdl) in enumerate(models_named):
        io_map = [(f"__tap{idx}_{vin.name}", vin.name) for vin in mdl.graph.input]
        merged = merge_models(
            merged, mdl,
            io_map=io_map,
            prefix2=f"{tgt}_",
        )

    # 병합 후: tap 출력(이름이 "__tap"로 시작)은 최종 출력에서 제거
    keep = [o for o in merged.graph.output if not o.name.startswith("__tap")]
    del merged.graph.output[:]
    merged.graph.output.extend(keep)

    checker.check_model(merged)
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="개별 타깃 onnx 들이 있는 루트 폴더 (재귀 검색)")
    ap.add_argument("--dst", required=True, help="병합 onnx 저장 폴더")
    ap.add_argument("--out", default="fire_all.onnx", help="출력 파일명")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] src = {src}")
    files = _discover_models(src)
    if not files:
        print("[ERROR] .onnx 파일을 찾지 못했습니다.")
        raise SystemExit(1)
    print(f"[INFO] found {len(files)} models:")
    for p in files:
        print("  -", p)

    # 모델 로드 + 타깃명
    loaded: List[Tuple[str, onnx.ModelProto, Path]] = []
    for p in files:
        mdl = onnx.load(str(p))
        tgt = p.stem
        loaded.append((tgt, mdl, p))

    # 선호 순서 정렬
    def _order_key(item):
        name = item[0]
        return (TARGET_ORDER.index(name) if name in TARGET_ORDER else 10_000, name)
    loaded.sort(key=_order_key)

    # 입력 시그니처 검증
    tgt0, m0, p0 = loaded[0]
    sig_ref = _signature_of_inputs(m0)
    if set(sig_ref.keys()) != set(INPUT_FEATURES):
        print("[ERROR] 첫 모델의 입력 이름 집합이 기대와 다릅니다.")
        print("  expected:", INPUT_FEATURES)
        print("  actual  :", list(sig_ref.keys()))
        raise SystemExit(1)

    for tgt, mdl, p in loaded[1:]:
        sig_cur = _signature_of_inputs(mdl)
        if set(sig_cur.keys()) != set(INPUT_FEATURES):
            print(f"[ERROR] 입력 이름 집합 불일치: {p}")
            print("  expected:", INPUT_FEATURES)
            print("  actual  :", list(sig_cur.keys()))
            raise SystemExit(1)
        mismatches = []
        for name in INPUT_FEATURES:
            t_ref, s_ref = sig_ref[name]
            t_cur, s_cur = sig_cur[name]
            if (t_ref != t_cur) or (s_ref != s_cur):
                mismatches.append((name, (t_ref, s_ref), (t_cur, s_cur)))
        if mismatches:
            print(f"[ERROR] 입력 타입/shape 불일치: {p}")
            for nm, ref, cur in mismatches:
                print(f"  - {nm}: ref={ref}, cur={cur}")
            raise SystemExit(1)

    models_named = [(t, m) for (t, m, _) in loaded]
    merged = _merge_all(models_named)

    out_path = dst / args.out
    onnx.save(merged, str(out_path))
    out_names = [o.name for o in merged.graph.output]
    print(f"[DONE] merged model saved to: {out_path}")
    print(f"[INFO] inputs ({len(merged.graph.input)}): {[i.name for i in merged.graph.input]}")
    print(f"[INFO] outputs({len(out_names)}): {out_names[:12]}{' ...' if len(out_names) > 12 else ''}")

if __name__ == "__main__":
    main()
