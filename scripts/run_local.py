#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- [중요] 프로젝트 루트를 import 경로에 추가 ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -----------------------------------------------

import argparse
import json
from typing import Any, Dict, Optional

from src.infer_model import EnsembleRF  # 위에서 sys.path를 잡았기 때문에 정상 import 됨


# 기본 입력 샘플 (--json 생략 시 사용)
DEFAULT_PAYLOAD: Dict[str, Any] = {
    "building_agreement_count": 1,
    "building_structure": "철근콘크리트조 슬라브",
    "building_usage_status": "사용중",
    "total_floor_area": 82,
    "soot_area": 0,
    "multi_use_flag": "N",
    "fuel_type": "",
    "fire_management_target_flag": "N",
    "unit_temperature": 4.4,
    "unit_humidity": 29,
    "unit_wind_speed": None,
    "facility_location": "공동주택",
    "forest_fire_flag": "N",
    "total_floor_count": 6,
    "vehicle_fire_flag": "N",
    "ignition_material": "담뱃불, 라이터불",
    "special_fire_object_name": "쓰레기류 기타 쓰레기",
    "wind_direction": "남서",
}


def load_payload(json_path: Optional[str]) -> Dict[str, Any]:
    if not json_path:
        return DEFAULT_PAYLOAD
    p = Path(json_path)
    text = p.read_text(encoding="utf-8")
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="입력 JSON 경로 (생략 시 기본 샘플 사용)", default=None)
    parser.add_argument("--model_dir", default="models/rf_v1")
    parser.add_argument("--index_dir", default="models/index_v1")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    payload = load_payload(args.json)

    model = EnsembleRF(model_dir=args.model_dir, index_dir=args.index_dir)
    result = model.predict(payload, return_similar=True, topk=10)

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
