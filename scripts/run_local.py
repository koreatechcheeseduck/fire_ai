# scripts/run_local.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

# --- src 임포트가 어디서든 되도록 프로젝트 루트 추가 ---
THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- 모델 래퍼 ---
try:
    from src.infer_model import EnsembleRF  # LocalEnsembleRF에 alias 걸려 있어도 OK
except Exception as e:
    print(f"[ERROR] src.infer_model에서 EnsembleRF를 import하지 못했습니다: {e}")
    raise


def load_payload(path: Optional[str]) -> Dict[str, Any]:
    """JSON 경로가 있으면 로드, 없으면 기본 예시 입력 반환."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 파일이 없을 때 사용할 기본 입력 (너가 올려준 케이스)
    return {
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


def main():
    parser = argparse.ArgumentParser(description="로컬에서 바로 예측 실행/출력 스크립트")
    parser.add_argument(
        "--json",
        required=False,
        help="입력 JSON 파일 경로 (생략하면 기본 예시 payload 사용)"
    )
    parser.add_argument(
        "--model_dir",
        default=str(PROJECT_ROOT / "models" / "rf_v1"),
        help="학습된 모델 폴더 (기본: models/rf_v1)"
    )
    parser.add_argument(
        "--index_dir",
        default=str(PROJECT_ROOT / "models" / "index_v1"),
        help="유사사례 인덱스 폴더 (없어도 됨, 기본: models/index_v1)"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="예쁜 포맷으로 출력"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="유사사례 상위 K (기본: 10)"
    )
    args = parser.parse_args()

    payload = load_payload(args.json)

    # 모델 로드
    model = EnsembleRF(model_dir=args.model_dir, index_dir=args.index_dir)

    # 예측 수행 (유사사례 출력 포함)
    result = model.predict(payload, return_similar=True, topk=args.topk)

    # 출력
    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
