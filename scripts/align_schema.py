# scripts/align_schema.py
# CSV/XLSX 자동 감지 + 인코딩/구분자 강건화 + 입력/출력 스키마 정렬 + 지연시간 파생
import argparse, os, csv
from pathlib import Path
import pandas as pd
import numpy as np

# ===== 표준 입력(피처) / 출력(타깃) 스키마 =====
INPUT_COLS = [
    "building_agreement_count","building_structure","building_usage_status",
    "total_floor_area","soot_area","multi_use_flag","fuel_type",
    "fire_management_target_flag","unit_temperature","unit_humidity","unit_wind_speed",
    "facility_location","forest_fire_flag","total_floor_count","vehicle_fire_flag",
    "ignition_material","special_fire_object_name","wind_direction"
]

# 분류 타깃
TARGET_CLASS = [
    "ignition_device","ignition_heat_source","ignition_cause",
    "fire_station_name","combustion_expansion_material","fire_type"
]
# 회귀 타깃 (시간계는 분 단위 지연으로 파생)
TARGET_REG = [
    "casualty_count","property_damage_amount",
    "arrival_delay_min","initial_extinguish_delay_min"
]

# 실제 원본 컬럼명 후보 → 표준 스키마 매핑
MAP_IN = {
    "building_agreement_count": ["building_agreement_count","agreement_count"],
    "building_structure": ["building_structure","bldg_structure_nm","bldg_structure"],
    "building_usage_status": ["building_usage_status","usage_status"],
    "total_floor_area": ["total_floor_area","floor_area_total","total_area"],
    "soot_area": ["soot_area","smoke_area"],
    "multi_use_flag": ["multi_use_flag","is_multi_use","multi_use"],
    "fuel_type": ["fuel_type","power_source_nm","power_type"],
    "fire_management_target_flag": ["fire_management_target_flag","is_fire_mgmt_target","fire_mgmt_target"],
    "unit_temperature": ["unit_temperature","hr_unit_temp","temperature"],
    "unit_humidity": ["unit_humidity","humidity"],
    "unit_wind_speed": ["unit_wind_speed","wind_speed"],
    "facility_location": ["facility_location","bldg_inside_nm","location_in_facility"],
    "forest_fire_flag": ["forest_fire_flag","is_forest_fire","forest_fire"],
    "total_floor_count": ["total_floor_count","total_floors","ground_high"],
    "vehicle_fire_flag": ["vehicle_fire_flag","is_vehicle_fire","vehicle_fire"],
    "ignition_material": ["ignition_material","first_ignition_nm"],
    "special_fire_object_name": ["special_fire_object_name","special_bldg_nm","facility_info_nm"],
    "wind_direction": ["wind_direction","wd","wind_dir"]
}
MAP_OUT = {
    "ignition_device": ["ignition_device","ignition_structure_nm","ignition_object"],
    "ignition_heat_source": ["ignition_heat_source","ignition_heat_nm","heat_source"],
    "ignition_cause": ["ignition_cause","ignition_nm","cause"],
    "casualty_count": ["casualty_count","human_dam_count"],
    "fire_station_name": ["fire_station_name","station_nm"],
    "combustion_expansion_material": ["combustion_expansion_material","combusion_enlarger_nm"],
    "property_damage_amount": ["property_damage_amount","prpt_dam_amount"],
    "report_datetime": ["report_datetime","occur_datetime","report_time"],
    "initial_extinguish_datetime": ["initial_extinguish_datetime","initial_extinguish_time","fire_down_time","fire_down_hour"],
    "arrival_datetime": ["arrival_datetime","arrival_time"],
    "fire_type": ["fire_type","fire_type_nm"]
}

def _pick(df: pd.DataFrame, candidates, default=np.nan):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series([default]*len(df))

def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path.resolve()}")
    suf = path.suffix.lower()
    # Excel
    if suf in [".xlsx", ".xlsm", ".xls"]:
        try:
            df = pd.read_excel(path, engine="openpyxl")
            print(f"[INFO] read_excel ok: {path.name}, rows={len(df)}, cols={len(df.columns)}")
            return df
        except Exception as e:
            raise RuntimeError(f"read_excel failed: {e}")
    # CSV류: 인코딩/구분자 자동 추정
    encodings = ["utf-8","utf-8-sig","cp949","euc-kr","latin1"]
    seps = [",","\t",";","|"]
    last_err = None
    # csv.Sniffer 먼저
    try:
        with open(path, "rb") as f:
            sample = f.read(4096)
        for enc in encodings:
            try:
                txt = sample.decode(enc, errors="ignore")
                dialect = csv.Sniffer().sniff(txt, delimiters=",\t;|")
                sep_guess = dialect.delimiter
                df = pd.read_csv(path, encoding=enc, sep=sep_guess)
                print(f"[INFO] read_csv ok (enc={enc}, sep='{sep_guess}') rows={len(df)}")
                return df
            except Exception as e:
                last_err = e
    except Exception:
        pass
    # 전수 시도
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                print(f"[INFO] read_csv ok (enc={enc}, sep='{sep}') rows={len(df)}")
                return df
            except Exception as e:
                last_err = e
    raise RuntimeError(f"failed to load CSV: {last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/raw/fire_incidents_aligned.csv")
    args = ap.parse_args()

    df = _load_table(Path(args.input))

    out = pd.DataFrame()
    # 입력 매핑
    for k, cands in MAP_IN.items():
        out[k] = _pick(df, cands)
    # 출력 매핑
    for k, cands in MAP_OUT.items():
        out[k] = _pick(df, cands)

    # 숫자형 변환
    for c in ["building_agreement_count","total_floor_area","soot_area",
              "unit_temperature","unit_humidity","unit_wind_speed",
              "total_floor_count","casualty_count","property_damage_amount"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # 불리언/플래그 정규화 (0/1)
    for c in ["multi_use_flag","fire_management_target_flag","forest_fire_flag","vehicle_fire_flag"]:
        if c in out.columns:
            out[c] = out[c].map({True:1, False:0, "Y":1, "N":0, 1:1, 0:0}).fillna(0).astype(int)

    # 시간 파생: 분 단위 지연 (report → arrival / initial_extinguish)
    def to_dt(s):
        return pd.to_datetime(s, errors="coerce", utc=True)
    rep = to_dt(out.get("report_datetime"))
    arr = to_dt(out.get("arrival_datetime"))
    ext = to_dt(out.get("initial_extinguish_datetime"))
    out["arrival_delay_min"] = ((arr - rep).dt.total_seconds() / 60.0).astype("float")
    out["initial_extinguish_delay_min"] = ((ext - rep).dt.total_seconds() / 60.0).astype("float")

    # 저장
    Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")

    # 리포트
    print("[OK] wrote:", args.output, "rows=", len(out))
    print("[REPORT] class targets notnull:", {k:int(out[k].notna().sum()) for k in TARGET_CLASS if k in out.columns})
    print("[REPORT] reg targets notnull:", {k:int(out[k].notna().sum()) for k in TARGET_REG if k in out.columns})

if __name__ == "__main__":
    main()
