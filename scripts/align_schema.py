import argparse, pandas as pd, numpy as np, os
from pathlib import Path

LABEL_MAP = {
    "ignition_nm": ["ignition_cause","fire_cause","cause"],
    "ignition_structure_nm": ["ignition_material","special_fire_object_name","ignition_object"]
}
FEATURE_MAP = {
    "fire_type_nm": ["fire_type","fire_type_name"],
    "bldg_frame_nm": ["building_structure","building_frame_name"],
    "bldg_structure_nm": ["building_structure"],
    "bldg_inside_nm": ["special_fire_object_name","facility_location"],
    "power_source_nm": ["fuel_type","power_type"],
    "ignition_start_floor": ["ignition_floor"],
    "hr_unit_temp": ["unit_temperature"],
    "dispatch_time": ["dispatch_time"],
    "prpt_dam_amount": ["property_damage_amount"]
}

def pick(df,cands):
    for c in cands:
        if c in df.columns:
            return df[c]
    return pd.Series([np.nan]*len(df))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/raw/fire_incidents_aligned.csv")
    args = ap.parse_args()

    df=None
    for enc in ["utf-8","cp949","utf-8-sig"]:
        try:
            df=pd.read_csv(args.input,encoding=enc)
            print("[INFO] loaded with",enc,"rows=",len(df))
            break
        except: continue
    if df is None: raise RuntimeError("fail load")

    out=pd.DataFrame()
    for k,v in LABEL_MAP.items(): out[k]=pick(df,v)
    for k,v in FEATURE_MAP.items(): out[k]=pick(df,v)

    if "ground_high" in df.columns and "ground_low" in df.columns:
        out["ground_high"]=pd.to_numeric(df["ground_high"],errors="coerce")
        out["ground_low"]=pd.to_numeric(df["ground_low"],errors="coerce")
    else:
        out["ground_high"]=np.nan; out["ground_low"]=np.nan

    # keep some meta if exists
    for keep in ["occur_datetime","report_datetime","arrival_datetime"]:
        if keep in df.columns: out[keep]=df[keep]

    Path(os.path.dirname(args.output)).mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output,index=False,encoding="utf-8")
    print("[OK] wrote",args.output)
    print("ignition_nm notnull=",out["ignition_nm"].notna().sum())
    print("ignition_structure_nm notnull=",out["ignition_structure_nm"].notna().sum())

if __name__=="__main__": main()
