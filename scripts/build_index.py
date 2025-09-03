import argparse, pandas as pd, joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--outdir",default="models/index_v1")
    args=ap.parse_args()
    Path(args.outdir).mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(args.input)
    txt=(df["fire_type_nm"].fillna("")+" "+df["bldg_frame_nm"].fillna("")+" "+df["bldg_structure_nm"].fillna("")+" "+df["bldg_inside_nm"].fillna("")+" "+df["ignition_nm"].fillna("")+" "+df["ignition_structure_nm"].fillna("")).str.strip()
    vec=TfidfVectorizer(min_df=3,ngram_range=(1,2))
    X=vec.fit_transform(txt.values.astype(str))
    # keep minimal meta
    keep_cols=[c for c in ["occur_datetime","fire_type_nm","ignition_nm","ignition_structure_nm"] if c in df.columns]
    df[keep_cols].to_parquet(f"{args.outdir}/meta.parquet",index=False)
    joblib.dump(vec,f"{args.outdir}/tfidf_vectorizer.joblib")
    joblib.dump(X,f"{args.outdir}/tfidf_matrix.joblib")
    print("[OK] index built at",args.outdir)
if __name__=="__main__": main()
