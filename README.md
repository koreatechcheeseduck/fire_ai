# Fire ML Project (v2)

## 0) 가상환경 & 패키지
```bash
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 1) 데이터 스키마 정렬
```bash
python scripts/align_schema.py --input data/raw/학습데이터.csv --output data/raw/fire_incidents_aligned.csv
```

## 2) 모델 학습 (원인/발화 지점 멀티헤드 RandomForest)
```bash
python -m src.train_model --input data/raw/fire_incidents_aligned.csv --outdir models/rf_v1
```

## 3) 유사사례 인덱스 생성 (TF-IDF)
```bash
python scripts/build_index.py --input data/processed/dataset.parquet --outdir models/index_v1
```

## 4) 테스트 실행
```bash
python scripts/run_local.py --pretty
```
