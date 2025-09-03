import glob, joblib, numpy as np, pandas as pd
from .train_model import engineer_features

class EnsembleRF:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.cause_paths = sorted(glob.glob(f"{model_dir}/rf_cause_fold*.joblib"))
        self.ign_paths   = sorted(glob.glob(f"{model_dir}/rf_ignition_fold*.joblib"))
        self.le_cause = None
        self.le_ign = None
        try:
            self.le_cause = joblib.load(f"{model_dir}/label_encoder_cause.joblib")
        except: pass
        try:
            self.le_ign = joblib.load(f"{model_dir}/label_encoder_ign.joblib")
        except: pass
        self.cause_models = [joblib.load(p) for p in self.cause_paths]
        self.ign_models = [joblib.load(p) for p in self.ign_paths]

    def _avg_proba(self, models, X):
        if not models:
            return None, None
        arr = []
        for m in models:
            p = m.predict_proba(X)
            if p.ndim == 1:
                p = np.vstack([1-p, p]).T
            arr.append(p)
        arr = np.stack(arr, axis=0)  # (n_models, n_samples, n_classes)
        return arr.mean(axis=0), arr.std(axis=0)

    def predict(self, df: pd.DataFrame):
        df = engineer_features(df)
        out = []
        # cause
        mu_c, std_c = self._avg_proba(self.cause_models, df)
        if mu_c is not None and self.le_cause is not None:
            idx_c = mu_c.argmax(axis=1)
            conf_c = 1 - np.clip(std_c[range(len(df)), idx_c]/0.8, 0, 1)
            cause = self.le_cause.inverse_transform(idx_c)
        else:
            cause = np.array(["정보부족"]*len(df)); conf_c = np.zeros(len(df))

        # ignition
        mu_i, std_i = self._avg_proba(self.ign_models, df)
        if mu_i is not None and self.le_ign is not None:
            idx_i = mu_i.argmax(axis=1)
            conf_i = 1 - np.clip(std_i[range(len(df)), idx_i]/0.8, 0, 1)
            ign = self.le_ign.inverse_transform(idx_i)
        else:
            ign = np.array(["정보부족"]*len(df)); conf_i = np.zeros(len(df))

        for i in range(len(df)):
            out.append(dict(
                cause=str(cause[i]), cause_confidence=float(conf_c[i]),
                ignition=str(ign[i]), ignition_confidence=float(conf_i[i])
            ))
        return out
