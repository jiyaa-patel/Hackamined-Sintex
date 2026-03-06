# app/ml_service.py
import joblib
import numpy as np
import shap

class MLService:
    def __init__(self, model_path: str):
        bundle = joblib.load(model_path)
        self.pipe = bundle["pipeline"]
        self.schema = bundle["schema"]
        self.feature_names = bundle["feature_names"]

        # SHAP explainer for tree model inside the pipeline
        rf = self.pipe.named_steps["model"]
        self.explainer = shap.TreeExplainer(rf)

    def predict_one(self, features: dict):
        """
        Contract:
          input: dict of {feature_name: value_or_null}
          output: (risk_score float 0..1, top_factors list[dict])
        """
        # Order features
        X_row = [[features.get(f) for f in self.feature_names]]

        # preprocess then predict
        X_proc = self.pipe.named_steps["preproc"].transform(
            {f: [features.get(f)] for f in self.feature_names}
        )
        risk = float(self.pipe.named_steps["model"].predict_proba(X_proc)[:, 1][0])

        shap_vals = self.explainer.shap_values(X_proc)[1][0]  # class 1
        vals = np.array([features.get(f) for f in self.feature_names], dtype=object)

        top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
        top = []
        for j in top_idx:
            top.append({
                "feature": self.feature_names[j],
                "value": None if vals[j] is None else float(vals[j]),
                "shap": float(shap_vals[j])
            })
        return risk, top