"""
Score patients with the trained heart disease model.

Usage:
    python predict_heart.py <input.csv> [output.csv]

The input CSV needs the same feature columns as heart.csv (age, sex, cp, trestbps,
chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal). A "target" column
is optional and ignored. The output adds:
    risk_probability  - ensemble probability of heart disease (0 to 1)
    prediction        - 1 = heart disease predicted, 0 = no heart disease (threshold 0.5)
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

from train_heart import HERE, predict_proba


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else inp.with_name(inp.stem + "_predictions.csv")

    artifact = joblib.load(HERE / "heart_model.joblib")
    df = pd.read_csv(inp)

    missing = [c for c in artifact["features"] if c not in df.columns]
    if missing:
        sys.exit(f"Input is missing required columns: {missing}")

    probs = predict_proba(artifact, df)
    result = df.copy()
    result["risk_probability"] = probs["ensemble"].round(4)
    result["prediction"] = (probs["ensemble"] >= 0.5).astype(int)
    result.to_csv(out, index=False)

    print(f"Scored {len(result)} patients -> {out}")
    print(f"Predicted heart disease: {int(result['prediction'].sum())} of {len(result)}")


if __name__ == "__main__":
    main()
