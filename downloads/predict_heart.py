"""
Score patients with the trained heart disease model.

Usage:
    python predict_heart.py <input.csv> [output.csv]

The input CSV needs the same feature columns as heart.csv:
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
A "target" column is optional and ignored. The output keeps every input column and adds:
    status            - "ok", or the reason the row could not be scored
    risk_probability  - ensemble probability of heart disease (0 to 1), blank if not scored
    prediction        - 1 = heart disease predicted, 0 = no heart disease (cut-off 0.5), blank if not scored

Rows with missing, non-numeric, out-of-range, or unknown-code values are not scored;
their status column says what is wrong.
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_heart import CATEGORICAL, HERE, predict_proba

MODEL_FILE = HERE / "heart_model.joblib"
SKLEARN_VERSION_USED = "1.8.0"  # the version the model was trained with (see requirements.txt)

# Plausible ranges, matching the input limits on the web predictor (predict.html).
RANGES = {
    "age": (1, 120), "sex": (0, 1), "cp": (0, 3), "trestbps": (60, 260), "chol": (80, 700),
    "fbs": (0, 1), "restecg": (0, 2), "thalach": (50, 250), "exang": (0, 1),
    "oldpeak": (0, 10), "slope": (0, 2), "ca": (0, 4), "thal": (0, 3),
}


def load_model(path: Path = MODEL_FILE):
    import sklearn

    if sklearn.__version__ != SKLEARN_VERSION_USED:
        warnings.warn(
            f"Model was trained with scikit-learn {SKLEARN_VERSION_USED}, "
            f"but {sklearn.__version__} is installed. Predictions may differ slightly; "
            f"install the pinned version from requirements.txt."
        )
    return joblib.load(path)


def check_rows(artifact, df: pd.DataFrame) -> pd.Series:
    """Return a status per row: 'ok' or a description of the problem."""
    status = pd.Series("ok", index=df.index, dtype=object)
    missing = [c for c in artifact["features"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    cat_codes = artifact["pre"].named_transformers_["cat"].categories_
    known = {name: set(int(c) for c in codes) for name, codes in zip(CATEGORICAL, cat_codes)}

    for i, row in df.iterrows():
        problems = []
        for col in artifact["features"]:
            value = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(value):
                problems.append(f"{col} missing or not a number")
                continue
            lo, hi = RANGES[col]
            if not lo <= value <= hi:
                problems.append(f"{col}={value:g} outside {lo} to {hi}")
            elif col in known and int(value) not in known[col]:
                problems.append(f"{col}={int(value)} is not a known code")
        if problems:
            status.at[i] = "; ".join(problems)
    return status


def score_dataframe(artifact, df: pd.DataFrame) -> pd.DataFrame:
    """Return df with status, risk_probability and prediction columns added."""
    status = check_rows(artifact, df)
    ok = status == "ok"

    result = df.copy()
    result["status"] = status
    result["risk_probability"] = np.nan
    result["prediction"] = np.nan

    if ok.any():
        numeric = df.loc[ok, artifact["features"]].apply(pd.to_numeric)
        probs = predict_proba(artifact, numeric)["ensemble"]
        result.loc[ok, "risk_probability"] = np.round(probs, 4)
        result.loc[ok, "prediction"] = (probs >= 0.5).astype(int)
    result["prediction"] = result["prediction"].astype("Int64")  # whole numbers, blank when not scored
    return result


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else inp.with_name(inp.stem + "_predictions.csv")
    if not inp.exists():
        sys.exit(f"Input file not found: {inp}")

    artifact = load_model()
    try:
        df = pd.read_csv(inp)
        result = score_dataframe(artifact, df)
    except ValueError as err:
        sys.exit(str(err))

    result.to_csv(out, index=False)
    scored = int((result["status"] == "ok").sum())
    flagged = len(result) - scored
    print(f"Scored {scored} of {len(result)} patients -> {out}")
    if scored:
        print(f"Predicted heart disease: {int(result['prediction'].sum(skipna=True))} of {scored}")
    if flagged:
        print(f"{flagged} row(s) not scored. See the status column in {out}.")


if __name__ == "__main__":
    main()
