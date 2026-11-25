"""
Tests for the heart disease model. Run with:
    python -m unittest test_heart -v

They use only the Python standard library plus the project's own code.
"""

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from predict_heart import check_rows, load_model, score_dataframe
from train_heart import HERE, NUMERIC, CATEGORICAL, load_data, predict_proba

FEATURES = NUMERIC + CATEGORICAL
EXAMPLE = {
    "age": 58, "sex": 0, "cp": 0, "trestbps": 100, "chol": 248, "fbs": 0, "restecg": 0,
    "thalach": 122, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2,
}


class HeartModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.artifact = load_model()
        cls.X, cls.y, cls.n_raw = load_data(HERE / "dataset.zip")

    def test_data_has_302_unique_patients(self):
        self.assertEqual(len(self.y), 302)
        self.assertEqual(self.n_raw, 1025)
        self.assertFalse(self.X.duplicated().any())

    def test_artifact_features_match_dataset(self):
        self.assertEqual(self.artifact["features"], list(self.X.columns))

    def test_probabilities_are_valid_and_ensemble_is_average(self):
        probs = predict_proba(self.artifact, self.X.head(20))
        for name in ["logreg", "svm", "random_forest", "hist_gb", "mlp", "ensemble"]:
            self.assertTrue(np.all((probs[name] >= 0) & (probs[name] <= 1)), name)
        models = ["logreg", "svm", "random_forest", "hist_gb", "mlp"]
        mean = np.mean([probs[m] for m in models], axis=0)
        np.testing.assert_allclose(probs["ensemble"], mean, rtol=1e-9)

    def test_example_patient_scores_as_disease(self):
        row = pd.DataFrame([EXAMPLE])
        result = score_dataframe(self.artifact, row)
        self.assertEqual(result.loc[0, "status"], "ok")
        self.assertEqual(int(result.loc[0, "prediction"]), 1)

    def test_bad_rows_are_flagged_not_scored(self):
        bad = pd.DataFrame([
            EXAMPLE,
            {**EXAMPLE, "age": 300},            # out of range
            {**EXAMPLE, "chol": None},          # missing
            {**EXAMPLE, "cp": 9},               # unknown code
            {**EXAMPLE, "trestbps": "high"},    # not a number
        ])
        result = score_dataframe(self.artifact, bad)
        self.assertEqual(list(result["status"].str.startswith("ok")), [True, False, False, False, False])
        self.assertTrue(result["risk_probability"].iloc[1:].isna().all())
        self.assertIn("age=300", result.loc[1, "status"])
        self.assertIn("chol missing", result.loc[2, "status"])
        self.assertIn("cp=9 outside 0 to 3", result.loc[3, "status"])

    def test_missing_column_is_an_error(self):
        df = pd.DataFrame([EXAMPLE]).drop(columns=["thal"])
        with self.assertRaises(ValueError):
            check_rows(self.artifact, df)

    def test_cli_writes_predictions_file(self):
        import predict_heart

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "patients.csv"
            out = Path(tmp) / "scored.csv"
            pd.DataFrame([EXAMPLE]).to_csv(src, index=False)
            import sys
            old_argv = sys.argv
            sys.argv = ["predict_heart.py", str(src), str(out)]
            try:
                predict_heart.main()
            finally:
                sys.argv = old_argv
            scored = pd.read_csv(out)
            self.assertIn("risk_probability", scored.columns)
            self.assertEqual(int(scored.loc[0, "prediction"]), 1)


if __name__ == "__main__":
    unittest.main()
