"""
Export the trained heart model to web/model.json so it can run in a web browser.

The browser predictor (web/heart_model.js) rebuilds the same ensemble as
predict_proba() in train_heart.py: scaling and one-hot encoding, logistic
regression, RBF SVM with Platt scaling, random forest, histogram gradient
boosting, and the 5-seed neural network, averaged with equal weights.

Usage (after train_heart.py has written heart_model.joblib):
    python export_web_model.py [reference_predictions.json]

The optional second argument writes Python predictions for every unique
patient. web/heart_model.js is checked against these numbers.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np

from train_heart import HERE, CATEGORICAL, NUMERIC, load_data, predict_proba


def tree_to_dict(sk_tree):
    t = sk_tree.tree_
    value = t.value[:, 0, :]
    value = value / value.sum(axis=1, keepdims=True)  # class fractions per node
    return {
        "l": t.children_left.tolist(),
        "r": t.children_right.tolist(),
        "f": t.feature.tolist(),
        "t": t.threshold.tolist(),
        "p": value[:, 1].tolist(),  # probability of class 1 (heart disease)
    }


def export_random_forest(clf):
    return {"trees": [tree_to_dict(est) for est in clf.estimators_]}


def export_hist_gb(clf):
    trees = []
    for stage in clf._predictors:
        nodes = stage[0].nodes
        trees.append({
            "leaf": nodes["is_leaf"].astype(int).tolist(),
            "f": nodes["feature_idx"].astype(int).tolist(),
            "t": nodes["num_threshold"].astype(float).tolist(),
            "l": nodes["left"].astype(int).tolist(),
            "r": nodes["right"].astype(int).tolist(),
            "v": nodes["value"].astype(float).tolist(),
            "ml": nodes["missing_go_to_left"].astype(int).tolist(),
        })
    return {"baseline": float(clf._baseline_prediction.ravel()[0]), "trees": trees}


def export_svm(clf):
    return {
        "sv": clf.support_vectors_.tolist(),
        "dual": clf.dual_coef_[0].tolist(),
        "b": float(clf.intercept_[0]),
        "gamma": float(clf._gamma),
        "A": float(clf.probA_[0]),
        "B": float(clf.probB_[0]),
    }


def export_logreg(clf):
    return {"coef": clf.coef_[0].tolist(), "b": float(clf.intercept_[0])}


def export_mlp(states):
    seeds = []
    for st in states:
        layers = []
        for idx in ("0", "3", "6"):  # Linear layers in nn.Sequential
            layers.append({
                "W": st[f"net.{idx}.weight"].numpy().tolist(),
                "b": st[f"net.{idx}.bias"].numpy().tolist(),
            })
        seeds.append(layers)
    return {"seeds": seeds}


def main():
    artifact = joblib.load(HERE / "heart_model.joblib")
    pre = artifact["pre"]
    num = pre.named_transformers_["num"]
    cat = pre.named_transformers_["cat"]
    sk = artifact["sk_models"]

    model = {
        "version": 1,
        "features": artifact["features"],
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "scaler_mean": num.mean_.tolist(),
        "scaler_scale": num.scale_.tolist(),
        "categories": {name: [int(c) for c in cats]
                       for name, cats in zip(CATEGORICAL, cat.categories_)},
        "logreg": export_logreg(sk["logreg"].named_steps["clf"]),
        "svm": export_svm(sk["svm"].named_steps["clf"]),
        "random_forest": export_random_forest(sk["random_forest"].named_steps["clf"]),
        "hist_gb": export_hist_gb(sk["hist_gb"].named_steps["clf"]),
        "mlp": export_mlp(artifact["mlp_states"]),
    }

    out = HERE / "web"
    out.mkdir(exist_ok=True)
    (out / "model.json").write_text(json.dumps(model), encoding="utf-8")
    print(f"Wrote {out / 'model.json'}")

    if len(sys.argv) > 1:
        X, y, _ = load_data(HERE / "dataset.zip")
        probs = predict_proba(artifact, X)
        ref = {
            "rows": X.to_dict(orient="records"),
            "probs": {k: np.asarray(v, dtype=float).tolist() for k, v in probs.items()},
        }
        Path(sys.argv[1]).write_text(json.dumps(ref), encoding="utf-8")
        print(f"Wrote reference predictions for {len(y)} patients: {sys.argv[1]}")


if __name__ == "__main__":
    main()
