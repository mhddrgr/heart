"""
Heart disease classifier for dataset.zip (heart.csv).

Evaluation protocol (chosen to avoid leakage):
- heart.csv has 1,025 rows but only 302 unique patients (723 exact duplicate
  rows). Exact duplicates are removed BEFORE any split, so no patient can be
  in both the training and the test data.
- Outer evaluation: 5-fold stratified CV repeated 3 times. Every repeat puts
  each unique patient in a test fold exactly once.
- Hyperparameters are tuned only inside each outer training fold (nested CV),
  so test labels never influence tuning.
- Ensemble = equal-weight average of all model probabilities. It is fixed in
  advance, not picked by its test score.

Models: logistic regression, SVM, random forest, histogram gradient boosting,
and a small PyTorch MLP (5 seeds, trained on the GPU when one is available).

Usage:
    python train_heart.py                 # evaluate + train final model
    python train_heart.py --repeats 5     # more repeats for tighter estimates
"""

import argparse
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
TARGET = "target"
CATEGORICAL = ["cp", "restecg", "slope", "ca", "thal"]
NUMERIC = ["age", "sex", "trestbps", "chol", "fbs", "thalach", "exang", "oldpeak"]


# ---------- data ----------

def load_data(zip_path: Path):
    """Read heart.csv straight from the zip (no extraction), drop exact duplicates."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("heart.csv") as fh:
            df = pd.read_csv(fh)
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    X = df.drop(columns=TARGET)
    y = df[TARGET].to_numpy().astype(int)
    return X, y, n_raw


def make_pre():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


# ---------- scikit-learn models ----------

def sk_specs(seed):
    def pipe(clf):
        return Pipeline([("pre", make_pre()), ("clf", clf)])

    return {
        "logreg": (
            pipe(LogisticRegression(max_iter=5000)),
            {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
        ),
        "svm": (
            pipe(SVC(probability=True, random_state=seed)),
            {"clf__C": [0.1, 0.3, 1, 3, 10], "clf__gamma": ["scale", 0.01, 0.03, 0.1]},
        ),
        "random_forest": (
            pipe(RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)),
            {
                "clf__max_depth": [None, 4, 8],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_features": ["sqrt", 0.5],
            },
        ),
        "hist_gb": (
            pipe(HistGradientBoostingClassifier(random_state=seed)),
            {
                "clf__learning_rate": [0.03, 0.1],
                "clf__max_depth": [2, 3, None],
                "clf__l2_regularization": [0.0, 1.0],
            },
        ),
    }


def fit_tuned(est, grid, X, y, seed):
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    gs = GridSearchCV(est, grid, cv=inner, scoring="roc_auc", n_jobs=-1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


# ---------- PyTorch MLP (GPU if available) ----------

class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp(Z, y, seed, device, epochs=400):
    torch.manual_seed(seed)
    model = MLP(Z.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    zt = torch.tensor(Z, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):  # full-batch training: the dataset is tiny
        opt.zero_grad(set_to_none=True)
        loss_fn(model(zt), yt).backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def mlp_proba(models, Z, device):
    zt = torch.tensor(Z, dtype=torch.float32, device=device)
    return np.mean([torch.sigmoid(m.to(device)(zt)).cpu().numpy() for m in models], axis=0)


MLP_SEEDS = range(5)


# ---------- evaluation ----------

def evaluate(X, y, device, seed=0, n_repeats=3):
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    specs = sk_specs(seed)
    names = list(specs) + ["mlp"]
    oof = {n: np.zeros((n_repeats, len(y))) for n in names}
    chosen = []

    for i, (tr, te) in enumerate(splitter.split(X, y)):
        rep = i // 5
        X_tr, X_te, y_tr = X.iloc[tr], X.iloc[te], y[tr]

        for name, (est, grid) in specs.items():
            best, params = fit_tuned(est, grid, X_tr, y_tr, seed)
            oof[name][rep, te] = best.predict_proba(X_te)[:, 1]
            chosen.append({"fold": i, "model": name, "params": params})

        pre = make_pre().fit(X_tr)
        Z_tr = pre.transform(X_tr).astype(np.float32)
        Z_te = pre.transform(X_te).astype(np.float32)
        mlps = [train_mlp(Z_tr, y_tr, s, device) for s in MLP_SEEDS]
        oof["mlp"][rep, te] = mlp_proba(mlps, Z_te, device)

        print(f"  outer fold {i + 1}/{5 * n_repeats} done", flush=True)

    oof["ensemble"] = np.mean([oof[n] for n in names], axis=0)
    return oof, chosen


def summarize(oof, y, n_repeats):
    out = {}
    for name, p in oof.items():
        accs = [accuracy_score(y, (p[r] >= 0.5).astype(int)) for r in range(n_repeats)]
        aucs = [roc_auc_score(y, p[r]) for r in range(n_repeats)]
        out[name] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
        }
    return out


# ---------- final model ----------

def fit_final(X, y, device, seed=0):
    sk_models, params = {}, {}
    for name, (est, grid) in sk_specs(seed).items():
        best, p = fit_tuned(est, grid, X, y, seed)
        sk_models[name] = best
        params[name] = p

    pre = make_pre().fit(X)
    Z = pre.transform(X).astype(np.float32)
    mlps = [train_mlp(Z, y, s, device) for s in MLP_SEEDS]
    mlp_states = [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in mlps]

    return {
        "features": list(X.columns),
        "sk_models": sk_models,
        "pre": pre,
        "mlp_states": mlp_states,
        "mlp_n_in": Z.shape[1],
        "params": params,
    }


def predict_proba(artifact, df: pd.DataFrame):
    """Return {model_name: P(heart disease)} for each row of df (CPU inference)."""
    X = df[artifact["features"]]
    probs = {n: m.predict_proba(X)[:, 1] for n, m in artifact["sk_models"].items()}

    Z = artifact["pre"].transform(X).astype(np.float32)
    mlps = []
    for state in artifact["mlp_states"]:
        m = MLP(artifact["mlp_n_in"])
        m.load_state_dict(state)
        mlps.append(m.eval())
    probs["mlp"] = mlp_proba(mlps, Z, "cpu")

    probs["ensemble"] = np.mean([probs[n] for n in list(artifact["sk_models"]) + ["mlp"]], axis=0)
    return probs


def holdout_eval(X, y, device, seed=0, test_size=0.2):
    """Train on 80% of unique patients (tuning included), score once on the untouched 20%."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    artifact = fit_final(X_tr, y_tr, device, seed)
    probs = predict_proba(artifact, X_te)

    results = {}
    for name, p in probs.items():
        pred = (p >= 0.5).astype(int)
        results[name] = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "auc": float(roc_auc_score(y_te, p)),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),  # [[TN, FP], [FN, TP]]
        }
    return {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "results": results,
    }, artifact


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "dataset.zip")
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {device} ({dev_name})")

    X, y, n_raw = load_data(args.data)
    print(f"Raw rows: {n_raw} | unique patients used: {len(y)} | positives: {int(y.sum())}")

    print("Evaluating (nested CV, duplicate-free):")
    oof, chosen = evaluate(X, y, device, n_repeats=args.repeats)
    summary = summarize(oof, y, args.repeats)

    print("\nHeld-out results (mean ± std over repeats):")
    for name, s in summary.items():
        print(f"  {name:14s} acc {s['accuracy_mean'] * 100:6.2f}% ± {s['accuracy_std'] * 100:4.2f}"
              f"   AUC {s['auc_mean']:.3f} ± {s['auc_std']:.3f}")

    print("\nHold-out test (stratified 80/20 split; test patients never used for training or tuning):")
    holdout, _ = holdout_eval(X, y, device)
    print(f"  train patients: {holdout['n_train']} | test patients: {holdout['n_test']}")
    for name, r in holdout["results"].items():
        print(f"  {name:14s} acc {r['accuracy'] * 100:6.2f}%   AUC {r['auc']:.3f}")

    cm_cv = confusion_matrix(y, (oof["ensemble"][0] >= 0.5).astype(int)).tolist()

    print("\nTraining final model on all unique patients...")
    artifact = fit_final(X, y, device)
    print("Tuned hyperparameters (final fit):")
    for name, p in artifact["params"].items():
        print(f"  {name}: {p}")

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out / "heart_model.joblib")
    (args.out / "results.json").write_text(json.dumps({
        "raw_rows": n_raw,
        "unique_patients": int(len(y)),
        "repeats": args.repeats,
        "summary": summary,
        "confusion_matrix_ensemble_cv": cm_cv,
        "holdout": holdout,
        "final_params": artifact["params"],
        "outer_fold_params": chosen,
    }, indent=2, default=str))
    print(f"\nSaved: {args.out / 'heart_model.joblib'}")
    print(f"Saved: {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
