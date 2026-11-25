# Heart Disease Prediction Model

A machine learning model that estimates heart disease risk from 13 clinical measurements. Trained and evaluated on 302 unique patients from the heart dataset (1,025 rows, with 723 duplicate rows removed before any split).

**Live pages**

- [Model report](https://mhddrgr.github.io/heart-disease-report/): results, method, architecture, and code
- [Interactive predictor](https://mhddrgr.github.io/heart-disease-report/predict.html): enter a patient and see each model's risk estimate. Runs in the browser.

**Results**

| Evaluation | Accuracy | ROC AUC |
|---|---|---|
| Repeated 5-fold cross-validation (final ensemble) | 84.9% ± 1.1 | 0.909 |
| Separate 80/20 held-out test (61 patients) | 88.5% | 0.942 |

The model is for a class or research project. It is not a medical device.

## Files

| File | Purpose |
|---|---|
| `index.html` | The model report |
| `predict.html`, `heart_model.js` | Interactive browser predictor |
| `model.json` | The trained model in browser-readable form |
| `heart_model.joblib` | The trained model for Python |
| `train_heart.py` | Trains the model, runs both evaluations, writes `results.json` |
| `export_web_model.py` | Exports the model to `model.json` |
| `predict_heart.py` | Scores a CSV of patients with the Python model |
| `results.json` | All metrics and tuned settings |

## Run it in Python

Put `dataset.zip` (containing `heart.csv`) in the same folder, then:

```
python train_heart.py                       # train and evaluate
python export_web_model.py                  # write model.json for the browser
python predict_heart.py new_patients.csv    # score new patients
```

Requirements: Python 3.11, scikit-learn 1.8, pandas 2.3, numpy, joblib, PyTorch (optional CUDA GPU).
