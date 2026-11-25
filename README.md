# Heart Disease Prediction Model

A machine learning model that estimates heart disease risk from 13 clinical measurements, trained and tested on 302 unique patients.

**Pages**

- Home: https://mhddrgr.github.io/heart/
- Model report: https://mhddrgr.github.io/heart/report/
- Interactive predictor: https://mhddrgr.github.io/heart/predict/
- Downloads (Python model and scripts): https://mhddrgr.github.io/heart/downloads/

**Results**

| Evaluation | Accuracy | ROC AUC |
|---|---|---|
| Repeated 5-fold cross-validation (final ensemble) | 84.9% ± 1.1 | 0.909 |
| Separate 80/20 held-out test (61 patients) | 88.5% | 0.942 |

This is a student project. It is not a medical tool.

## Run in Python

Put `dataset.zip` (containing `heart.csv`) in the same folder as the scripts, then:

```
python train_heart.py                       # train and evaluate
python export_web_model.py                  # write model.json for the browser
python predict_heart.py new_patients.csv    # score new patients
```

Requirements: Python 3.11, scikit-learn 1.8, pandas 2.3, numpy, joblib, PyTorch (optional CUDA GPU).
