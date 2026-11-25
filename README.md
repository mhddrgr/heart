# Heart Disease Prediction Model

Estimates heart disease risk from 13 clinical measurements. Trained and tested on 302 unique patients.

**Pages**

- Summary (simple explanation): https://mhddrgr.github.io/heart/summary/
- Report (full details): https://mhddrgr.github.io/heart/report/
- Predict (try the model in your browser): https://mhddrgr.github.io/heart/predict/
- Download (Python model, scripts, tests): https://mhddrgr.github.io/heart/download/

**Results**

| Evaluation | Accuracy | ROC AUC |
|---|---|---|
| Repeated 5-fold cross-validation (final ensemble) | 84.9% Â± 1.1 | 0.909 |
| Separate 80/20 held-out test (61 patients) | 88.5% | 0.942 |

This is a student project. It is not a medical tool.

## Run in Python

```
pip install -r requirements.txt
python predict_heart.py new_patients.csv
python -m unittest test_heart -v
```
