# Heart Disease Risk Prediction Dashboard

An interactive Streamlit application that predicts heart disease risk from clinical measurements, compares a Random Forest and an XGBoost model, and explains each prediction with SHAP.

This project is for educational purposes only and is not a medical device or a substitute for professional medical advice.

## Features

| Tab | What it shows |
| --- | --- |
| Risk Assessment | Patient inputs in the sidebar, predicted risk gauge for each model, and a radar chart comparing the patient to the typical range |
| Explainability | SHAP waterfall for the current patient and global feature importance |
| Model Performance | ROC curves, confusion matrices, accuracy, ROC AUC, and cross-validation scores for both models |

## Data

The models are trained on the UCI Heart Disease (Cleveland) dataset (UCI repository id 45), downloaded with `ucimlrepo`. The target is binarized: any value above 0 counts as heart disease present.

## Setup

```bash
git clone https://github.com/dellis5/heart-disease-dashboard.git
cd heart-disease-dashboard
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

`train_model.py` downloads the dataset, trains both models, and writes the artifacts to `models/` (not tracked by git). The app reads from that folder, so run the training script before launching the app.

If you have no internet access, `python train_model.py --demo-data` trains on a synthetic dataset so you can try the app. The app shows a warning banner when demo data is in use, and its metrics should not be taken as real results.

## Project layout

```
app.py              Streamlit dashboard
train_model.py      Data loading, preprocessing, training, evaluation
requirements.txt    Python dependencies
.streamlit/         Dark theme configuration
```

## Notes

The test split is saved unscaled and scaled at inference time, so SHAP values and the displayed metrics are computed on the same inputs the models were trained on. Accuracy figures are printed by `train_model.py` and shown in the Model Performance tab.
