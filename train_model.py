"""Train Random Forest and XGBoost classifiers on the UCI Heart Disease dataset.

Models, the fitted scaler, metadata and the held-out test split are written to
the models/ directory. Run this once before launching the dashboard:

    python train_model.py

The real UCI dataset is downloaded by default. If you are offline you can use
`python train_model.py --demo-data` to train on synthetic data instead. Demo
data exists only so the dashboard can be tested without a connection; its
metrics are meaningless and the dashboard shows a warning when it is in use.
"""

import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex (1=M, 0=F)",
    "cp": "Chest Pain Type (0-3)",
    "trestbps": "Resting BP (mm Hg)",
    "chol": "Serum Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar >120 (1=T)",
    "restecg": "Resting ECG Result (0-2)",
    "thalach": "Max Heart Rate Achieved",
    "exang": "Exercise Induced Angina (1=Y)",
    "oldpeak": "ST Depression (exercise vs rest)",
    "slope": "Slope of Peak Exercise ST (0-2)",
    "ca": "# Major Vessels Colored (0-3)",
    "thal": "Thal (1=Normal, 2=Fixed, 3=Reversible)",
}

# (min, max, default) used for the dashboard sliders
FEATURE_RANGES = {
    "age": (29, 77, 54),
    "sex": (0, 1, 1),
    "cp": (0, 3, 0),
    "trestbps": (94, 200, 130),
    "chol": (126, 564, 246),
    "fbs": (0, 1, 0),
    "restecg": (0, 2, 0),
    "thalach": (71, 202, 150),
    "exang": (0, 1, 0),
    "oldpeak": (0.0, 6.2, 1.0),
    "slope": (0, 2, 1),
    "ca": (0, 3, 0),
    "thal": (1, 3, 2),
}


def loadUciData():
    """Download the UCI Heart Disease dataset (Cleveland subset)."""
    try:
        from ucimlrepo import fetch_ucirepo
        heart = fetch_ucirepo(id=45)
    except Exception as error:
        raise SystemExit(
            f"Could not download the UCI dataset ({error}).\n"
            "Check your connection, or run with --demo-data to test the dashboard "
            "on synthetic data."
        )
    df = pd.concat([heart.data.features, heart.data.targets], axis=1)
    df.columns = FEATURE_NAMES + ["target"]
    print(f"[INFO] Loaded UCI Heart Disease dataset: {len(df)} rows")
    return df


def makeDemoData(rowCount=800):
    """Reproducible synthetic data with loose correlations. For testing only."""
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "age": rng.integers(29, 78, rowCount),
        "sex": rng.integers(0, 2, rowCount),
        "cp": rng.integers(0, 4, rowCount),
        "trestbps": rng.integers(94, 201, rowCount),
        "chol": rng.integers(126, 565, rowCount),
        "fbs": rng.integers(0, 2, rowCount),
        "restecg": rng.integers(0, 3, rowCount),
        "thalach": rng.integers(71, 203, rowCount),
        "exang": rng.integers(0, 2, rowCount),
        "oldpeak": np.round(rng.uniform(0, 6.2, rowCount), 1),
        "slope": rng.integers(0, 3, rowCount),
        "ca": rng.integers(0, 4, rowCount),
        "thal": rng.integers(1, 4, rowCount),
    })
    score = (
        0.03 * df["age"] + 0.5 * df["sex"] - 0.4 * df["cp"]
        + 0.01 * df["trestbps"] + 0.003 * df["chol"] + 0.2 * df["fbs"]
        - 0.01 * df["thalach"] + 0.5 * df["exang"] + 0.3 * df["oldpeak"]
        + 0.4 * df["ca"] + 0.3 * df["thal"]
    )
    probability = 1 / (1 + np.exp(-score + 3))
    df["target"] = (rng.uniform(0, 1, rowCount) < probability).astype(int)
    print("[WARN] Using synthetic DEMO data. Metrics are not meaningful.")
    return df


def preprocess(df):
    """Split, then fit the scaler on the training portion only.

    Returns scaled train/test arrays, the unscaled test features (saved for the
    dashboard, which applies the scaler itself), labels, and the fitted scaler.
    """
    df = df.dropna().copy()
    df["target"] = (df["target"] > 0).astype(int)  # 0 = no disease, 1 = disease

    features = df[FEATURE_NAMES]
    target = df["target"]
    xTrain, xTest, yTrain, yTest = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    xTestScaled = scaler.transform(xTest)
    return xTrainScaled, xTestScaled, xTest, yTrain, yTest, scaler


def trainRandomForest(xTrain, yTrain):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(xTrain, yTrain)
    return model


def trainXgboost(xTrain, yTrain):
    positiveWeight = float((yTrain == 0).sum()) / float((yTrain == 1).sum())
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=positiveWeight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(xTrain, yTrain)
    return model


def evaluate(name, model, xTest, yTest, xTrain, yTrain):
    predictions = model.predict(xTest)
    probabilities = model.predict_proba(xTest)[:, 1]

    accuracy = accuracy_score(yTest, predictions)
    rocAuc = roc_auc_score(yTest, probabilities)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cvScores = cross_val_score(model, xTrain, yTrain, cv=folds, scoring="roc_auc", n_jobs=-1)

    print(f"\n{'=' * 50}\n  {name}\n{'=' * 50}")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  ROC-AUC  : {rocAuc:.4f}")
    print(f"  CV AUC   : {cvScores.mean():.4f} +/- {cvScores.std():.4f}")
    print(classification_report(yTest, predictions, target_names=["No Disease", "Disease"]))
    return {
        "accuracy": accuracy,
        "rocAuc": rocAuc,
        "cvMean": cvScores.mean(),
        "cvStd": cvScores.std(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the heart disease models.")
    parser.add_argument("--demo-data", action="store_true",
                        help="use synthetic data instead of downloading the UCI dataset")
    args = parser.parse_args()

    print("\nHeart Disease Risk: Model Training Pipeline")
    print("=" * 50)

    df = makeDemoData() if args.demo_data else loadUciData()
    xTrain, xTest, xTestRaw, yTrain, yTest, scaler = preprocess(df)

    print("\n[1/4] Training Random Forest ...")
    randomForest = trainRandomForest(xTrain, yTrain)
    print("[2/4] Training XGBoost ...")
    xgboost = trainXgboost(xTrain, yTrain)

    print("\n[3/4] Evaluating models ...")
    rfMetrics = evaluate("Random Forest", randomForest, xTest, yTest, xTrain, yTrain)
    xgbMetrics = evaluate("XGBoost", xgboost, xTest, yTest, xTrain, yTrain)

    print("\n[4/4] Saving artifacts ...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(randomForest, f"{MODELS_DIR}/random_forest.pkl")
    joblib.dump(xgboost, f"{MODELS_DIR}/xgboost.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    joblib.dump({
        "featureNames": FEATURE_NAMES,
        "featureLabels": FEATURE_LABELS,
        "featureRanges": FEATURE_RANGES,
        "rfMetrics": rfMetrics,
        "xgbMetrics": xgbMetrics,
        "dataSource": "demo" if args.demo_data else "uci",
    }, f"{MODELS_DIR}/metadata.pkl")

    # Unscaled on purpose: the dashboard applies the scaler when it reads this file.
    testSplit = xTestRaw.copy()
    testSplit["target"] = yTest.values
    testSplit.to_csv(f"{MODELS_DIR}/test_split.csv", index=False)

    print(f"\nArtifacts saved to ./{MODELS_DIR}/")
    print("Next: streamlit run app.py")


if __name__ == "__main__":
    main()
