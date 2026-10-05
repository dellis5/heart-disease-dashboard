"""CardioRisk: Heart Disease Risk Prediction Dashboard.

Run with:   streamlit run app.py
Requires:   python train_model.py   (run once first)
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CardioRisk | Heart Disease Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #151820;
    --card: #1c2030;
    --border: #2a3040;
    --accent: #e05c5c;
    --text: #e8eaf0;
    --muted: #7a8299;
}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main .block-container {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3 { font-family: 'DM Serif Display', serif; color: var(--text); }

[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetric"] label { color: var(--muted) !important; font-size: 0.8rem; }
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.8rem !important;
    font-weight: 600 !important;
}

/* Tabs: selectors cover both the older (baseweb) and newer (role) Streamlit markup. */
.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {
    background: var(--surface);
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
    display: flex;
    width: 100%;
}
.stTabs [data-baseweb="tab"], .stTabs [role="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-weight: 500;
    flex: 1 1 0 !important;
    text-align: center !important;
    justify-content: center !important;
    font-size: 1rem !important;
    padding: 0.6rem 0 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"], .stTabs [role="tab"][aria-selected="true"] {
    background: var(--accent) !important;
    color: #fff !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

.risk-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
}
.risk-low    { background: rgba(92,224,176,0.15); color: #5ce0b0; border: 1px solid #5ce0b0; }
.risk-medium { background: rgba(224,200,92,0.15); color: #e0c85c; border: 1px solid #e0c85c; }
.risk-high   { background: rgba(224,92,92,0.15);  color: #e05c5c; border: 1px solid #e05c5c; }

.stSlider [data-baseweb="slider"] div[role="slider"] { background-color: var(--accent) !important; }

.stButton button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

hr { border-color: var(--border) !important; }
.js-plotly-plot .plotly .bg { fill: transparent !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = "models"
CHART_CONFIG = {"displayModeBar": False}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e8eaf0", family="DM Sans"),
    xaxis=dict(gridcolor="#2a3040", linecolor="#2a3040"),
    yaxis=dict(gridcolor="#2a3040", linecolor="#2a3040"),
)


@st.cache_resource
def loadArtifacts():
    if not os.path.exists(f"{MODELS_DIR}/random_forest.pkl"):
        return None, None, None, None
    randomForest = joblib.load(f"{MODELS_DIR}/random_forest.pkl")
    xgboost = joblib.load(f"{MODELS_DIR}/xgboost.pkl")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    metadata = joblib.load(f"{MODELS_DIR}/metadata.pkl")
    return randomForest, xgboost, scaler, metadata


@st.cache_data
def loadTestData():
    path = f"{MODELS_DIR}/test_split.csv"
    return pd.read_csv(path) if os.path.exists(path) else None


def positiveClassShap(model, scaledRows):
    """SHAP values for the 'disease' class, always shaped (rows, features).

    Depending on the model and SHAP version, TreeExplainer returns a list with
    one array per class, a (rows, features, classes) array, or a plain
    (rows, features) array. Normalize all three to the positive class.
    """
    values = shap.TreeExplainer(model).shap_values(scaledRows)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def riskLabel(probability):
    if probability < 0.35:
        return "Low", "risk-low"
    if probability < 0.65:
        return "Moderate", "risk-medium"
    return "High", "risk-high"


def gaugeChart(probability):
    barColor = "#e05c5c" if probability >= 0.65 else ("#e0c85c" if probability >= 0.35 else "#5ce0b0")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"suffix": "%", "font": {"size": 36, "color": "#e8eaf0"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#7a8299"},
            "bar": {"color": barColor},
            "bgcolor": "#1c2030",
            "bordercolor": "#2a3040",
            "steps": [
                {"range": [0, 35], "color": "rgba(92,224,176,0.08)"},
                {"range": [35, 65], "color": "rgba(224,200,92,0.08)"},
                {"range": [65, 100], "color": "rgba(224,92,92,0.08)"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 3}, "thickness": 0.8,
                          "value": probability * 100},
        },
    ))
    fig.update_layout(height=260, margin=dict(t=20, b=0, l=10, r=10), **PLOTLY_LAYOUT)
    return fig


def shapWaterfall(model, scaler, featureNames, featureLabels, patientDf):
    scaledRow = scaler.transform(patientDf)
    shapValues = positiveClassShap(model, scaledRow)[0]
    labels = [featureLabels.get(name, name) for name in featureNames]

    order = [int(i) for i in np.argsort(np.abs(shapValues))[::-1][:10]]
    sortedValues = [float(shapValues[i]) for i in order]
    sortedLabels = [labels[i] for i in order]
    sortedData = [float(patientDf.iloc[0, i]) for i in order]
    colors = ["#e05c5c" if value > 0 else "#5ce0b0" for value in sortedValues]

    fig = go.Figure(go.Bar(
        x=sortedValues[::-1],
        y=[f"{label} = {data:.1f}" for label, data in zip(sortedLabels[::-1], sortedData[::-1])],
        orientation="h",
        marker_color=colors[::-1],
        marker_line_width=0,
    ))
    fig.update_layout(
        title="Top 10 Feature Contributions (SHAP)",
        xaxis_title="SHAP value (impact on predicted risk)",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        **PLOTLY_LAYOUT,
    )
    fig.add_vline(x=0, line_color="#7a8299", line_width=1)
    return fig


def featureImportanceChart(model, featureNames, featureLabels, title):
    importances = model.feature_importances_
    labels = [featureLabels.get(name, name) for name in featureNames]
    order = np.argsort(importances)
    fig = go.Figure(go.Bar(
        x=importances[order],
        y=[labels[i] for i in order],
        orientation="h",
        marker=dict(color=importances[order],
                    colorscale=[[0, "#2a3040"], [1, "#e05c5c"]], showscale=False),
    ))
    fig.update_layout(title=title, xaxis_title="Importance", height=420,
                      margin=dict(l=10, r=10, t=40, b=10), **PLOTLY_LAYOUT)
    return fig


def rocChart(modelsByName, scaler, testDf, featureNames):
    fig = go.Figure()
    colors = {"Random Forest": "#e05c5c", "XGBoost": "#5ce0b0"}
    scaledTest = scaler.transform(testDf[featureNames])
    truth = testDf["target"]

    for name, model in modelsByName.items():
        probabilities = model.predict_proba(scaledTest)[:, 1]
        fpr, tpr, _ = roc_curve(truth, probabilities)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={auc(fpr, tpr):.3f})",
            line=dict(color=colors[name], width=2.5),
        ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                             line=dict(color="#7a8299", width=1.5, dash="dash")))
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        legend=dict(x=0.55, y=0.1, bgcolor="rgba(0,0,0,0)"),
        **PLOTLY_LAYOUT,
    )
    return fig


def confusionChart(model, scaler, testDf, featureNames, name):
    predictions = model.predict(scaler.transform(testDf[featureNames]))
    matrix = confusion_matrix(testDf["target"], predictions)
    fig = px.imshow(
        matrix,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["No Disease", "Disease"],
        y=["No Disease", "Disease"],
        color_continuous_scale=[[0, "#151820"], [1, "#e05c5c"]],
        title=f"Confusion Matrix: {name}",
    )
    fig.update_layout(height=350, margin=dict(t=50, b=10, l=10, r=10), **PLOTLY_LAYOUT)
    return fig


def radarChart(featureRanges, patientValues):
    """Each axis is the patient's value scaled to the observed range of that feature."""
    categories = ["Age", "BP", "Chol", "Max HR", "ST Dep", "Vessels", "Thal"]
    keys = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal"]
    patientScaled = []
    for key in keys:
        low, high, _ = featureRanges[key]
        patientScaled.append((float(patientValues[key]) - low) / max(high - low, 1e-6))
    midpoint = [0.5] * len(keys)

    categories = categories + [categories[0]]
    patientScaled = patientScaled + [patientScaled[0]]
    midpoint = midpoint + [midpoint[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=midpoint, theta=categories, fill="toself", name="Range midpoint",
        line=dict(color="#7a8299", width=1.5), fillcolor="rgba(122,130,153,0.10)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=patientScaled, theta=categories, fill="toself", name="Patient",
        line=dict(color="#e05c5c", width=2), fillcolor="rgba(224,92,92,0.15)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1c2030",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2a3040",
                            tickfont=dict(color="#7a8299")),
            angularaxis=dict(gridcolor="#2a3040"),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=360,
        margin=dict(t=30, b=10, l=60, r=60),
        **PLOTLY_LAYOUT,
    )
    return fig


def main():
    randomForest, xgboost, scaler, metadata = loadArtifacts()
    testDf = loadTestData()

    st.markdown("""
    <div style='padding:1.5rem 0 0.5rem'>
        <h1 style='margin:0;font-size:2.4rem;'>CardioRisk</h1>
        <p style='color:#7a8299;margin:0.25rem 0 0;font-size:1rem;'>
            Heart Disease Risk Prediction | Random Forest and XGBoost | SHAP Interpretability
        </p>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    if randomForest is None:
        st.error("Models not found. Run `python train_model.py` first, then reload this page.")
        st.code("python train_model.py", language="bash")
        return

    if metadata.get("dataSource") == "demo":
        st.warning("These models were trained on synthetic demo data. "
                   "Metrics and predictions are not meaningful. "
                   "Run `python train_model.py` with an internet connection to use the real dataset.")

    featureNames = metadata["featureNames"]
    featureLabels = metadata["featureLabels"]
    featureRanges = metadata["featureRanges"]
    rfMetrics = metadata["rfMetrics"]
    xgbMetrics = metadata["xgbMetrics"]

    def rangeSlider(key, step=None):
        low, high, default = featureRanges[key]
        label = featureLabels[key]
        if isinstance(default, float):
            return st.slider(label, float(low), float(high), float(default), step=step or 0.1)
        return st.slider(label, int(low), int(high), int(default))

    st.sidebar.markdown("## Patient Parameters")
    st.sidebar.markdown("Adjust the controls to enter patient values.")

    # Widgets inside an expander must use st.*, not st.sidebar.*, to render in it.
    with st.sidebar.expander("Demographics", expanded=True):
        age = rangeSlider("age")
        sex = st.selectbox("Sex", [0, 1], index=1, format_func=lambda x: "Male" if x else "Female")

    with st.sidebar.expander("Clinical", expanded=True):
        cp = st.selectbox(
            "Chest Pain Type", [0, 1, 2, 3],
            format_func=lambda x: ["Typical Angina", "Atypical Angina",
                                   "Non-Anginal", "Asymptomatic"][x])
        trestbps = rangeSlider("trestbps")
        chol = rangeSlider("chol")
        fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1],
                           format_func=lambda x: "Yes" if x else "No")
        restecg = st.selectbox(
            "Resting ECG", [0, 1, 2],
            format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])

    with st.sidebar.expander("Exercise", expanded=True):
        thalach = rangeSlider("thalach")
        exang = st.selectbox("Exercise Induced Angina", [0, 1],
                             format_func=lambda x: "Yes" if x else "No")
        oldpeak = rangeSlider("oldpeak")
        slope = st.selectbox(
            "Slope of Peak Exercise ST", [0, 1, 2],
            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])

    with st.sidebar.expander("Imaging", expanded=True):
        ca = st.selectbox("# Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox(
            "Thal", [1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x])

    modelChoice = st.sidebar.radio("Model", ["Random Forest", "XGBoost"], horizontal=True)

    patientDict = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs,
                       restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak,
                       slope=slope, ca=ca, thal=thal)
    patientDf = pd.DataFrame([patientDict])[featureNames]

    model = randomForest if modelChoice == "Random Forest" else xgboost
    probability = float(model.predict_proba(scaler.transform(patientDf))[0, 1])
    label, badgeClass = riskLabel(probability)

    assessmentTab, performanceTab, featuresTab = st.tabs(
        ["Patient Assessment", "Model Performance", "Feature Analysis"])

    with assessmentTab:
        gaugeCol, radarCol, vitalsCol = st.columns([1.1, 1.4, 1.1])

        with gaugeCol:
            st.markdown("#### Risk Score")
            st.plotly_chart(gaugeChart(probability), config=CHART_CONFIG)
            st.markdown(
                f"<div style='text-align:center;margin-top:-1rem'>"
                f"<span class='risk-badge {badgeClass}'>{label} Risk</span></div>",
                unsafe_allow_html=True,
            )

        with radarCol:
            st.markdown("#### Patient Profile vs Observed Range")
            st.plotly_chart(radarChart(featureRanges, patientDict), config=CHART_CONFIG)

        with vitalsCol:
            st.markdown("#### Key Vitals")
            st.metric("Age", f"{age} yrs")
            st.metric("Resting BP", f"{trestbps} mmHg")
            st.metric("Cholesterol", f"{chol} mg/dl")
            st.metric("Max Heart Rate", f"{thalach} bpm")
            st.metric("ST Depression", f"{oldpeak}")
            st.metric("Vessels (CA)", f"{ca}")

        st.divider()

        st.markdown("#### SHAP: Feature Contribution Breakdown")
        st.markdown(
            "<p style='color:#7a8299;font-size:0.88rem;'>"
            "<span style='color:#e05c5c'>Red bars</span> push the predicted risk <b>higher</b>. "
            "<span style='color:#5ce0b0'>Green bars</span> push it <b>lower</b>.</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            shapWaterfall(model, scaler, featureNames, featureLabels, patientDf),
            config=CHART_CONFIG,
        )

        st.markdown("#### Clinical Decision Support")
        notes = []
        if probability >= 0.65:
            notes.append("**High cardiovascular risk.** Cardiology referral recommended.")
        elif probability >= 0.35:
            notes.append("**Moderate risk.** Further diagnostic workup advised (ECG, stress test).")
        else:
            notes.append("**Low risk.** Continue routine monitoring and preventive care.")
        if chol > 240:
            notes.append("Elevated cholesterol (>240 mg/dl): consider lipid management.")
        if trestbps > 140:
            notes.append("Resting BP above 140 mmHg: hypertension assessment recommended.")
        if thalach < 100:
            notes.append("Low max heart rate may indicate reduced cardiovascular reserve.")
        if exang == 1:
            notes.append("Exercise-induced angina present: a risk factor for obstructive CAD.")
        if ca >= 2:
            notes.append("Two or more major vessels colored: consider angiography.")
        for note in notes:
            st.markdown(f"- {note}")

    with performanceTab:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RF Accuracy", f"{rfMetrics['accuracy']:.1%}")
        col2.metric("RF ROC-AUC", f"{rfMetrics['rocAuc']:.3f}")
        col3.metric("XGB Accuracy", f"{xgbMetrics['accuracy']:.1%}")
        col4.metric("XGB ROC-AUC", f"{xgbMetrics['rocAuc']:.3f}")

        st.divider()

        if testDf is not None:
            rocCol, matrixCol = st.columns(2)
            with rocCol:
                st.plotly_chart(
                    rocChart({"Random Forest": randomForest, "XGBoost": xgboost},
                             scaler, testDf, featureNames),
                    config=CHART_CONFIG)
            with matrixCol:
                st.plotly_chart(
                    confusionChart(model, scaler, testDf, featureNames, modelChoice),
                    config=CHART_CONFIG)

            st.markdown("#### Predicted Probability Distribution")
            scaledTest = scaler.transform(testDf[featureNames])
            distribution = go.Figure()
            distribution.add_trace(go.Histogram(
                x=randomForest.predict_proba(scaledTest)[:, 1], name="Random Forest",
                marker_color="#e05c5c", opacity=0.7, nbinsx=30))
            distribution.add_trace(go.Histogram(
                x=xgboost.predict_proba(scaledTest)[:, 1], name="XGBoost",
                marker_color="#5ce0b0", opacity=0.7, nbinsx=30))
            distribution.update_layout(
                barmode="overlay",
                xaxis_title="Predicted probability of disease",
                yaxis_title="Count",
                height=320,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(distribution, config=CHART_CONFIG)
        else:
            st.info("Test split not found. Re-run train_model.py to generate it.")

    with featuresTab:
        rfCol, xgbCol = st.columns(2)
        with rfCol:
            st.plotly_chart(
                featureImportanceChart(randomForest, featureNames, featureLabels,
                                       "Random Forest: Feature Importances"),
                config=CHART_CONFIG)
        with xgbCol:
            st.plotly_chart(
                featureImportanceChart(xgboost, featureNames, featureLabels,
                                       "XGBoost: Feature Importances"),
                config=CHART_CONFIG)

        st.divider()
        st.markdown("#### Population-level SHAP Summary (Random Forest, test set)")

        if testDf is not None:
            with st.spinner("Computing SHAP values for the test set ..."):
                scaledTest = scaler.transform(testDf[featureNames])
                meanAbsShap = np.abs(positiveClassShap(randomForest, scaledTest)).mean(axis=0)
                labels = [featureLabels.get(name, name) for name in featureNames]
                order = [int(i) for i in np.argsort(meanAbsShap)]

                fig = go.Figure(go.Bar(
                    x=meanAbsShap[order],
                    y=[labels[i] for i in order],
                    orientation="h",
                    marker=dict(color=meanAbsShap[order],
                                colorscale=[[0, "#2a3040"], [0.5, "#5ca8e0"], [1, "#e05c5c"]],
                                showscale=False),
                ))
                fig.update_layout(
                    xaxis_title="Mean |SHAP value|",
                    title="Global Feature Importance via SHAP",
                    height=420,
                    margin=dict(l=10, r=10, t=40, b=10),
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig, config=CHART_CONFIG)
        else:
            st.info("Test split not available. Re-run train_model.py.")

    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#7a8299;font-size:0.8rem;'>"
        "CardioRisk | UCI Heart Disease Dataset | Daniel Ellis, UMBC Data Science</p>",
        unsafe_allow_html=True,
    )


main()
