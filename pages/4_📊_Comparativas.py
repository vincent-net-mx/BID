import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Imports for cross-validation and metrics
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
import catboost as cb
import matplotlib.pyplot as plt

# Imports for the BID system backend
from motor_bayesiano import MotorBayesiano
import predictor as P

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="BID Performance & Ablation", page_icon="📊", layout="wide")

st.title("📊 Performance Audit, Ablation, and Baseline Evaluation")
st.markdown("""
This section implements the required evaluations: **5-Fold Cross-Validation**, 
architectural comparison (Logistic Regression / Bayes-Only, Pure CatBoost, and Hybrid BID), and the probabilistic baseline analysis.
""")

@st.cache_data(show_spinner=False)
def ejecutar_experimentos_completos():
    df = pd.read_csv("data/BID_dataset.csv").drop(columns=P.COLUMNAS_PROHIBIDAS, errors='ignore')
    y = df[P.TARGET].astype(int).to_numpy()
    grupos = df["asn"].fillna("unknown").astype(str).to_numpy()
    FR = datetime(2026, 7, 28)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    res_lr_reg, res_cb_puro_reg, res_cb_hib_reg = {"acc": [], "f1": [], "rec": []}, {"acc": [], "f1": [], "rec": []}, {"acc": [], "f1": [], "rec": []}
    res_lr_grp, res_cb_puro_grp, res_cb_hib_grp = {"acc": [], "f1": [], "rec": []}, {"acc": [], "f1": [], "rec": []}, {"acc": [], "f1": [], "rec": []}

    # --- CYCLE 1: StratifiedKFold (Record-level) ---
    for tr, te in skf.split(df, y):
        dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        motor = MotorBayesiano().fit(dtr, P.V_ACTIVAS, target=P.TARGET)
        ptr, pte = motor.inferir_lote(dtr, P.V_ACTIVAS), motor.inferir_lote(dte, P.V_ACTIVAS)
        
        lr = LogisticRegression()
        lr.fit(np.array(ptr).reshape(-1, 1), y[tr])
        pr_lr = lr.predict_proba(np.array(pte).reshape(-1, 1))[:, 1]
        res_lr_reg["acc"].append(accuracy_score(y[te], (pr_lr > 0.5).astype(int)))
        res_lr_reg["f1"].append(f1_score(y[te], (pr_lr > 0.5).astype(int), zero_division=0))
        res_lr_reg["rec"].append(recall_score(y[te], (pr_lr > 0.5).astype(int), zero_division=0))
        
        Xtr_puro = P.construir_features(dtr, np.zeros(len(dtr)), FR)[P.FEATURES]
        Xte_puro = P.construir_features(dte, np.zeros(len(dte)), FR)[P.FEATURES]
        cb_puro = cb.CatBoostClassifier(**P.PARAMS_CATBOOST)
        cb_puro.fit(Xtr_puro, y[tr], cat_features=[c for c in P.CAT_FEATURES if c in P.FEATURES], verbose=False)
        pr_puro = cb_puro.predict_proba(Xte_puro)[:, 1]
        res_cb_puro_reg["acc"].append(accuracy_score(y[te], (pr_puro > 0.5).astype(int)))
        res_cb_puro_reg["f1"].append(f1_score(y[te], (pr_puro > 0.5).astype(int), zero_division=0))
        res_cb_puro_reg["rec"].append(recall_score(y[te], (pr_puro > 0.5).astype(int), zero_division=0))

        Xtr_hib = P.construir_features(dtr, ptr, FR)[P.FEATURES]
        Xte_hib = P.construir_features(dte, pte, FR)[P.FEATURES]
        cb_hib = cb.CatBoostClassifier(**P.PARAMS_CATBOOST)
        cb_hib.fit(Xtr_hib, y[tr], cat_features=[c for c in P.CAT_FEATURES if c in P.FEATURES], verbose=False)
        pr_hib = cb_hib.predict_proba(Xte_hib)[:, 1]
        res_cb_hib_reg["acc"].append(accuracy_score(y[te], (pr_hib > 0.5).astype(int)))
        res_cb_hib_reg["f1"].append(f1_score(y[te], (pr_hib > 0.5).astype(int), zero_division=0))
        res_cb_hib_reg["rec"].append(recall_score(y[te], (pr_hib > 0.5).astype(int), zero_division=0))

    # --- CYCLE 2: StratifiedGroupKFold (Grouped by ASN) ---
    for tr, te in sgkf.split(df, y, grupos):
        dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        motor = MotorBayesiano().fit(dtr, P.V_ACTIVAS, target=P.TARGET)
        ptr, pte = motor.inferir_lote(dtr, P.V_ACTIVAS), motor.inferir_lote(dte, P.V_ACTIVAS)
        
        lr = LogisticRegression()
        lr.fit(np.array(ptr).reshape(-1, 1), y[tr])
        pr_lr = lr.predict_proba(np.array(pte).reshape(-1, 1))[:, 1]
        res_lr_grp["acc"].append(accuracy_score(y[te], (pr_lr > 0.5).astype(int)))
        res_lr_grp["f1"].append(f1_score(y[te], (pr_lr > 0.5).astype(int), zero_division=0))
        res_lr_grp["rec"].append(recall_score(y[te], (pr_lr > 0.5).astype(int), zero_division=0))
        
        Xtr_puro = P.construir_features(dtr, np.zeros(len(dtr)), FR)[P.FEATURES]
        Xte_puro = P.construir_features(dte, np.zeros(len(dte)), FR)[P.FEATURES]
        cb_puro = cb.CatBoostClassifier(**P.PARAMS_CATBOOST)
        cb_puro.fit(Xtr_puro, y[tr], cat_features=[c for c in P.CAT_FEATURES if c in P.FEATURES], verbose=False)
        pr_puro = cb_puro.predict_proba(Xte_puro)[:, 1]
        res_cb_puro_grp["acc"].append(accuracy_score(y[te], (pr_puro > 0.5).astype(int)))
        res_cb_puro_grp["f1"].append(f1_score(y[te], (pr_puro > 0.5).astype(int), zero_division=0))
        res_cb_puro_grp["rec"].append(recall_score(y[te], (pr_puro > 0.5).astype(int), zero_division=0))

        Xtr_hib = P.construir_features(dtr, ptr, FR)[P.FEATURES]
        Xte_hib = P.construir_features(dte, pte, FR)[P.FEATURES]
        cb_hib = cb.CatBoostClassifier(**P.PARAMS_CATBOOST)
        cb_hib.fit(Xtr_hib, y[tr], cat_features=[c for c in P.CAT_FEATURES if c in P.FEATURES], verbose=False)
        pr_hib = cb_hib.predict_proba(Xte_hib)[:, 1]
        res_cb_hib_grp["acc"].append(accuracy_score(y[te], (pr_hib > 0.5).astype(int)))
        res_cb_hib_grp["f1"].append(f1_score(y[te], (pr_hib > 0.5).astype(int), zero_division=0))
        res_cb_hib_grp["rec"].append(recall_score(y[te], (pr_hib > 0.5).astype(int), zero_division=0))

    return {
        "registro": {
            "LR": {k: np.mean(v) for k, v in res_lr_reg.items()},
            "CB_Puro": {k: np.mean(v) for k, v in res_cb_puro_reg.items()},
            "CB_Hibrido": {k: np.mean(v) for k, v in res_cb_hib_reg.items()}
        },
        "grupo": {
            "LR": {k: np.mean(v) for k, v in res_lr_grp.items()},
            "CB_Puro": {k: np.mean(v) for k, v in res_cb_puro_grp.items()},
            "CB_Hibrido": {k: np.mean(v) for k, v in res_cb_hib_grp.items()}
        }
    }

with st.spinner("Loading cross-validation metrics..."):
    data = ejecutar_experimentos_completos()

# --- MODE SELECTOR ---
modo_eval = st.radio(
    "🔬 Select Validation Protocol:",
    [
        "Standard Mode (StratifiedKFold at record level)", 
        "New Infrastructure Mode (StratifiedGroupKFold by ASN)"
    ],
    index=1
)

res_actual = data["registro"] if "Standard" in modo_eval else data["grupo"]

st.markdown("### 🏆 Architectural Comparison")
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("1. Regression (Bayes-Only)")
    st.write(f"Acc: **{res_actual['LR']['acc']*100:.2f}%**")
    st.write(f"F1: **{res_actual['LR']['f1']*100:.2f}%**")
    st.write(f"Recall: **{res_actual['LR']['rec']*100:.2f}%**")
with c2:
    st.subheader("2. Pure CatBoost (No Bayes)")
    st.write(f"Acc: **{res_actual['CB_Puro']['acc']*100:.2f}%**")
    st.write(f"F1: **{res_actual['CB_Puro']['f1']*100:.2f}%**")
    st.write(f"Recall: **{res_actual['CB_Puro']['rec']*100:.2f}%**")
with c3:
    st.subheader("3. BID Hybrid (Bayes + CatBoost)")
    st.write(f"Acc: **{res_actual['CB_Hibrido']['acc']*100:.2f}%** 🔥")
    st.write(f"F1: **{res_actual['CB_Hibrido']['f1']*100:.2f}%** 🔥")
    st.write(f"Recall: **{res_actual['CB_Hibrido']['rec']*100:.2f}%** 🔥")

# --- VISUAL SECTION: LOGISTIC CURVE (SIGMOID) ---
st.markdown("---")
st.markdown("### 📈 Baseline Logistic Regression Curve")
st.markdown("Visualization of the fitted logistic probability function derived from the risk calculated by the Bayesian engine.")

@st.cache_data
def generar_curva_logistica():
    df = pd.read_csv("data/BID_dataset.csv").drop(columns=P.COLUMNAS_PROHIBIDAS, errors='ignore')
    y = df[P.TARGET].astype(int).to_numpy()
    motor = MotorBayesiano().fit(df, P.V_ACTIVAS, target=P.TARGET)
    ptr = motor.inferir_lote(df, P.V_ACTIVAS)
    
    # Train simple logistic regression with the Bayesian score
    X = np.array(ptr).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(X, y)
    
    # Generate points for the smooth sigmoid curve
    x_test = np.linspace(X.min(), X.max(), 300)
    y_prob = lr.predict_proba(x_test.reshape(-1, 1))[:, 1]
    
    return X.flatten(), y, x_test, y_prob

x_vals, y_vals, x_curve, y_curve = generar_curva_logistica()

fig, ax = plt.subplots(figsize=(10, 5))
# Real scatter points
ax.scatter(x_vals, y_vals, color='gray', alpha=0.6, s=25, label='Real IPs (Observations)')
# Fitted sigmoid curve
ax.plot(x_curve, y_curve, color='#b03060', linewidth=3, label='Sigmoidal Logistic Curve')

# NUEVO TÍTULO Y ESTILO PARA EL PAPER
ax.set_title("Logistic Regression Baseline Model", fontsize=12, fontweight='bold')
ax.set_xlabel("Calculated Bayesian Risk (x)", fontsize=10)
ax.set_ylabel("Maliciousness Probability", fontsize=10)
# Cuadrícula más sutil para un aspecto más limpio
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='lower right')

st.pyplot(fig)

with st.expander("📝 Formatting Note for Paper"):
    st.markdown("""
    When pasting this chart into your document, use the following caption directly below the image:
    
    **Fig. X. Logistic regression baseline modeling the continuous threat probability as a function of the isolated Bayesian risk.**
    """)