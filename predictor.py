"""
predictor.py — Sistema BID (Bad IP Detector)
============================================
Contiene:
  * ejecutar_prediccion_hibrida   -> motor de inferencia (INTERFAZ INTACTA)
  * entrenar_modelo_kfold         -> 5-Fold StratifiedKFold (respuesta a Revisor #4)
  * entrenar_modelo_grupo_kfold   -> 5-Fold agrupado por infraestructura (ASN/owner)
  * entrenar_modelo_final         -> reentrenamiento con el 100% de los datos

DEPENDENCIAS
------------
Producción (Streamlit / inferencia):  pandas, numpy, catboost, motor_bayesiano
Investigación (validación cruzada):   + scikit-learn

scikit-learn se importa DENTRO de las funciones de entrenamiento, no a nivel
de módulo. La app de Streamlit puede importar este archivo sin tenerlo
instalado. Si se llama a una función de validación sin scikit-learn presente,
se lanza un ImportError con instrucciones claras.

Este módulo depende únicamente de `motor_bayesiano.py`. No requiere
`motor_bayesiano_fold.py` ni ningún otro respaldo.

Principio de diseño: la ingeniería de características vive en UNA sola función
(`construir_features`) usada tanto por entrenamiento como por inferencia, para
garantizar paridad train/serving y eliminar el riesgo de skew silencioso.
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import catboost as cb

import motor_bayesiano as bayes
from motor_bayesiano import MotorBayesiano

warnings.filterwarnings("ignore")


def crear_motor_bayesiano(alpha=1.0):
    """Fabrica un motor bayesiano ajustable por pliegue."""
    return MotorBayesiano(alpha=alpha)


# --------------------------------------------------------------------------- #
# Carga diferida de scikit-learn (solo para entrenamiento / validación)
# --------------------------------------------------------------------------- #
_MENSAJE_SKLEARN = (
    "Las funciones de entrenamiento y validación cruzada requieren scikit-learn.\n"
    "    Instalar con:  pip install scikit-learn\n"
    "La inferencia en producción (ejecutar_prediccion_hibrida) NO lo requiere; "
    "si este error aparece al levantar Streamlit, significa que algo está "
    "importando una función de entrenamiento por error."
)


def _cargar_sklearn():
    """Importa scikit-learn bajo demanda y devuelve lo necesario."""
    try:
        from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            log_loss,
            roc_auc_score,
            average_precision_score,
            confusion_matrix,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_MENSAJE_SKLEARN) from exc

    return {
        "StratifiedKFold": StratifiedKFold,
        "StratifiedGroupKFold": StratifiedGroupKFold,
        "accuracy_score": accuracy_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "log_loss": log_loss,
        "roc_auc_score": roc_auc_score,
        "average_precision_score": average_precision_score,
        "confusion_matrix": confusion_matrix,
    }


# --------------------------------------------------------------------------- #
# Constantes del pipeline
# --------------------------------------------------------------------------- #
V_ACTIVAS = [
    "abuseip_score",
    "usage_type",
    "abuseip_categories",
    "abuseip_distinct_users",
    "abuseip_last_reported",
    "country",
    "asn",
    "isp",
    "infra_owner",
]

FEATURES = [
    "abuseip_score",
    "usage_type",
    "attack_type_count",
    "abuseip_distinct_users",
    "days_since_last_report",
    "country",
    "asn",
    "isp",
    "infra_owner",
    "bayesian_risk",
]

CAT_FEATURES = ["usage_type", "country", "asn", "isp", "infra_owner"]

NUM_FEATURES = [
    "abuseip_score",
    "attack_type_count",
    "abuseip_distinct_users",
    "days_since_last_report",
    "bayesian_risk",
]

TARGET = "is_malicious"

# Columnas derivadas de la etiqueta: NUNCA deben entrar como features.
COLUMNAS_PROHIBIDAS = ["attack_type", "threat_theme", "threat_level"]

DIAS_SIN_REPORTE = 3650  # valor centinela para IPs nunca reportadas

PARAMS_CATBOOST = dict(
    iterations=400,
    depth=4,
    learning_rate=0.05,
    l2_leaf_reg=6.0,
    loss_function="Logloss",
    eval_metric="Logloss",
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
)


# =========================================================================== #
# 1. INGENIERÍA DE CARACTERÍSTICAS (compartida entrenamiento / inferencia)
# =========================================================================== #
def calcular_dias_desde_reporte(serie: pd.Series, fecha_ref: datetime) -> pd.Series:
    """Convierte `abuseip_last_reported` en días transcurridos (vectorizado)."""
    fechas = pd.to_datetime(serie, errors="coerce", utc=True)
    try:
        fechas = fechas.dt.tz_localize(None)
    except TypeError:
        fechas = fechas.dt.tz_convert(None)
    dias = (pd.Timestamp(fecha_ref) - fechas).dt.days
    return dias.fillna(DIAS_SIN_REPORTE).clip(lower=0).astype(float)


def contar_tipos_ataque(serie: pd.Series) -> pd.Series:
    """Cuenta categorías de abuso reportadas en AbuseIPDB."""
    def _contar(x):
        if pd.isna(x) or str(x).strip() in ("No_Reports", "", "nan"):
            return 0
        return len([t for t in str(x).split(",") if t.strip()])

    return serie.apply(_contar).astype(float)


def construir_features(
    df: pd.DataFrame,
    riesgo_bayesiano,
    fecha_ref: datetime = None,
) -> pd.DataFrame:
    """
    Construye la matriz X en el orden estricto de `FEATURES`.

    Parameters
    ----------
    df : DataFrame con las columnas crudas del perfil.
    riesgo_bayesiano : escalar o array con la posterior bayesiana por registro.
    fecha_ref : fecha de referencia para `days_since_last_report`.
                En entrenamiento debe fijarse para garantizar reproducibilidad.
    """
    fecha_ref = fecha_ref or datetime.now()
    out = df.copy().reset_index(drop=True)

    out["bayesian_risk"] = np.asarray(riesgo_bayesiano, dtype=float).reshape(-1)
    out["days_since_last_report"] = calcular_dias_desde_reporte(
        out["abuseip_last_reported"], fecha_ref
    )
    out["attack_type_count"] = contar_tipos_ataque(out["abuseip_categories"])

    X = out[FEATURES].copy()
    for col in CAT_FEATURES:
        X[col] = X[col].fillna("unknown").astype(str)
    for col in NUM_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X


# =========================================================================== #
# 2. INFERENCIA — INTERFAZ CONSERVADA (compatible con 1_📡_Analizador.py)
# =========================================================================== #
def ejecutar_prediccion_hibrida(perfil_limpio, modelo_path, verbose=True, motor_bayes=None):
    """
    Motor de inferencia pura. Recibe datos ya normalizados y retorna
    el veredicto final combinando Bayes e IA.

    NOTA: firma y diccionario de retorno idénticos a la versión original.
    El parámetro `motor_bayes` es opcional y retrocompatible: si se omite,
    se usa la interfaz de módulo de `motor_bayesiano`.

    No requiere scikit-learn.
    """
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    vprint(f"\n" + "=" * 60)
    vprint(f"SISTEMA BID: MOTOR DE INFERENCIA (STEP 4)")
    vprint("=" * 60)

    # 1. Inferencia Bayesiana (Contrapeso Estadístico)
    vprint("[1/3] Calculando Riesgo Bayesiano...")
    motor = motor_bayes if motor_bayes is not None else bayes
    _, p_bayesian = motor.ejecutar_inferencia(perfil_limpio, V_ACTIVAS)

    # 2. Ingeniería de características "en vuelo" (función compartida)
    vprint("[2/3] Preparando vector de características para CatBoost...")
    ahora = datetime.now()
    X_pred = construir_features(pd.DataFrame([perfil_limpio]), p_bayesian, fecha_ref=ahora)

    # 3. Clasificación final
    vprint("[3/3] Ejecutando veredicto de IA...")
    modelo = cb.CatBoostClassifier()
    try:
        modelo.load_model(modelo_path)
        prob_malicia = modelo.predict_proba(X_pred)[0][1]
        veredicto = "MALICIOSA" if prob_malicia > 0.5 else "BENIGNA"
    except Exception as e:
        vprint(f"❌ Error crítico en el motor de IA: {e}")
        return None

    vprint(f"\n[+] Análisis completado: {veredicto} ({prob_malicia:.2%})")
    vprint("=" * 60 + "\n")

    return {
        "veredicto": veredicto,
        "probabilidad_ia": prob_malicia,
        "riesgo_bayesiano": p_bayesian,
        "isp_normalizado": perfil_limpio.get("isp"),
        "timestamp": ahora.strftime("%Y-%m-%d %H:%M:%S"),
    }


# =========================================================================== #
# 3. VALIDACIÓN CRUZADA  (requiere scikit-learn)
# =========================================================================== #
def _metricas_pliegue(y_true, y_pred, y_prob, M):
    return {
        "accuracy": M["accuracy_score"](y_true, y_pred),
        "precision": M["precision_score"](y_true, y_pred, zero_division=0),
        "recall": M["recall_score"](y_true, y_pred, zero_division=0),
        "f1": M["f1_score"](y_true, y_pred, zero_division=0),
        "logloss": M["log_loss"](y_true, y_prob, labels=[0, 1]),
        "roc_auc": M["roc_auc_score"](y_true, y_prob) if len(set(y_true)) > 1 else np.nan,
        "pr_auc": M["average_precision_score"](y_true, y_prob) if len(set(y_true)) > 1 else np.nan,
    }


def _ejecutar_cv(
    df,
    splitter,
    M,
    grupos=None,
    fecha_ref=None,
    params=None,
    umbral=0.5,
    verbose=True,
    etiqueta="StratifiedKFold",
):
    """Núcleo de la validación cruzada. El motor bayesiano se REAJUSTA por pliegue."""
    fecha_ref = fecha_ref or datetime.now()
    params = params or PARAMS_CATBOOST
    y = df[TARGET].astype(int).to_numpy()

    filas, oof_prob = [], np.zeros(len(df))
    split_args = (df, y, grupos) if grupos is not None else (df, y)

    for k, (idx_tr, idx_te) in enumerate(splitter.split(*split_args), start=1):
        df_tr = df.iloc[idx_tr].reset_index(drop=True)
        df_te = df.iloc[idx_te].reset_index(drop=True)

        # --- Motor bayesiano ajustado SOLO con el pliegue de entrenamiento ---
        motor = crear_motor_bayesiano().fit(df_tr, V_ACTIVAS, target=TARGET)
        p_tr = motor.inferir_lote(df_tr, V_ACTIVAS)
        p_te = motor.inferir_lote(df_te, V_ACTIVAS)

        X_tr = construir_features(df_tr, p_tr, fecha_ref)
        X_te = construir_features(df_te, p_te, fecha_ref)
        y_tr, y_te = y[idx_tr], y[idx_te]

        modelo = cb.CatBoostClassifier(**params)
        modelo.fit(X_tr, y_tr, cat_features=CAT_FEATURES)

        prob = modelo.predict_proba(X_te)[:, 1]
        pred = (prob > umbral).astype(int)
        oof_prob[idx_te] = prob

        m = _metricas_pliegue(y_te, pred, prob, M)
        m["pliegue"] = k
        m["n_test"] = len(idx_te)
        m["n_mal_test"] = int(y_te.sum())
        filas.append(m)

        if verbose:
            print(
                f"  [{etiqueta}] Pliegue {k}/{splitter.get_n_splits()} "
                f"(n={len(idx_te)}, mal={int(y_te.sum())}) -> "
                f"Acc={m['accuracy']:.4f} P={m['precision']:.4f} "
                f"R={m['recall']:.4f} F1={m['f1']:.4f} LL={m['logloss']:.4f}"
            )

    res = pd.DataFrame(filas)
    metricas = ["accuracy", "precision", "recall", "f1", "logloss", "roc_auc", "pr_auc"]
    resumen = pd.DataFrame(
        {
            "metrica": metricas,
            "media": [res[m].mean() for m in metricas],
            "std": [res[m].std(ddof=1) for m in metricas],
            "min": [res[m].min() for m in metricas],
            "max": [res[m].max() for m in metricas],
        }
    )
    cm = M["confusion_matrix"](y, (oof_prob > umbral).astype(int), labels=[0, 1])
    return {"por_pliegue": res, "resumen": resumen, "oof_prob": oof_prob,
            "matriz_confusion_oof": cm}


def entrenar_modelo_kfold(
    csv_path,
    modelo_path="modelo_bid.cbm",
    n_splits=5,
    random_state=42,
    fecha_ref=None,
    params=None,
    umbral=0.5,
    guardar_modelo=True,
    verbose=True,
):
    """
    Validación cruzada estratificada de k pliegues + entrenamiento final.

    Preserva la proporción de clases (~80/20) en cada pliegue y reajusta el
    motor bayesiano dentro de cada pliegue para evitar fuga de la etiqueta.

    Requiere scikit-learn.
    """
    M = _cargar_sklearn()
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in COLUMNAS_PROHIBIDAS if c in df.columns])
    fecha_ref = fecha_ref or datetime.now()

    if verbose:
        n_pos = int(df[TARGET].sum())
        print("=" * 68)
        print("SISTEMA BID — VALIDACIÓN CRUZADA ESTRATIFICADA")
        print("=" * 68)
        print(f"Registros: {len(df)} | Benignos: {len(df) - n_pos} | Maliciosos: {n_pos}")
        print(f"Pliegues: {n_splits} | shuffle=True | random_state={random_state}\n")

    skf = M["StratifiedKFold"](n_splits=n_splits, shuffle=True, random_state=random_state)
    salida = _ejecutar_cv(df, skf, M, None, fecha_ref, params, umbral, verbose,
                          etiqueta="Registro")

    if guardar_modelo:
        entrenar_modelo_final(df, modelo_path, fecha_ref, params, verbose)
        salida["modelo_path"] = modelo_path

    if verbose:
        print("\n" + tabla_markdown(salida["resumen"]))
    return salida


def entrenar_modelo_grupo_kfold(
    csv_path,
    columna_grupo="asn",
    n_splits=5,
    random_state=42,
    fecha_ref=None,
    params=None,
    umbral=0.5,
    verbose=True,
):
    """
    Validación cruzada estratificada AGRUPADA por infraestructura.

    Ningún ASN/ISP/propietario aparece simultáneamente en entrenamiento y
    prueba. Mide la capacidad de generalizar a infraestructura NO vista,
    que es el escenario real de despliegue.

    Requiere scikit-learn.
    """
    M = _cargar_sklearn()
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in COLUMNAS_PROHIBIDAS if c in df.columns])
    fecha_ref = fecha_ref or datetime.now()
    grupos = df[columna_grupo].fillna("unknown").astype(str).to_numpy()

    if verbose:
        print("=" * 68)
        print(f"SISTEMA BID — VALIDACIÓN CRUZADA AGRUPADA POR '{columna_grupo}'")
        print("=" * 68)
        print(f"Grupos únicos: {len(set(grupos))} | Pliegues: {n_splits}\n")

    sgkf = M["StratifiedGroupKFold"](n_splits=n_splits, shuffle=True,
                                     random_state=random_state)
    salida = _ejecutar_cv(df, sgkf, M, grupos, fecha_ref, params, umbral, verbose,
                          etiqueta=f"Grupo:{columna_grupo}")

    if verbose:
        print("\n" + tabla_markdown(salida["resumen"]))
    return salida


def entrenar_modelo_final(df, modelo_path, fecha_ref=None, params=None, verbose=True):
    """
    Reentrena con el 100% de los datos y persiste el modelo de producción.

    No requiere scikit-learn.
    """
    fecha_ref = fecha_ref or datetime.now()
    params = params or PARAMS_CATBOOST
    if isinstance(df, str):
        df = pd.read_csv(df)
    df = df.drop(columns=[c for c in COLUMNAS_PROHIBIDAS if c in df.columns])

    y = df[TARGET].astype(int).to_numpy()
    motor = crear_motor_bayesiano().fit(df, V_ACTIVAS, target=TARGET)
    X = construir_features(df, motor.inferir_lote(df, V_ACTIVAS), fecha_ref)

    modelo = cb.CatBoostClassifier(**params)
    modelo.fit(X, y, cat_features=CAT_FEATURES)
    modelo.save_model(modelo_path)

    if verbose:
        print(f"\n[+] Modelo final entrenado con {len(df)} registros -> {modelo_path}")
    return modelo


# =========================================================================== #
# 4. UTILIDADES DE REPORTE
# =========================================================================== #
NOMBRES = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Score",
    "logloss": "Log-Loss",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


def tabla_markdown(resumen: pd.DataFrame) -> str:
    """Tabla Markdown lista para la sección de Resultados."""
    lineas = [
        "| Métrica | Promedio ± Std | Mín | Máx |",
        "|---|---|---|---|",
    ]
    for _, r in resumen.iterrows():
        lineas.append(
            f"| {NOMBRES.get(r['metrica'], r['metrica'])} "
            f"| {r['media']:.4f} ± {r['std']:.4f} "
            f"| {r['min']:.4f} | {r['max']:.4f} |"
        )
    return "\n".join(lineas)


def tabla_latex(resumen: pd.DataFrame, caption="", label="tab:cv") -> str:
    """Tabla en formato IEEEtran."""
    filas = "\n".join(
        f"{NOMBRES.get(r['metrica'], r['metrica'])} & "
        f"${r['media']:.4f} \\pm {r['std']:.4f}$ \\\\"
        for _, r in resumen.iterrows()
    )
    return (
        "\\begin{table}[!t]\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
        "\\centering\n\\begin{tabular}{lc}\n\\hline\n"
        "\\textbf{Metric} & \\textbf{Mean} $\\pm$ \\textbf{Std}\\\\\n\\hline\n"
        + filas + "\n\\hline\n\\end{tabular}\n\\end{table}"
    )


# =========================================================================== #
# 5. MAIN — ejecución del experimento
# =========================================================================== #
if __name__ == "__main__":
    CSV = "data/BID_dataset.csv"
    MODELO = "data/modelo_bid.cbm"
    # Fecha de referencia FIJA: hace reproducible `days_since_last_report`.
    FECHA_REF = datetime(2026, 7, 28)

    # (A) Protocolo solicitado: 5-Fold estratificado a nivel de registro
    res_registro = entrenar_modelo_kfold(
        csv_path=CSV,
        modelo_path=MODELO,
        n_splits=5,
        random_state=42,
        fecha_ref=FECHA_REF,
    )

    # (B) Protocolo de control: 5-Fold agrupado por ASN (infraestructura no vista)
    res_grupo = entrenar_modelo_grupo_kfold(
        csv_path=CSV,
        columna_grupo="asn",
        n_splits=5,
        random_state=42,
        fecha_ref=FECHA_REF,
    )

    print("\n--- LaTeX (IEEEtran) ---")
    print(tabla_latex(res_registro["resumen"],
                      caption="5-fold stratified cross-validation (record-level).",
                      label="tab:cv_record"))
