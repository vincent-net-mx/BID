"""
motor_bayesiano.py — Sistema BID (Bad IP Detector)
==================================================
Motor de inferencia bayesiana (Naive Bayes) sobre el perfil de una IPv4.

CAMBIOS RESPECTO A LA VERSIÓN ORIGINAL
--------------------------------------
1. Suavizado de Laplace CORRECTO: (conteo + alpha) / (N_clase + alpha * K).
   La versión previa sustituía los conteos cero por 0.1 sin ajustar el
   denominador; eso no normaliza (las verosimilitudes sumaban hasta 1.109)
   y sobrepondera los niveles raros en ~9x.
2. Separación fit / predict. Las tablas de verosimilitud se estiman UNA vez
   y quedan almacenadas en el objeto. Antes, cada llamada releía el CSV
   completo y recalculaba todo, lo que impedía cualquier evaluación honesta:
   el registro evaluado alimentaba sus propios conteos (target leakage).
3. Coerción explícita de tipos. Antes, pasar un DataFrame no-string dejaba
   los subconjuntos vacíos y devolvía el prior en silencio, sin excepción.
4. Centinelas validados por columna ('No_Reports' pertenece a
   abuseip_categories, 'Never_Reported' a abuseip_last_reported). Antes un
   cruce entre ambos caía al piso numérico sin aviso.
5. Cómputo en espacio logarítmico (evita subdesbordamiento) e inferencia
   vectorizada por lotes.

COMPATIBILIDAD
--------------
`ejecutar_inferencia(perfil_ip, variables_activas, df=None)` conserva su
firma y su retorno `(prior, posterior)`. El código de Streamlit no cambia.

ADVERTENCIA METODOLÓGICA
------------------------
La función a nivel de módulo ajusta el motor con el corpus COMPLETO. Eso es
correcto para inferencia en producción sobre IPs nuevas, pero NO debe usarse
para evaluar registros que ya pertenecen al corpus. Para validación cruzada,
instanciar `MotorBayesiano()` y llamar `.fit()` con el pliegue de
entrenamiento únicamente.
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuración del espacio de variables
# --------------------------------------------------------------------------- #
NULOS = ["Unknown", "Error_Timeout", "None", "", "Fetch_Error", "Empty_JSON",
         "nan", "NaN", "None_Reported"]

# Variables de razón: se binarizan en (> 0) vs (== 0)
VARIABLES_RAZON = ["abuseip_score", "abuseip_distinct_users"]

# Variables de existencia: centinela propio de cada columna
CENTINELAS = {
    "abuseip_categories": "No_Reports",
    "abuseip_last_reported": "Never_Reported",
}

TARGET = "is_malicious"
ALPHA = 1.0  # parámetro de suavizado de Laplace

NIVEL_NO_VISTO = "__no_visto__"


# --------------------------------------------------------------------------- #
# Normalización y transformación (compartida por fit e inferencia)
# --------------------------------------------------------------------------- #
def _serie_target(df: pd.DataFrame) -> pd.Series:
    """Coerción robusta de la etiqueta: acepta 1/'1'/True indistintamente."""
    s = df[TARGET]
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def _transformar_variable(serie: pd.Series, variable: str) -> pd.Series:
    """Proyecta una columna cruda al espacio de niveles usado por el motor."""
    if variable in VARIABLES_RAZON:
        num = pd.to_numeric(serie, errors="coerce").fillna(0.0)
        return pd.Series(np.where(num > 0, "positivo", "cero"), index=serie.index)

    if variable in CENTINELAS:
        centinela = CENTINELAS[variable]
        txt = serie.astype(str).str.strip()
        sin = txt.isin([centinela] + NULOS)
        return pd.Series(np.where(sin, "sin_reportes", "con_reportes"), index=serie.index)

    # Nominales exactas: country, asn, isp, usage_type, infra_owner
    txt = serie.astype(str).str.strip()
    return txt.mask(txt.isin(NULOS), "Unknown")


def _transformar_df(df: pd.DataFrame, variables_activas) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for var in variables_activas:
        if var not in df.columns:
            out[var] = "Unknown"
        else:
            out[var] = _transformar_variable(df[var], var)
    return out


def _transformar_perfil(perfil: dict, variables_activas) -> dict:
    """Misma transformación para un único perfil (paridad train/serving)."""
    fila = {}
    for var in variables_activas:
        valor = perfil.get(var, "Unknown")
        serie = pd.Series([valor])
        fila[var] = _transformar_variable(serie, var).iloc[0]
    return fila


# --------------------------------------------------------------------------- #
# Motor
# --------------------------------------------------------------------------- #
class MotorBayesiano:
    """
    Naive Bayes categórico con suavizado de Laplace.

    Uso en validación cruzada
    -------------------------
        motor = MotorBayesiano().fit(df_train, variables_activas)
        p_test = motor.inferir_lote(df_test, variables_activas)

    Uso en inferencia unitaria
    --------------------------
        prior, posterior = motor.ejecutar_inferencia(perfil_ip, variables_activas)
    """

    def __init__(self, alpha: float = ALPHA):
        self.alpha = alpha
        self.prior_ = None
        self.log_prior_odds_ = None
        self.tablas_ = {}       # {var: {clase: {nivel: log P(nivel|clase)}}}
        self.k_ = {}            # {var: cardinalidad usada en el suavizado}
        self.variables_activas_ = None
        self.n_ = {}

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, variables_activas, target: str = TARGET):
        """Estima priors y verosimilitudes. Usar SOLO datos de entrenamiento."""
        self.variables_activas_ = list(variables_activas)
        y = _serie_target(df.rename(columns={target: TARGET}))
        X = _transformar_df(df, self.variables_activas_)

        n_total = len(y)
        if n_total == 0:
            raise ValueError("El DataFrame de entrenamiento está vacío.")
        n_mal = int(y.sum())
        n_ben = n_total - n_mal
        if n_mal == 0 or n_ben == 0:
            raise ValueError(
                f"El conjunto de entrenamiento no contiene ambas clases "
                f"(maliciosas={n_mal}, benignas={n_ben})."
            )

        self.n_ = {1: n_mal, 0: n_ben, "total": n_total}
        self.prior_ = {1: n_mal / n_total, 0: n_ben / n_total}
        self.log_prior_odds_ = np.log(self.prior_[1]) - np.log(self.prior_[0])

        self.tablas_, self.k_ = {}, {}
        for var in self.variables_activas_:
            niveles = sorted(set(X[var].unique()))
            # Las ramas binarias tienen espacio de niveles cerrado (K = 2).
            # Las nominales reservan una casilla para el nivel no observado.
            if var in VARIABLES_RAZON:
                niveles, K = ["cero", "positivo"], 2
            elif var in CENTINELAS:
                niveles, K = ["sin_reportes", "con_reportes"], 2
            else:
                K = len(niveles) + 1

            self.k_[var] = K
            tabla = {}
            for clase, n_c in ((1, n_mal), (0, n_ben)):
                conteos = X.loc[y == clase, var].value_counts()
                denom = n_c + self.alpha * K
                tabla[clase] = {
                    niv: np.log((conteos.get(niv, 0) + self.alpha) / denom)
                    for niv in niveles
                }
                tabla[clase][NIVEL_NO_VISTO] = np.log(self.alpha / denom)
            self.tablas_[var] = tabla
        return self

    # ------------------------------------------------------------------ #
    def _validar_ajuste(self):
        if self.prior_ is None:
            raise RuntimeError(
                "El motor no ha sido ajustado. Llamar .fit(df_train, variables_activas) "
                "antes de inferir."
            )

    def _log_odds(self, X: pd.DataFrame, variables_activas) -> np.ndarray:
        lo = np.full(len(X), self.log_prior_odds_, dtype=float)
        for var in variables_activas:
            tabla = self.tablas_[var]
            col = X[var].astype(str)
            l1 = col.map(tabla[1]).fillna(tabla[1][NIVEL_NO_VISTO]).to_numpy(dtype=float)
            l0 = col.map(tabla[0]).fillna(tabla[0][NIVEL_NO_VISTO]).to_numpy(dtype=float)
            lo += l1 - l0
        return lo

    @staticmethod
    def _sigmoide(lo):
        return 1.0 / (1.0 + np.exp(-np.clip(lo, -60, 60)))

    # ------------------------------------------------------------------ #
    def inferir_lote(self, df: pd.DataFrame, variables_activas=None) -> np.ndarray:
        """Posterior P(maliciosa | evidencia) para cada fila. Vectorizado."""
        self._validar_ajuste()
        v = list(variables_activas) if variables_activas is not None else self.variables_activas_
        faltantes = [x for x in v if x not in self.tablas_]
        if faltantes:
            raise ValueError(f"Variables no presentes en el ajuste: {faltantes}")
        X = _transformar_df(df, v)
        return self._sigmoide(self._log_odds(X, v))

    def ejecutar_inferencia(self, perfil_ip: dict, variables_activas):
        """Retorna (prior, posterior). Firma compatible con la versión original."""
        self._validar_ajuste()
        v = list(variables_activas)
        fila = _transformar_perfil(perfil_ip, v)
        X = pd.DataFrame([fila])
        lo = float(self._log_odds(X, v)[0])
        return self.prior_[1], float(self._sigmoide(lo))

    # ------------------------------------------------------------------ #
    def diagnostico(self, variables_activas=None) -> pd.DataFrame:
        """Verifica que cada variable defina una distribución normalizada."""
        self._validar_ajuste()
        v = list(variables_activas) if variables_activas is not None else self.variables_activas_
        filas = []
        for var in v:
            tabla = self.tablas_[var]
            for clase in (1, 0):
                niveles = [n for n in tabla[clase] if n != NIVEL_NO_VISTO]
                masa = sum(np.exp(tabla[clase][n]) for n in niveles)
                masa += np.exp(tabla[clase][NIVEL_NO_VISTO]) * (self.k_[var] - len(niveles))
                filas.append({"variable": var, "clase": clase,
                              "K": self.k_[var], "masa_total": masa})
        return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# Interfaz a nivel de módulo — RETROCOMPATIBLE con 1_📡_Analizador.py
# --------------------------------------------------------------------------- #
RUTA_CORPUS = "data/BID_dataset.csv"
_MOTOR_GLOBAL = None
_FIRMA_GLOBAL = None


def _obtener_motor(df, variables_activas):
    """Motor singleton ajustado con el corpus completo (modo producción)."""
    global _MOTOR_GLOBAL, _FIRMA_GLOBAL
    firma = (len(df), tuple(variables_activas))
    if _MOTOR_GLOBAL is None or _FIRMA_GLOBAL != firma:
        _MOTOR_GLOBAL = MotorBayesiano().fit(df, variables_activas)
        _FIRMA_GLOBAL = firma
    return _MOTOR_GLOBAL


def ejecutar_inferencia(perfil_ip, variables_activas, df=None):
    """
    Retorna (prior, posterior) para una IP.

    Conserva la firma original. El corpus se lee una sola vez y el motor
    queda cacheado, en lugar de releer el CSV en cada llamada.
    """
    if df is None:
        df = pd.read_csv(RUTA_CORPUS)
    motor = _obtener_motor(df, variables_activas)
    return motor.ejecutar_inferencia(perfil_ip, variables_activas)


def calcular_likelihood(df, variable, valor_api, is_malicious, alpha=ALPHA):
    """
    Verosimilitud suavizada P(valor | clase), con Laplace correcto.
    Se conserva por compatibilidad con scripts que la invoquen directamente.
    """
    y = _serie_target(df)
    clase = int(is_malicious)
    n_c = int((y == clase).sum())
    if n_c == 0:
        return alpha / (alpha * 2)

    col = _transformar_variable(df[variable], variable)
    if variable in VARIABLES_RAZON or variable in CENTINELAS:
        K = 2
    else:
        K = col.nunique() + 1

    nivel = _transformar_perfil({variable: valor_api}, [variable])[variable]
    casos = int((col[y == clase] == nivel).sum())
    return (casos + alpha) / (n_c + alpha * K)


def reiniciar_cache():
    """Invalida el motor cacheado (útil tras reentrenar o actualizar el corpus)."""
    global _MOTOR_GLOBAL, _FIRMA_GLOBAL
    _MOTOR_GLOBAL, _FIRMA_GLOBAL = None, None
