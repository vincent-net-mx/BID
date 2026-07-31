#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auditor_bid.py — Auditor forense del dataset BID (Bad IP Detector).

No asume nada: lee el CSV real y contrasta, una por una, las afirmaciones
numericas publicadas en el articulo. Cada verificacion imprime el valor
declarado, el valor observado y un veredicto PASS / FAIL.

Uso:
    python auditor_bid.py --csv dataset.csv
    python auditor_bid.py --csv dataset.csv --out auditoria.md
    python auditor_bid.py --csv dataset.csv --col-score abuseip_score --col-target is_malicious

Requiere: pandas, numpy
"""

import argparse
import io
import json
import sys

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("Falta pandas/numpy. Instala con: pip install pandas numpy")


# --------------------------------------------------------------------------
# AFIRMACIONES DEL ARTICULO (editar aqui si cambian las cifras del paper)
# --------------------------------------------------------------------------
CLAIMS = {
    "total_registros": 970,
    "n_benignas": 780,
    "n_maliciosas": 190,
    "score_media": 19.54,
    "score_std": 39.63,
    "pct_score_cero_global": 75.0,          # "75% ... remains at a zero-abuse level"
    "maliciosas_rango_alto": (99.0, 100.0),  # "abrupt jump toward maximum values"
    "benignas_con_score_positivo": 6,        # "exactly 6 potential false positive cases"
    "maliciosas_score_cero": 186,            # "abuseip_score = 0, n = 186 de 190"
    "prob_base_maliciosa": 19.58,            # %
    "prob_datacenter": 30.65,                # %
    "recall_apt_catboost": 83.87,            # %
    "recall_apt_baseline": 85.48,            # %
}

# Nombres alternativos aceptados por columna (el CSV puede venir en ES o EN)
COLUMN_CANDIDATES = {
    "score": ["abuseip_score", "abuseipdb_score", "abuse_score", "score",
              "puntaje_abuseip", "puntuacion_abuse", "abuseConfidenceScore"],
    "target": ["is_malicious", "es_maliciosa", "malicious", "label", "target",
               "y", "clase", "etiqueta"],
    "ip": ["ip", "ip_address", "direccion_ip", "ipv4"],
    "usage": ["usage_type", "infrastructure", "infraestructura", "tipo_uso",
              "usageType", "infra_tipo"],
    "reports": ["abuseip_total_reports", "total_reports", "num_reports",
                "reportes", "totalReports"],
    "pred_catboost": ["pred_catboost", "y_pred_catboost", "prediccion_catboost",
                      "pred_cb", "y_pred"],
    "pred_baseline": ["pred_lr", "pred_baseline", "y_pred_lr",
                      "prediccion_bayes", "pred_bayes"],
}

TOL_PCT = 0.5   # tolerancia absoluta en puntos porcentuales
TOL_ABS = 0.05  # tolerancia absoluta para media / desviacion


# --------------------------------------------------------------------------
# Carga robusta
# --------------------------------------------------------------------------
def cargar_csv(ruta):
    """Intenta varias combinaciones de separador y codificacion."""
    intentos = [
        {"sep": None, "engine": "python", "encoding": "utf-8"},
        {"sep": None, "engine": "python", "encoding": "latin-1"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ";", "encoding": "latin-1"},
    ]
    ultimo_error = None
    for kw in intentos:
        try:
            df = pd.read_csv(ruta, **kw)
            if df.shape[1] > 1:
                return df
        except Exception as exc:  # noqa: BLE001
            ultimo_error = exc
    sys.exit(f"No se pudo leer el CSV: {ultimo_error}")


def resolver(df, clave, override=None):
    """Devuelve el nombre real de la columna o None."""
    if override:
        if override in df.columns:
            return override
        sys.exit(f"La columna indicada '{override}' no existe. "
                 f"Disponibles: {list(df.columns)}")
    normalizadas = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    for cand in COLUMN_CANDIDATES[clave]:
        real = normalizadas.get(cand.strip().lower().replace(" ", "_"))
        if real is not None:
            return real
    return None


def a_binario(serie):
    """Normaliza la etiqueta objetivo a 0/1."""
    if serie.dtype.kind in "biufc":
        return (serie.astype(float) > 0.5).astype(int)
    mapa = {"1": 1, "true": 1, "si": 1, "sí": 1, "yes": 1, "malicious": 1,
            "maliciosa": 1, "malicioso": 1, "bad": 1,
            "0": 0, "false": 0, "no": 0, "benign": 0, "benigna": 0,
            "benigno": 0, "good": 0}
    return serie.astype(str).str.strip().str.lower().map(mapa).fillna(0).astype(int)


# --------------------------------------------------------------------------
# Motor de verificacion
# --------------------------------------------------------------------------
class Auditoria:
    def __init__(self):
        self.filas = []

    def check(self, nombre, declarado, observado, ok, nota=""):
        self.filas.append({
            "verificacion": nombre,
            "declarado": declarado,
            "observado": observado,
            "veredicto": "PASS" if ok else "FAIL",
            "nota": nota,
        })

    def info(self, nombre, valor, nota=""):
        self.filas.append({
            "verificacion": nombre,
            "declarado": "-",
            "observado": valor,
            "veredicto": "INFO",
            "nota": nota,
        })

    def imprimir(self, buf):
        anchos = [54, 18, 30, 9]
        cab = ("VERIFICACION".ljust(anchos[0]) + "DECLARADO".ljust(anchos[1])
               + "OBSERVADO".ljust(anchos[2]) + "VEREDICTO")
        print(cab, file=buf)
        print("-" * (sum(anchos) + 2), file=buf)
        for f in self.filas:
            print(str(f["verificacion"])[:anchos[0] - 1].ljust(anchos[0])
                  + str(f["declarado"])[:anchos[1] - 1].ljust(anchos[1])
                  + str(f["observado"])[:anchos[2] - 1].ljust(anchos[2])
                  + f["veredicto"], file=buf)
            if f["nota"]:
                print("    -> " + f["nota"], file=buf)
        fails = sum(1 for f in self.filas if f["veredicto"] == "FAIL")
        print("-" * (sum(anchos) + 2), file=buf)
        print(f"Total verificaciones: "
              f"{sum(1 for f in self.filas if f['veredicto'] != 'INFO')} | "
              f"FAIL: {fails}", file=buf)
        return fails


def auditar(df, cols, aud, buf):
    c_score = cols["score"]
    c_target = cols["target"]

    n = len(df)
    aud.check("Total de registros", CLAIMS["total_registros"], n,
              n == CLAIMS["total_registros"])

    y = a_binario(df[c_target])
    n_mal = int((y == 1).sum())
    n_ben = int((y == 0).sum())
    aud.check("Registros benignos", CLAIMS["n_benignas"], n_ben,
              n_ben == CLAIMS["n_benignas"])
    aud.check("Registros maliciosos", CLAIMS["n_maliciosas"], n_mal,
              n_mal == CLAIMS["n_maliciosas"])

    score = pd.to_numeric(df[c_score], errors="coerce")
    nulos = int(score.isna().sum())
    if nulos:
        aud.info("Valores nulos en score", nulos,
                 "Se excluyen de media/std; se cuentan aparte en el bloque de cero.")

    media = float(score.mean())
    desv = float(score.std(ddof=1))
    aud.check("Media de abuseip_score", CLAIMS["score_media"], round(media, 4),
              abs(media - CLAIMS["score_media"]) <= TOL_ABS)
    aud.check("Desv. estandar de abuseip_score", CLAIMS["score_std"],
              round(desv, 4), abs(desv - CLAIMS["score_std"]) <= TOL_ABS)

    # --- El nucleo de la inconsistencia -----------------------------------
    cero_global = int((score == 0).sum())
    pct_cero = 100.0 * cero_global / n
    aud.check("% global con score = 0", f"{CLAIMS['pct_score_cero_global']}%",
              f"{pct_cero:.2f}% ({cero_global}/{n})",
              abs(pct_cero - CLAIMS["pct_score_cero_global"]) <= TOL_PCT)

    ben_pos = int(((y == 0) & (score > 0)).sum())
    aud.check("Benignas con score > 0 (falsos positivos)",
              CLAIMS["benignas_con_score_positivo"], ben_pos,
              ben_pos == CLAIMS["benignas_con_score_positivo"])

    mal_cero = int(((y == 1) & (score == 0)).sum())
    aud.check("Maliciosas con score = 0 (escenario APT)",
              f"{CLAIMS['maliciosas_score_cero']}/{CLAIMS['n_maliciosas']}",
              f"{mal_cero}/{n_mal}",
              mal_cero == CLAIMS["maliciosas_score_cero"])

    lo, hi = CLAIMS["maliciosas_rango_alto"]
    mal_alto = int(((y == 1) & (score >= lo) & (score <= hi)).sum())
    pct_mal_alto = 100.0 * mal_alto / n_mal if n_mal else 0.0
    aud.info(f"Maliciosas con score en [{lo}, {hi}]",
             f"{mal_alto}/{n_mal} ({pct_mal_alto:.2f}%)",
             "El articulo afirma que el segmento malicioso salta a valores maximos.")

    # --- Prueba de coherencia interna -------------------------------------
    # Si N maliciosas estan en 0 y solo 6 benignas superan 0, la media global
    # esta acotada superiormente por 100*(4+6)/970. Se contrasta contra la
    # media observada para detectar afirmaciones mutuamente excluyentes.
    max_media_si_186 = 100.0 * ((CLAIMS["n_maliciosas"] - CLAIMS["maliciosas_score_cero"])
                                + CLAIMS["benignas_con_score_positivo"]) / CLAIMS["total_registros"]
    coherente = media <= max_media_si_186 + TOL_ABS
    aud.check("Coherencia: media compatible con n=186 en cero",
              f"<= {max_media_si_186:.2f}", round(media, 4), coherente,
              "Si 186/190 maliciosas estuvieran en 0 y solo 6 benignas fueran > 0, "
              "la media global no podria superar ese techo aritmetico.")

    # Reconstruccion de la fraccion implicada por la media observada
    frac_implicada = media / 100.0
    aud.info("Registros en score ~100 implicados por la media",
             f"~{frac_implicada * n:.0f} de {n} ({frac_implicada * 100:.2f}%)",
             "Estimacion bajo distribucion bimodal 0/100.")

    # --- Definiciones alternativas de 'reputacion cero' -------------------
    # El 186 puede provenir de otra columna. Se prueban las alternativas.
    aud.info("--- Origenes alternativos del n = 186 ---", "", "")
    aud.info("Maliciosas con score nulo/NaN",
             int(((y == 1) & (score.isna())).sum()), "")
    aud.info("Maliciosas con score = 0 o nulo",
             int(((y == 1) & ((score == 0) | (score.isna()))).sum()), "")
    if cols.get("reports"):
        rep = pd.to_numeric(df[cols["reports"]], errors="coerce")
        aud.info(f"Maliciosas con {cols['reports']} = 0",
                 int(((y == 1) & (rep == 0)).sum()),
                 "Candidato si 'reputacion cero' se definio por reportes, no por score.")
    aud.info("Maliciosas con score < 1",
             int(((y == 1) & (score < 1)).sum()), "")
    aud.info("Maliciosas con score < 50",
             int(((y == 1) & (score < 50)).sum()), "")

    # --- Probabilidades bayesianas ----------------------------------------
    p_base = 100.0 * n_mal / n
    aud.check("P(maliciosa) base", f"{CLAIMS['prob_base_maliciosa']}%",
              f"{p_base:.2f}%",
              abs(p_base - CLAIMS["prob_base_maliciosa"]) <= TOL_PCT)

    if cols.get("usage"):
        u = df[cols["usage"]].astype(str).str.lower()
        mask = u.str.contains("data center", na=False) | u.str.contains("hosting", na=False)
        if mask.sum():
            p_dc = 100.0 * float(y[mask].mean())
            aud.check("P(maliciosa | Data Center/Hosting)",
                      f"{CLAIMS['prob_datacenter']}%",
                      f"{p_dc:.2f}% (n={int(mask.sum())})",
                      abs(p_dc - CLAIMS["prob_datacenter"]) <= TOL_PCT)
        else:
            aud.info("P(maliciosa | Data Center/Hosting)", "sin coincidencias",
                     f"Revisa los valores de '{cols['usage']}'.")

    # --- Recall en el subconjunto APT (si hay predicciones) ---------------
    subset = (y == 1) & (score == 0)
    for etiqueta, clave, declarado in (
        ("CatBoost", "pred_catboost", CLAIMS["recall_apt_catboost"]),
        ("Baseline LR/Bayes", "pred_baseline", CLAIMS["recall_apt_baseline"]),
    ):
        col = cols.get(clave)
        if col and subset.sum():
            pred = a_binario(df.loc[subset, col])
            recall = 100.0 * float(pred.mean())
            aud.check(f"Recall APT — {etiqueta}", f"{declarado}%",
                      f"{recall:.2f}% ({int(pred.sum())}/{int(subset.sum())})",
                      abs(recall - declarado) <= TOL_PCT)
        elif col:
            aud.info(f"Recall APT — {etiqueta}", "subconjunto vacio",
                     "No hay maliciosas con score = 0 en el CSV.")

    # --- Consistencia del denominador del recall --------------------------
    # 83.87% y 85.48% solo son exactos sobre un denominador de 186.
    for etiqueta, pct in (("CatBoost", CLAIMS["recall_apt_catboost"]),
                          ("Baseline", CLAIMS["recall_apt_baseline"])):
        aciertos = pct / 100.0 * CLAIMS["maliciosas_score_cero"]
        aud.info(f"Denominador implicado por recall {etiqueta}",
                 f"{pct}% x 186 = {aciertos:.2f}",
                 "Un entero exacto confirma que el 186 se calculo sobre datos reales.")

    print(file=buf)


def main():
    ap = argparse.ArgumentParser(
        description="Audita las afirmaciones numericas del articulo BID contra el CSV real.")
    ap.add_argument("--csv", required=True, help="Ruta al dataset CSV.")
    ap.add_argument("--col-score", default=None, help="Nombre de la columna de score.")
    ap.add_argument("--col-target", default=None, help="Nombre de la columna objetivo.")
    ap.add_argument("--out", default=None, help="Ruta opcional para guardar el reporte.")
    ap.add_argument("--json", default=None, help="Ruta opcional para volcar resultados en JSON.")
    args = ap.parse_args()

    df = cargar_csv(args.csv)

    cols = {
        "score": resolver(df, "score", args.col_score),
        "target": resolver(df, "target", args.col_target),
        "ip": resolver(df, "ip"),
        "usage": resolver(df, "usage"),
        "reports": resolver(df, "reports"),
        "pred_catboost": resolver(df, "pred_catboost"),
        "pred_baseline": resolver(df, "pred_baseline"),
    }
    if not cols["score"] or not cols["target"]:
        sys.exit("No se localizaron las columnas de score y/o etiqueta. "
                 f"Columnas disponibles: {list(df.columns)}\n"
                 "Usa --col-score y --col-target para indicarlas.")

    buf = io.StringIO()
    print("=" * 92, file=buf)
    print("AUDITORIA NUMERICA — Bad IP Detector (BID)", file=buf)
    print(f"Archivo: {args.csv}   |   Filas: {len(df)}   |   Columnas: {len(df.columns)}", file=buf)
    print(f"Columna score: {cols['score']}   |   Columna objetivo: {cols['target']}", file=buf)
    print("=" * 92, file=buf)
    print(file=buf)

    aud = Auditoria()
    auditar(df, cols, aud, buf)
    fails = aud.imprimir(buf)

    print(file=buf)
    print("LECTURA: cada FAIL es una cifra del articulo que el dataset no respalda.", file=buf)
    print("Corrige el texto para que coincida con la columna 'OBSERVADO'.", file=buf)

    reporte = buf.getvalue()
    print(reporte)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("```\n" + reporte + "```\n")
        print(f"[+] Reporte guardado en {args.out}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(aud.filas, fh, ensure_ascii=False, indent=2)
        print(f"[+] JSON guardado en {args.json}")

    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
