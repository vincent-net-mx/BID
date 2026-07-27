import catboost as cb
import pandas as pd
from datetime import datetime
import motor_bayesiano as bayes # El nombre solicitado

def ejecutar_prediccion_hibrida(perfil_limpio, modelo_path, verbose=True):
    """
    Motor de inferencia pura. Recibe datos ya normalizados y retorna 
    el veredicto final combinando Bayes e IA.
    """
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    vprint(f"\n" + "="*60)
    vprint(f"SISTEMA BID: MOTOR DE INFERENCIA (STEP 4)")
    vprint("="*60)

    # 1. Inferencia Bayesiana (Contrapeso Estadístico)
    vprint("[1/3] Calculando Riesgo Bayesiano...")
    v_activas = [
        'abuseip_score', 'usage_type', 'abuseip_categories', 
        'abuseip_distinct_users', 'abuseip_last_reported', 
        'country', 'asn', 'isp', 'infra_owner'
    ]
    
    # Obtenemos la probabilidad posterior del motor bayesiano
    _, p_bayesian = bayes.ejecutar_inferencia(perfil_limpio, v_activas)

    # 2. Ingeniería de Características "En Vuelo"
    vprint("[2/3] Preparando vector de características para CatBoost...")
    
    # Creamos el DataFrame base
    df_input = pd.DataFrame([perfil_limpio])
    df_input['bayesian_risk'] = p_bayesian
    
    # Transformación de tiempo (Días desde el reporte)
    ahora = datetime.now()
    def calc_days_live(d):
        if d == 'Never_Reported' or pd.isna(d):
            return 3650
        try:
            return max(0, (ahora - pd.to_datetime(d).tz_localize(None)).days)
        except:
            return 3650

    df_input['days_since_last_report'] = df_input['abuseip_last_reported'].apply(calc_days_live)
    
    # Transformación de categorías (Conteo numérico)
    df_input['attack_type_count'] = df_input['abuseip_categories'].apply(
        lambda x: 0 if x == 'No_Reports' or pd.isna(x) else len(str(x).split(','))
    )

    # Definición y orden estricto de columnas (Paridad con el entrenamiento)
    features = [
        'abuseip_score', 'usage_type', 'attack_type_count', 
        'abuseip_distinct_users', 'days_since_last_report', 
        'country', 'asn', 'isp', 'infra_owner', 'bayesian_risk'
    ]
    
    X_pred = df_input[features].copy()
    
    # Asegurar tipos de datos para CatBoost
    cat_features = ['usage_type', 'country', 'asn', 'isp', 'infra_owner']
    for col in cat_features:
        X_pred[col] = X_pred[col].fillna('unknown').astype(str)

    num_cols = ['abuseip_score', 'attack_type_count', 'abuseip_distinct_users', 'days_since_last_report', 'bayesian_risk']
    for col in num_cols:
         X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce').fillna(0)

    # 3. Clasificación Final
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
    vprint("="*60 + "\n")

    # Retorno compatible con 1_📡_Analizador.py
    return {
        "veredicto": veredicto,
        "probabilidad_ia": prob_malicia,
        "riesgo_bayesiano": p_bayesian,
        "isp_normalizado": perfil_limpio.get('isp'),
        "timestamp": ahora.strftime("%Y-%m-%d %H:%M:%S")
    }