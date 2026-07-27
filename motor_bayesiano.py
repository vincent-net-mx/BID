import pandas as pd

# Valores que se consideran como "ausencia de datos"
NULOS = ['Unknown', 'Error_Timeout', 'None', '', 'Fetch_Error', 'Empty_JSON']

def calcular_likelihood(df, variable, valor_api, is_malicious):
    """Calcula P(Variable | Maliciosa) o P(Variable | Benigna) dinámicamente."""
    subset = df[df['is_malicious'] == str(is_malicious)]
    total_clase = len(subset)
    
    # 1. Variables de Razón (Lógica Binaria: > 0 o == 0)
    if variable in ['abuseip_score', 'abuseip_distinct_users']:
        try:
            valor_num = float(valor_api)
        except:
            valor_num = 0.0
            
        if valor_num > 0:
            casos = len(subset[pd.to_numeric(subset[variable], errors='coerce').fillna(0) > 0])
        else:
            casos = len(subset[pd.to_numeric(subset[variable], errors='coerce').fillna(0) == 0])
            
    # 2. Variables de Existencia (Alineadas con tu CSV)
    elif variable in ['abuseip_categories', 'abuseip_last_reported']:
        if valor_api in ['No_Reports', 'Never_Reported']:
            casos = len(subset[subset[variable] == valor_api])
        else:
            # Si tiene cualquier otro texto, asumimos que sí tiene reportes
            casos = len(subset[~subset[variable].isin(['No_Reports', 'Never_Reported', 'Unknown', 'None', ''])])
            
    # 3. Variables Nominales Exactas (country, asn, isp, usage_type, infra_owner)
    else:
        casos = len(subset[subset[variable] == str(valor_api)])
        
    # --- SUAVIZADO DE LAPLACE UNIVERSAL ---
    # ESTO DEBE ESTAR ALINEADO A LA IZQUIERDA, FUERA DE TODOS LOS IF/ELSE
    if casos == 0:
        casos = 0.1 

    # Si por alguna razón extrema la clase está vacía, evitamos dividir por 0
    if total_clase == 0:
        return 0.1

    return casos / total_clase

# ---> AQUI ESTÁ EL CAMBIO CLAVE: Agregamos df=None <---
def ejecutar_inferencia(perfil_ip, variables_activas, df=None):
    """
    Recibe el diccionario con los datos de la IP y la lista de variables a usar.
    Retorna la probabilidad Bayesiana actualizada.
    """
    # Si el script externo (como el generador) no le pasa el DataFrame en memoria, lo lee.
    # Si sí se lo pasa, se salta esta lectura y el proceso vuela.
    if df is None:
        df = pd.read_csv("data/BID_dataset.csv", dtype=str)
    
    # Probabilidades Previas (Priors)
    total_ips = len(df)
    maliciosas = len(df[df['is_malicious'] == '1'])
    benignas = len(df[df['is_malicious'] == '0'])
    
    p_m = maliciosas / total_ips
    p_no_m = benignas / total_ips
    
    # Variables de Likelihood acumulado
    likelihood_m = 1.0
    likelihood_no_m = 1.0
    
    # Iterar solo sobre las variables que el analista dejó encendidas
    for var in variables_activas:
        valor_api = perfil_ip.get(var, 'Unknown')
        
        p_e_dado_m = calcular_likelihood(df, var, valor_api, 1)
        p_e_dado_no_m = calcular_likelihood(df, var, valor_api, 0)
        
        # Multiplicación sucesiva (Naive Bayes)
        likelihood_m *= p_e_dado_m
        likelihood_no_m *= p_e_dado_no_m
        
    # Teorema de Bayes
    numerador = likelihood_m * p_m
    denominador = numerador + (likelihood_no_m * p_no_m)
    
    # Prevención de división por cero
    if denominador == 0:
        return p_m, p_m
        
    p_posterior = numerador / denominador
    return p_m, p_posterior