import pandas as pd
import numpy as np
import json

# ==========================================
# 1. CONFIGURACIÓN DE WHITELISTS
# ==========================================
# Mapeo de categorías estadísticas a los nombres de columnas permitidas
# basándonos en la naturaleza (cruda) de las variables de tu sistema BID.
WHITELISTS = {
    'tendencia_central': [
        'abuseip_score', 'abuseip_distinct_users', # Cuantitativas (Soportan Media, Mediana, Moda)
        'country', 'asn', 'isp', 'usage_type', 'infra_owner', 'abuseip_categories', 'abuseip_last_reported' # Cualitativas (Solo Moda)
    ],
    'dispersion': [
        'abuseip_score', 'abuseip_distinct_users' # Solo variables cuantitativas continuas/discretas
    ],
    'posicion': [
        'abuseip_score', 'abuseip_distinct_users' # Solo variables donde el orden y magnitud tienen sentido
    ]
}

# ==========================================
# 2. LÓGICA DE PROCESAMIENTO MODULAR
# ==========================================
def calcular_tendencia_central(serie):
    """Calcula métricas de tendencia central adaptándose al tipo de dato."""
    resultados = {}
    
    # La moda aplica para cualquier tipo de dato (incluso texto)
    modas = serie.mode()
    resultados['moda'] = modas.tolist() if not modas.empty else None
    
    # Media y Mediana solo si la serie es numérica
    if pd.api.types.is_numeric_dtype(serie):
        resultados['media'] = float(serie.mean())
        resultados['mediana'] = float(serie.median())
    else:
        resultados['media'] = "No aplica (No numérico)"
        resultados['mediana'] = "No aplica (No numérico)"
        
    return resultados

def calcular_dispersion(serie):
    """Calcula métricas de dispersión sobre series numéricas."""
    if not pd.api.types.is_numeric_dtype(serie):
        return {"error": "Métricas de dispersión requieren datos numéricos"}
    
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    
    return {
        'rango': float(serie.max() - serie.min()),
        'varianza': float(serie.var()),
        'desviacion_estandar': float(serie.std()),
        'iqr': float(q3 - q1)
    }

def calcular_posicion(serie):
    """Calcula métricas de posición relativa (cuartiles, deciles, percentiles)."""
    if not pd.api.types.is_numeric_dtype(serie):
        return {"error": "Métricas de posición requieren datos numéricos"}
    
    return {
        'cuartiles': {
            'Q1_25%': float(serie.quantile(0.25)),
            'Q2_50%': float(serie.quantile(0.50)),
            'Q3_75%': float(serie.quantile(0.75))
        },
        'deciles': {f'D{i}_%{(i)*10}': float(serie.quantile(i/10.0)) for i in range(1, 10)},
        'percentiles_criticos': {
            'P90': float(serie.quantile(0.90)),
            'P95': float(serie.quantile(0.95)),
            'P99': float(serie.quantile(0.99)) # Útil para detectar ataques extremos
        }
    }

# ==========================================
# 3. FUNCIÓN PRINCIPAL DE ANÁLISIS
# ==========================================
def analizar_columna(nombre_columna, serie_datos, medidas_solicitadas):
    """
    Recibe el nombre de la columna, sus datos reales y las medidas a calcular.
    Verifica contra la whitelist y retorna un diccionario con los resultados.
    """
    reporte_columna = {}
    
    # Limpieza básica: Eliminar nulos (NaN) para no sesgar los cálculos estadísticos
    serie_limpia = serie_datos.dropna()
    
    if serie_limpia.empty:
        return {"advertencia": "La columna está completamente vacía o contiene solo valores nulos."}

    for medida in medidas_solicitadas:
        if nombre_columna not in WHITELISTS.get(medida, []):
            reporte_columna[medida] = {"advertencia": f"Columna '{nombre_columna}' no autorizada en whitelist para '{medida}'."}
            continue
            
        try:
            if medida == 'tendencia_central':
                reporte_columna[medida] = calcular_tendencia_central(serie_limpia)
            elif medida == 'dispersion':
                # Convertimos a numérico forzosamente en caso de que vengan como strings ('80', '100')
                serie_num = pd.to_numeric(serie_limpia, errors='coerce').dropna()
                reporte_columna[medida] = calcular_dispersion(serie_num)
            elif medida == 'posicion':
                serie_num = pd.to_numeric(serie_limpia, errors='coerce').dropna()
                reporte_columna[medida] = calcular_posicion(serie_num)
        except Exception as e:
            reporte_columna[medida] = {"error_ejecucion": str(e)}
            
    return reporte_columna

# ==========================================
# 4. ORQUESTADOR Y GENERADOR DE REPORTE
# ==========================================
def generar_reporte_estadistico(ruta_csv, medidas_solicitadas=['tendencia_central', 'dispersion', 'posicion']):
    """Itera sobre las columnas del CSV y genera el JSON estructurado."""
    try:
        # Leer el dataset crudo
        df = pd.read_csv(ruta_csv, low_memory=False)
        reporte_global = {}
        
        # Iterar sobre todas las columnas del dataset
        for columna in df.columns:
            # Solo analizamos si la columna existe en al menos una whitelist
            if any(columna in sublista for sublista in WHITELISTS.values()):
                reporte_global[columna] = analizar_columna(columna, df[columna], medidas_solicitadas)
            else:
                reporte_global[columna] = {"advertencia": "Columna ignorada (No registrada en ninguna whitelist)"}
                
        return reporte_global
        
    except FileNotFoundError:
        return {"error": f"No se encontró el archivo en la ruta: {ruta_csv}"}
    except Exception as e:
        return {"error": f"Fallo crítico al leer el archivo: {str(e)}"}

# ==========================================
# BLOQUE DE EJECUCIÓN (PRUEBA)
# ==========================================
if __name__ == "__main__":
    # Simulación de uso en tu entorno local. 
    # Cambia "data/BID_dataset.csv" por la ruta real de tu archivo crudo.
    ruta_archivo = "BID_dataset.csv"
    
    # Como no tenemos el archivo real aquí, puedes probar creando un DF falso para probar el script:
    """
    df_prueba = pd.DataFrame({
        'abuseip_score': [0, 15, 100, 100, 80, 0, np.nan, 45],
        'country': ['MX', 'US', 'US', 'CN', 'RU', 'MX', 'MX', 'US'],
        'columna_desconocida': [1, 2, 3, 4, 5, 6, 7, 8]
    })
    df_prueba.to_csv("prueba_cruda.csv", index=False)
    ruta_archivo = "prueba_cruda.csv"
    """
    
    # Generamos el reporte solicitando las tres categorías estadísticas
    reporte = generar_reporte_estadistico(ruta_archivo)
    
    # Imprimir en formato JSON legible para análisis o exportación
    print(json.dumps(reporte, indent=4, ensure_ascii=False))