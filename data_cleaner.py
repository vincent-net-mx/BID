import pandas as pd
import os
import re

# =====================================================================
# PRE-COMPILACIÓN DE REGEX (Optimización de rendimiento para la Fase 2)
# =====================================================================
# Destrucción de sufijos legales
REGEX_SUFFIXES = re.compile(r'\b(Inc|LLC|Ltd|Corporation|Corp|Co|Pty|GmbH|SA|S\.A\.|AB|SpA|B\.V\.|Limited|Company|JSC|Ltda|SRL|Public|Enterprise|Holdings|Private|PJSC|PVT|OJSC|PLC|SRO|SIA|Joint|Stock|Group|Grup|BV|NV|E\.S\.P)\.?\b', flags=re.IGNORECASE)

# Destrucción de términos genéricos
REGEX_TRASH = re.compile(r'\b(Network|Networks|Hosting|Host|Technologies|Technology|Tech|Communications|Communication|Telecom|Telecommunications|Telecomunicacoes|Services|Service|Broadband|Internet|Systems|System|Solutions|Data|Center|Centers|Cloud|Computing|Online|Cable|\.com|\.net)\b', flags=re.IGNORECASE)

# =====================================================================
# DICCIONARIOS DE MAPEO
# =====================================================================
# Mapeo de gigantes tecnológicos
TECH_GIANTS = {
    'google': 'Google', 'amazon': 'Amazon', 'aws': 'Amazon', 'microsoft': 'Microsoft',
    'cloudflare': 'Cloudflare', 'alibaba': 'Alibaba', 'aliyun': 'Alibaba', 'tencent': 'Tencent',
    'digitalocean': 'DigitalOcean', 'digital ocean': 'DigitalOcean', 'ovh': 'OVH',
    'hetzner': 'Hetzner', 'linode': 'Linode', 'akamai': 'Akamai', 'fastly': 'Fastly',
    'm247': 'M247', 'orange': 'Orange', 'france telecom': 'Orange', 'bouygues': 'Bouygues',
    'bezeq': 'Bezeq', 'columbus': 'Columbus', 'upc': 'UPC', 'megafon': 'MegaFon',
    'bharti': 'Airtel', 'airtel': 'Airtel', 'at&t': 'AT&T', 'bellsouth': 'AT&T',
    'china telecom': 'China Telecom', 'chinanet': 'China Telecom', 'china unicom': 'China Unicom',
    'china mobile': 'China Mobile', 'vodafone': 'Vodafone', 'liberty': 'Liberty',
    'telefónica': 'Telefonica', 'telefonica': 'Telefonica', 'telmex': 'Telmex',
    'godaddy': 'GoDaddy', 'claro': 'Claro'
}

# Mapeo de normalización de países (Todo en minúsculas para coincidencia exacta)
COUNTRY_MAPPING = {
    'the netherlands': 'Netherlands',
    'united states of america': 'United States',
    'usa': 'United States',
    'us': 'United States',
    'united kingdom': 'United Kingdom',
    'uk': 'United Kingdom',
    'great britain': 'United Kingdom',
    'russian federation': 'Russia',
    'korea, republic of': 'South Korea',
    'republic of korea': 'South Korea',
    'korea, south': 'South Korea',
    'viet nam': 'Vietnam',
    'macao': 'Macau',
    'taiwan, province of china': 'Taiwan',
    'bolivia (plurinational state of)': 'Bolivia',
    'venezuela (bolivarian republic of)': 'Venezuela',
    'iran (islamic republic of)': 'Iran',
    'syrian arab republic': 'Syria',
    'tanzania, united republic of': 'Tanzania',
    'czechia': 'Czech Republic'
}

# =====================================================================
# FUNCIONES DE APOYO Y FASES
# =====================================================================

def obtener_mas_largo(serie):
    """Filtra nulos y Unknowns, y devuelve el string con mayor longitud."""
    valores_validos = [str(v).strip() for v in serie if pd.notna(v) and str(v).strip() not in ['Unknown', '', 'nan']]
    if not valores_validos:
        return 'Unknown'
    return max(set(valores_validos), key=len)

def clean_corporate_name(name):
    """Aplica la normalización corporativa general (Fase 2)."""
    if pd.isna(name) or str(name).strip() == 'Unknown': 
        return 'Unknown'
    
    name_str = str(name).strip()
    name_lower = name_str.lower()
    
    # 1. Capa de Gigantes Tecnológicos
    for key, clean_name in TECH_GIANTS.items():
        if key in name_lower:
            return clean_name
            
    # 2. Capa de Destrucción de Sufijos y Basura (Usando Regex pre-compilado)
    name_str = REGEX_SUFFIXES.sub('', name_str)
    name_str = REGEX_TRASH.sub('', name_str)
    
    # 3. Limpieza de caracteres residuales en los bordes y dobles espacios
    name_str = re.sub(r'^[\W_]+|[\W_]+$', '', name_str)
    name_str = re.sub(r'\s+', ' ', name_str).strip()
    
    return name_str if name_str else 'Unknown'

def clean_country_name(name):
    """Normaliza el nombre de un país utilizando el diccionario de mapeo."""
    if pd.isna(name) or str(name).strip() == 'Unknown':
        return 'Unknown'
    
    name_str = str(name).strip()
    name_lower = name_str.lower()
    
    # Si está en nuestro diccionario, lo reemplazamos por la versión oficial
    if name_lower in COUNTRY_MAPPING:
        return COUNTRY_MAPPING[name_lower]
    
    # Si no, simplemente devolvemos el país limpio y con formato Título (ej: "argentina" -> "Argentina")
    # Nota: .title() capitaliza la primera letra de cada palabra
    return name_str.title()

def fase_1_consolidar_asn(df):
    """Consolida las columnas isp e infra_owner asignando el nombre más largo por ASN."""
    print("[Fase 1] Consolidación Técnica por ASN...")
    
    etiquetas_excluidas = ['Unknown_ASN', 'Private_IP', 'API_Fail', 'Network_Failure', 'Unknown', 'Error']
    mascara_validos = ~df['ASN'].isin(etiquetas_excluidas) & df['ASN'].notna()
    
    for col in ['isp', 'infra_owner']:
        if col in df.columns:
            # Crea un diccionario {ASN: Nombre_Mas_Largo}
            mapeo_ganadores = df[mascara_validos].groupby('ASN')[col].apply(obtener_mas_largo).to_dict()
            
            # Sobrescribe los valores usando el mapeo
            df.loc[mascara_validos, col] = df.loc[mascara_validos, 'ASN'].map(mapeo_ganadores)
            print(f"  -> '{col}' consolidada con éxito.")
    return df

def fase_2_limpieza_general(df):
    """Aplica la poda de sufijos y normalización a todo el dataset."""
    print("[Fase 2] Normalización Corporativa General...")
    
    for col in ['isp', 'infra_owner']:
        if col in df.columns:
            df[col] = df[col].apply(clean_corporate_name)
            print(f"  -> '{col}' normalizada con éxito.")
    return df

def fase_3_uniformizar_paises(df):
    """Busca la columna de país y aplica la uniformización."""
    print("[Fase 3] Uniformización de Países...")
    
    # Lista de posibles nombres que podría tener la columna en tu CSV
    posibles_nombres_columna = ['country', 'Country', 'pais', 'país', 'Pais', 'País', 'country_name']
    
    # Encontrar qué columna de país existe en el DataFrame
    col_pais = next((col for col in posibles_nombres_columna if col in df.columns), None)
    
    if col_pais:
        # Llenar nulos antes de procesar
        df[col_pais] = df[col_pais].fillna('Unknown')
        df[col_pais] = df[col_pais].apply(clean_country_name)
        print(f"  -> Columna '{col_pais}' normalizada con éxito.")
    else:
        print("  -> No se detectó una columna de país conocida. (Buscando: 'country', 'pais', etc.) Saltando fase.")
        
    return df

# =====================================================================
# EJECUCIÓN PRINCIPAL
# =====================================================================

def principal():
    ruta_entrada = 'BID_dataset_con_asn.csv'
    ruta_salida = 'BID_dataset_final_limpio.csv'

    if not os.path.exists(ruta_entrada):
        print(f"Error: No se encontró el archivo '{ruta_entrada}'. Verifica la ruta.")
        return

    try:
        print(f"Cargando dataset: {ruta_entrada}\n" + "-"*40)
        df = pd.read_csv(ruta_entrada)
        
        # Llenar nulos iniciales en columnas clave para evitar fallos de Pandas
        for col in ['isp', 'infra_owner']:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')

        # Ejecutar el Flujo de Trabajo
        df = fase_1_consolidar_asn(df)
        print("-" * 40)
        df = fase_2_limpieza_general(df)
        print("-" * 40)
        df = fase_3_uniformizar_paises(df)
        print("-" * 40)
        
        # Guardado final
        df.to_csv(ruta_salida, index=False)
        print(f"¡Proceso completado! Archivo final guardado en: {ruta_salida}")

    except Exception as e:
        print(f"Error crítico durante el procesamiento: {e}")

if __name__ == '__main__':
    # Asegurar que el directorio 'data' exista (si se usa de forma local sin la carpeta creada)
    os.makedirs('data', exist_ok=True)
    principal()