import requests
import os
import streamlit as st
from dotenv import load_dotenv

# 1. Carga de entorno híbrida (Local .env + Streamlit Cloud)
load_dotenv()

def obtener_api_key():
    """
    Busca la API Key primero en variables de entorno (.env/Docker) 
    y como respaldo en st.secrets de Streamlit.
    """
    return os.getenv("ABUSEIP_API_KEY") or st.secrets.get("ABUSEIP_API_KEY")

def consultar_abuse_ip(ip_address):
    api_key = obtener_api_key()
    if not api_key:
        # Usamos st.error para la interfaz, pero el flujo de backend sigue igual
        st.error("⚠️ Error: No se encontró la ABUSEIP_API_KEY.")
        return None
        
    url = 'https://api.abuseipdb.com/api/v2/check'
    params = {
        'ipAddress': ip_address,
        'maxAgeInDays': '90',
        'verbose': True
    }
    headers = {
        'Accept': 'application/json',
        'Key': api_key
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()['data']
        else:
            print(f"Error API AbuseIPDB: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error de conexión: {e}")
        return None

def reportar_ip_abuse(ip, categorias_ids, comentario="Reportado desde BID UdeG"):
    """
    Mantiene la funcionalidad original de reportar IPs de consultas_api.py.
    """
    api_key = obtener_api_key()
    if not api_key:
        return {"error": True, "mensaje": "API Key no configurada"}

    url = 'https://api.abuseipdb.com/api/v2/report'
    params = {
        'ip': ip,
        'categories': ",".join(map(str, categorias_ids)),
        'comment': comentario
    }
    headers = {'Accept': 'application/json', 'Key': api_key}
    
    try:
        response = requests.post(url, headers=headers, data=params)
        return response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return {"error": True, "mensaje": str(e)}

def consultar_perfil_completo(ip):
    """
    Versión unificada: Lógica de api_requests.py adaptada para Streamlit.
    Mantiene los tipos de datos exactos (int para scores) y añade el ASN.
    """
    datos_abuse = consultar_abuse_ip(ip)
    if not datos_abuse:
        return None
        
    # Consulta Geográfica (IP-API)
    try:
        url_ip_api = f"http://ip-api.com/json/{ip}?fields=status,country,isp,org,as"
        res_geo = requests.get(url_ip_api, timeout=5)
        datos_geo = res_geo.json() if res_geo.status_code == 200 else {}
    except:
        datos_geo = {}

    # --- LÓGICA DE BACKEND DE API_REQUESTS.PY ---
    
    # Procesamiento de categorías: Devolver string separado por comas
    categorias = {str(c) for r in datos_abuse.get('reports', []) for c in r.get('categories', [])}
    
    # Cálculo de ASN: Lógica robusta de api_requests.py
    asn_raw = datos_abuse.get('asNumber')
    asn_final = f"AS{asn_raw}" if asn_raw else datos_geo.get('as', 'Unknown').split(' ')[0]

    # Retorno con tipos de datos consistentes para el Motor Bayesiano
    return {
        'abuseip_score': int(datos_abuse.get('abuseConfidenceScore', 0)), # Entero
        'usage_type': str(datos_abuse.get('usageType', 'Unknown')),
        'abuseip_categories': ", ".join(categorias) if categorias else 'No_Reports', # Formato CSV
        'abuseip_distinct_users': int(datos_abuse.get('numDistinctUsers', 0)), # Entero
        'abuseip_last_reported': datos_abuse.get('lastReportedAt', 'Never_Reported'),
        'country': str(datos_geo.get('country', 'Unknown')),
        'asn': asn_final, # Campo recuperado de api_requests.py
        'isp': str(datos_geo.get('isp', 'Unknown')),
        'infra_owner': str(datos_geo.get('org', 'Unknown'))
    }