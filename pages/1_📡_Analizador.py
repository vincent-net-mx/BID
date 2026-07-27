import streamlit as st
from motor_bayesiano import ejecutar_inferencia
from consultas_api import consultar_perfil_completo 
from data_cleaner import clean_corporate_name, clean_country_name

# --- INTEGRACIÓN DEL MOTOR DE IA (Código A) ---
from predictor import ejecutar_prediccion_hibrida

# --- CONFIGURACIÓN ---
# Asegúrate de que el nombre del archivo coincida con tu modelo entrenado
MODELO_IA_PATH = "data/training_BID_dataset.cbm"

st.set_page_config(page_title="Analizador de IPs", page_icon="📡", layout="wide")
st.title("📡 Simulador Probabilístico de IPs (BID System)")

# Proteger la API: La caché guarda la IP analizada por 1 hora
@st.cache_data(ttl=3600)
def obtener_inteligencia_cache(ip):
    """
    Obtiene la data de las APIs y la normaliza inmediatamente utilizando 
    las reglas de data_cleaner.py.
    """
    perfil = consultar_perfil_completo(ip)
    
    if perfil:
        # Normalización Corporativa
        perfil['isp'] = clean_corporate_name(perfil.get('isp', 'Unknown'))
        perfil['infra_owner'] = clean_corporate_name(perfil.get('infra_owner', 'Unknown'))
        
        # Normalización Geográfica
        perfil['country'] = clean_country_name(perfil.get('country', 'Unknown'))
        
    return perfil 

# --- INTERFAZ DEL BUSCADOR ---
st.markdown("Ingresa una dirección IP para ejecutar el análisis híbrido (Inferencia Bayesiana + CatBoost IA).")

ip_input = st.text_input("Dirección IP a analizar:", placeholder="Ej: 118.25.6.39")

st.markdown("### ⚙️ Parámetros del Modelo")
st.caption("Ajusta las variables para el cálculo del riesgo estadístico.")

# Matriz de interruptores
col1, col2, col3, col4 = st.columns(4)

with col1:
    v_score = st.checkbox("Puntaje de Abuso (Score)", value=True)
    v_usage = st.checkbox("Tipo de Uso", value=True)
    v_asn = st.checkbox("Sistema Autónomo (ASN)", value=True)
with col2:
    v_cats = st.checkbox("Categorías de Ataque", value=True)
    v_users = st.checkbox("Usuarios Distintos (Reportes)", value=True)
with col3:
    v_last = st.checkbox("Última vez reportada", value=True)
    v_country = st.checkbox("País de Origen", value=True)
with col4:
    v_isp = st.checkbox("Proveedor de Internet (ISP)", value=True)
    v_owner = st.checkbox("Propietario de Infraestructura", value=True)

# Mapa de variables
mapa_variables = {
    'abuseip_score': v_score,
    'usage_type': v_usage,
    'abuseip_categories': v_cats,
    'abuseip_distinct_users': v_users,
    'abuseip_last_reported': v_last,
    'country': v_country,
    'asn': v_asn,
    'isp': v_isp,
    'infra_owner': v_owner
}

variables_activas = [var for var, activa in mapa_variables.items() if activa]

if st.button("Ejecutar Análisis Híbrido"):
    if not variables_activas:
        st.error("Debes seleccionar al menos una variable para el análisis.")
    elif ip_input:
        with st.spinner("Correlacionando inteligencia de amenazas con motores de IA..."):
            perfil_ip = obtener_inteligencia_cache(ip_input)
            
            if perfil_ip:
                # LLAMADA AL MOTOR INTEGRADO (Código A)
                # Nota: 'ejecutar_prediccion_hibrida' ya maneja la inferencia bayesiana internamente
                p_m, p_posterior = ejecutar_inferencia(perfil_ip, variables_activas)
                resultado = ejecutar_prediccion_hibrida(perfil_ip, MODELO_IA_PATH, verbose=False)
                
                if resultado:
                    st.success(f"Análisis completado ({resultado['timestamp']}).")
                    st.markdown("---")
                    
                    # --- DASHBOARD DE RESULTADOS ---
                    m1, m2, m3 = st.columns(3)
                    
                    with m1:
                        st.info("📊 Riesgo calculado sin evidencia")
                        st.metric("$P(M)$", f"{p_m*100:.2f}%")
                        
                    
                    with m2:
                        st.info("📊 Riesgo calculado con evidencia")
            
                        delta = (p_posterior - p_m) * 100

                        st.metric("Posterior $P(M|E_1...E_n)$", f"{p_posterior*100:.2f}%", f"{delta:+.1f}% desde la línea base")
                        st.caption("Mediante teorema de Bayes")

                    with m3:
                        veredicto = resultado['veredicto']
                        confianza = resultado['probabilidad_ia']
                        
                        if veredicto == "MALICIOSA":
                            st.error(f"🤖 Veredicto IA (CatBoost)")
                        else:
                            st.success(f"🤖 Veredicto IA (CatBoost)")
                            
                        st.metric("Confianza del Modelo", f"{confianza*100:.2f}%")
                        st.metric("Veredicto", veredicto)
                    
                    st.markdown("---")
                    
                    # --- SECCIÓN DE TELEMETRÍA DETALLADA ---
                    c1, c2 = st.columns([1, 2])
                    
                    with c1:
                        st.markdown("#### Resumen de Inferencia")
                        st.write(f"Se evaluaron **{len(variables_activas)}** dimensiones de datos para llegar a este veredicto.")
                        if p_posterior > 0.8:
                            st.warning("⚠️ La evidencia estadística sugiere un comportamiento anómalo recurrente.")

                    with c2:
                        st.markdown("#### Telemetría Detectada")
                        
                        # Diccionarios de traducción
                        CATEGORIAS_ABUSEIP = {
                            '1': 'Compromiso DNS', '2': 'Envenenamiento DNS', '3': 'Fraude',
                            '4': 'DDoS', '5': 'Fuerza Bruta FTP', '6': 'Ping of Death',
                            '7': 'Phishing', '8': 'Fraude VoIP', '9': 'Proxy Abierto',
                            '10': 'Web Spam', '11': 'Email Spam', '12': 'Blog Spam',
                            '13': 'IP de VPN', '14': 'Escaneo de Puertos', '15': 'Hacking',
                            '16': 'Inyección SQL', '17': 'Spoofing', '18': 'Fuerza Bruta',
                            '19': 'Bot Malicioso', '20': 'Host Comprometido', '21': 'Ataque a App Web',
                            '22': 'Ataque SSH', '23': 'Ataque IoT'
                        }

                        TRADUCCION_USO = {
                            'Commercial': 'Comercial',
                            'Content Delivery Network': 'Red de Distribución de Contenido (CDN)',
                            'Data Center/Web Hosting/Transit': 'Data Center / Hosting',
                            'Fixed Line ISP': 'Proveedor de Línea Fija',
                            'Mobile ISP': 'Proveedor de Red Móvil',
                            'Reserved': 'Reservada (Privada)',
                            'Unknown': 'Desconocido'
                        }

                        TRADUCCION_PAISES = {
                            'Mexico': ('México', 'mx'),
                            'United States': ('Estados Unidos', 'us'),
                            'Canada': ('Canadá', 'ca'),
                            'Brazil': ('Brasil', 'br'),
                            'Argentina': ('Argentina', 'ar'),
                            'Colombia': ('Colombia', 'co'),
                            'Chile': ('Chile', 'cl'),
                            'Spain': ('España', 'es'),
                            'Germany': ('Alemania', 'de'),
                            'France': ('Francia', 'fr'),
                            'United Kingdom': ('Reino Unido', 'gb'),
                            'China': ('China', 'cn'),
                            'Japan': ('Japón', 'jp'),
                            'Russia': ('Rusia', 'ru'),
                            'South Korea': ('Corea del Sur', 'kr'),
                            'Netherlands': ('Países Bajos', 'nl'),
                            'India': ('India', 'in')
                        }

                        for var in mapa_variables.keys():
                            valor = perfil_ip.get(var, 'N/A')
                            
                            if var == 'abuseip_categories':
                                if valor in ['No_Reports', 'Unknown', 'N/A', '[]']:
                                    st.write("**Categorías:** Sin reportes")
                                else:
                                    try:
                                        lista_ids = valor.split(', ')
                                        nombres_cat = [CATEGORIAS_ABUSEIP.get(cat_id.strip(), f"Cat {cat_id}") for cat_id in lista_ids]
                                        with st.expander(f"**Categorías de Ataque ({len(nombres_cat)})**"):
                                            for nombre in nombres_cat:
                                                st.markdown(f"- {nombre}")
                                    except:
                                        st.write(f"**Categorías:** {valor}")

                            elif var == 'abuseip_last_reported':
                                if not valor or valor in ['Never_Reported', 'Unknown', 'N/A']:
                                    st.write("**Último Reporte:** Jamás / No disponible")
                                else:
                                    try:
                                        st.write(f"**Último Reporte:** {valor.split('T')[0]}")
                                    except (AttributeError, IndexError):
                                        st.write(f"**Último Reporte:** {valor}")

                            elif var == 'country':
                                datos_pais = TRADUCCION_PAISES.get(valor)
                                if datos_pais:
                                    nombre_es, iso = datos_pais
                                    bandera_url = f"https://flagcdn.com/24x18/{iso}.png"
                                    st.markdown(f"**País:** {nombre_es} <img src='{bandera_url}' width='20'>", unsafe_allow_html=True)
                                else:
                                    st.write(f"**País:** {valor}")
                            
                            elif var == 'usage_type':
                                st.write(f"**Uso:** {TRADUCCION_USO.get(valor, valor)}")

                            else:
                                label = var.replace('abuseip_', '').replace('_', ' ').title()
                                st.write(f"**{label}:** {valor}")