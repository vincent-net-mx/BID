import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Dashboard Estadístico", page_icon="📊", layout="wide")
st.title("📊 Dashboard de Inteligencia de Amenazas")

# ==========================================
# --- DICCIONARIOS GLOBALES ---
# ==========================================
DICCIONARIO_PAISES = {
    "Brazil": "Brasil",
    "United States": "Estados Unidos",
    "China": "China",
    "Russia": "Rusia",
    "France": "Francia",
    "Germany": "Alemania",
    "United Kingdom": "Reino Unido",
    "India": "India",
    "South Korea": "Corea del Sur",
    "Netherlands": "Países Bajos"
}

CATEGORIAS_ABUSEIP = {
    '1': 'Compromiso DNS', '2': 'Envenenamiento DNS', '3': 'Fraude',
    '4': 'DDoS', '5': 'Fuerza Bruta FTP', '6': 'Ping of Death',
    '7': 'Phishing', '8': 'Fraude VoIP', '9': 'Proxy Abierto',
    '10': 'Web Spam', '11': 'Email Spam', '12': 'Blog Spam',
    '13': 'IP de VPN', '14': 'Escaneo de Puertos', '15': 'Hacking',
    '16': 'Inyección SQL', '17': 'Spoofing', '18': 'Fuerza Bruta',
    '19': 'Bot Malicioso', '20': 'Host Comprometido', '21': 'Ataque a App Web',
    '22': 'Ataque SSH', '23': 'Ataque IoT',
    "unknown_attack": "Desconocido", 
    "brute_force": "Fuerza Bruta", 
    "port_scan": "Escaneo de Puertos", 
    "ddos": "DDoS", 
    "web_spam": "Spam Web",
    "espionage": "Espionaje",
    "none": "Sin Categoría", 
    "exploit_delivery": "Exploits",
    "phishing": "Phishing",
    "ransomware": "Ransomware"
}

# --- CARGA Y TRANSFORMACIÓN DE DATOS ---
@st.cache_data
def cargar_datos():
    df_raw = pd.read_csv("data/BID_dataset.csv")
    df_raw['País_ES'] = df_raw['country'].replace(DICCIONARIO_PAISES)
    df_raw['Vector_ES'] = df_raw['attack_type'].astype(str).replace(CATEGORIAS_ABUSEIP)
    return df_raw

try:
    df = cargar_datos()
except FileNotFoundError:
    st.error("No se encontró el archivo dataset. Verifica que esté en data/BID_dataset.csv")
    st.stop()

# ==========================================
# --- BARRA DE FILTROS ---
# ==========================================
st.sidebar.header("🔍 Filtros de Análisis")

niveles_amenaza = sorted(df['threat_level'].dropna().unique().tolist())
nivel_seleccionado = st.sidebar.multiselect("Nivel de Amenaza:", options=niveles_amenaza, default=niveles_amenaza)

tipos_ataque = sorted(df['Vector_ES'].dropna().unique().tolist())
ataque_seleccionado = st.sidebar.multiselect("Tipo de Ataque:", options=tipos_ataque, default=tipos_ataque)

paises = sorted(df['País_ES'].dropna().unique().tolist())
pais_seleccionado = st.sidebar.multiselect("País de Origen:", options=paises, default=paises)

isps = sorted(df['isp'].dropna().unique().tolist())
isp_seleccionado = st.sidebar.multiselect("Proveedor (ISP):", options=isps, default=isps)

# Aplicar todos los filtros
df_filtrado = df[
    (df['threat_level'].isin(nivel_seleccionado)) &
    (df['Vector_ES'].isin(ataque_seleccionado)) &
    (df['País_ES'].isin(pais_seleccionado)) &
    (df['isp'].isin(isp_seleccionado))
]

if df_filtrado.empty:
    st.warning("⚠️ Los filtros seleccionados no arrojaron ningún resultado. Por favor, ajusta tu selección.")
    st.stop() 

# ==========================================
# --- SEPARACIÓN EN PESTAÑAS (TABS) ---
# ==========================================
# Añadimos la tercera pestaña para probabilidad
tab1, tab2, tab3 = st.tabs(["📈 Análisis Estadístico", "📊 Visualización de Datos", "🎲 Probabilidad Clásica"])

# ------------------------------------------
# PESTAÑA 1: ESTADÍSTICA Y KPIs (ENTREGABLE 1)
# ------------------------------------------
with tab1:
    st.subheader("Resumen de la Muestra (Filtrada)")
    col1, col2, col3, col4 = st.columns(4)

    total_registros = len(df_filtrado)
    maliciosos = len(df_filtrado[df_filtrado['is_malicious'] == 1])
    benignos = len(df_filtrado[df_filtrado['is_malicious'] == 0])
    tasa_prev = (maliciosos / total_registros) * 100 if total_registros > 0 else 0

    col1.metric("Total de Registros", f"{total_registros}")
    col2.metric("Infraestructura Benigna", f"{benignos}")
    col3.metric("Infraestructura Maliciosa", f"{maliciosos}")
    col4.metric("Tasa de Prevalencia P(M)", f"{tasa_prev:.1f}%")

    st.markdown("---")
    st.subheader("Estadística Descriptiva Inteligente")
    
    columnas_analizables = ['abuseip_score', 'abuseip_distinct_users', 'usage_type', 'abuseip_categories', 'Vector_ES', 'País_ES']
    columnas_validas = [col for col in columnas_analizables if col in df_filtrado.columns]
    
    variable_seleccionada = st.selectbox(
        "Selecciona la variable a evaluar:", 
        options=columnas_validas,
        help="El sistema detectará el tipo de variable y aplicará los cálculos estadísticos correctos."
    )

    if df_filtrado[variable_seleccionada].isnull().all() or df_filtrado.empty:
        st.warning(f"⚠️ No hay datos suficientes para analizar '{variable_seleccionada}'.")
    else:
        es_numerica = pd.api.types.is_numeric_dtype(df_filtrado[variable_seleccionada])
        
        if variable_seleccionada in ['usage_type', 'abuseip_categories', 'Vector_ES', 'País_ES'] or not es_numerica:
            # NOMINAL
            st.info("📊 **Variable Categórica Nominal.** No es una variable aleatoria numérica. Se aplican Medidas de Frecuencia y Moda.")
            moda_val = df_filtrado[variable_seleccionada].mode()
            moda = moda_val[0] if not moda_val.empty else "No definida"
            st.metric("Moda (Valor más frecuente)", str(moda))
            
            st.markdown("#### Frecuencias y Proporciones (Top 5)")
            conteo = df_filtrado[variable_seleccionada].value_counts().head(5)
            proporcion = df_filtrado[variable_seleccionada].value_counts(normalize=True).head(5) * 100
            df_frecuencias = pd.DataFrame({'Frecuencia Absoluta': conteo, 'Proporción (%)': proporcion.map("{:.2f}%".format)})
            st.dataframe(df_frecuencias, use_container_width=True)
            
        else:
            # RAZÓN / ESCALA (Detección Discreta vs Continua para el Reporte)
            es_discreta = pd.api.types.is_integer_dtype(df_filtrado[variable_seleccionada])
            tipo_var_texto = "Discreta (Valores enteros, ej. Conteos)" if es_discreta else "Continua (Valores decimales, ej. Escalas/Porcentajes)"
            
            st.info(f"🔢 **Variable Aleatoria {tipo_var_texto}.** Se aplican Medidas de Tendencia Central y Dispersión.")
            
            media = df_filtrado[variable_seleccionada].mean()
            mediana = df_filtrado[variable_seleccionada].median()
            desviacion = df_filtrado[variable_seleccionada].std()
            varianza = df_filtrado[variable_seleccionada].var()
            rango = df_filtrado[variable_seleccionada].max() - df_filtrado[variable_seleccionada].min()
            q1 = df_filtrado[variable_seleccionada].quantile(0.25)
            q3 = df_filtrado[variable_seleccionada].quantile(0.75)
            iqr = q3 - q1 
            
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1:
                st.markdown("**Tendencia Central**")
                st.metric("Media (x̄)", f"{media:.2f}")
                st.metric("Mediana (x̃)", f"{mediana:.2f}")
            with col_c2:
                st.markdown("**Medidas de Dispersión**")
                st.metric("Desviación Estándar (s)", f"{desviacion:.2f}")
                st.metric("Varianza (s²)", f"{varianza:.2f}")
                st.metric("Rango", f"{rango:.2f}")
            with col_c3:
                st.markdown("**Posición Relativa**")
                st.metric("Cuartil 1 (25%)", f"{q1:.2f}")
                st.metric("Cuartil 3 (75%)", f"{q3:.2f}")
                st.metric("IQR", f"{iqr:.2f}")

# ------------------------------------------
# PESTAÑA 2: VISUALIZACIONES GRÁFICAS
# ------------------------------------------
with tab2:
    st.subheader("Patrones de Amenaza e Infraestructura")
    df_maliciosos_filt = df_filtrado[df_filtrado['is_malicious'] == 1].copy()

    if not df_maliciosos_filt.empty:
        # 1. MAPA DE CALOR (GEO) - VERSIÓN MEJORADA POR EL EQUIPO
        st.markdown("### 🗺️ Origen Geográfico de las Amenazas")
        
        # Agrupamos conservando ambas columnas de país (Inglés para Plotly, Español para la UI)
        df_geo = df_maliciosos_filt.groupby(['country', 'País_ES']).agg(
            Ataques=('country', 'size'),
            # Sacamos la moda de las columnas para saber el vector y el ISP más frecuente por país
            Vector_Principal=('Vector_ES', lambda x: x.mode()[0] if not x.mode().empty else 'N/A'),
            ISP_Principal=('isp', lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
        ).reset_index()
        
        fig_geo = px.choropleth(
            df_geo, 
            locations="country",          # Inglés (Obligatorio para la geometría del mapa)
            locationmode="country names",
            color="Ataques", 
            hover_name="País_ES",         # Español (Para el título del tooltip)
            custom_data=["Ataques", "Vector_Principal", "ISP_Principal"], 
            color_continuous_scale="Reds"
        )

        fig_geo.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br><br>" +                  
                "Total de Ataques: %{customdata[0]}<br>" +       
                "Vector Frecuente: %{customdata[1]}<br>" +       
                "ISP Atacante: %{customdata[2]}" +               
                "<extra></extra>"                                
            )
        )

        fig_geo.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',  
            margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_showscale=False      
        )

        fig_geo.update_geos(
            showframe=False, showcoastlines=True,
            coastlinecolor="rgba(127, 127, 127, 0.5)", countrycolor="rgba(127, 127, 127, 0.3)",
            projection_type='natural earth', bgcolor='rgba(0,0,0,0)',       
            showland=True, landcolor='rgba(127, 127, 127, 0.1)' 
        )

        st.plotly_chart(fig_geo, use_container_width=True, theme="streamlit")

        # ------------------------------------------
        # Gráficos complementarios (Barras y Treemap)
        # ------------------------------------------
        col_graf1, col_graf2 = st.columns(2)
        with col_graf1:
            st.markdown("### 🛡️ Vectores de Ataque")
            fig_ataques = px.histogram(df_maliciosos_filt, y="Vector_ES", color="Vector_ES", orientation='h')
            fig_ataques.update_layout(showlegend=False)
            st.plotly_chart(fig_ataques, use_container_width=True)
        with col_graf2:
            st.markdown("### 🏢 Infraestructura Atacante (ISPs)")
            df_tree = df_maliciosos_filt.dropna(subset=['isp'])
            df_tree_grouped = df_tree.groupby('isp').size().reset_index(name='Ataques')
            fig_tree = px.treemap(df_tree_grouped, path=[px.Constant("ISPs"), 'isp'], values='Ataques')
            st.plotly_chart(fig_tree, use_container_width=True)
    else:
        st.warning("No hay datos para mostrar gráficos con los filtros actuales.")

# ------------------------------------------
# PESTAÑA 3: PROBABILIDAD CLÁSICA (ENTREGABLE 2)
# ------------------------------------------
with tab3:
    st.header("🎲 Análisis de Probabilidad e Independencia")
    st.markdown("En esta sección aplicamos teoría de conjuntos para calcular probabilidades marginales, conjuntas y condicionales, resolviendo si dos eventos de ciberseguridad están correlacionados (dependientes) o si ocurren al azar (independientes).")
    
    st.markdown("### Construye tus Eventos")
    col_evt1, col_evt2 = st.columns(2)
    
    # Selector Evento A
    with col_evt1:
        st.info("**Evento A**")
        var_A = st.selectbox("Categoría A:", ['Vector_ES', 'País_ES', 'isp'], key="var_a")
        opciones_A = df_filtrado[var_A].dropna().unique().tolist()
        val_A = st.selectbox("Condición A:", opciones_A, key="val_a")
        
    # Selector Evento B
    with col_evt2:
        st.info("**Evento B**")
        var_B = st.selectbox("Categoría B:", ['País_ES', 'Vector_ES', 'isp'], key="var_b")
        opciones_B = df_filtrado[var_B].dropna().unique().tolist()
        val_B = st.selectbox("Condición B:", opciones_B, key="val_b")

    # Cálculos Matemáticos
    N = len(df_filtrado)
    
    if N > 0:
        # Frecuencias
        count_A = len(df_filtrado[df_filtrado[var_A] == val_A])
        count_B = len(df_filtrado[df_filtrado[var_B] == val_B])
        count_A_and_B = len(df_filtrado[(df_filtrado[var_A] == val_A) & (df_filtrado[var_B] == val_B)])
        
        # Probabilidades
        p_A = count_A / N
        p_B = count_B / N
        p_A_and_B = count_A_and_B / N
        p_A_given_B = count_A_and_B / count_B if count_B > 0 else 0
        p_B_given_A = count_A_and_B / count_A if count_A > 0 else 0
        
        st.markdown("---")
        st.markdown("### 🧮 Resultados de Probabilidad")
        
        c_res1, c_res2, c_res3, c_res4 = st.columns(4)
        c_res1.metric(f"P(A)", f"{p_A:.4f}", help=f"Probabilidad marginal de que ocurra: {val_A}")
        c_res2.metric(f"P(B)", f"{p_B:.4f}", help=f"Probabilidad marginal de que ocurra: {val_B}")
        c_res3.metric(f"P(A ∩ B)", f"{p_A_and_B:.4f}", help=f"Probabilidad conjunta de que ocurran A y B al mismo tiempo")
        c_res4.metric(f"P(A | B)", f"{p_A_given_B:.4f}", help=f"Probabilidad condicional de A dado que ya ocurrió B")

        # Comprobación de Independencia
        st.markdown("### ⚖️ Prueba de Independencia ")
        st.latex(r"¿ P(A \cap B) = P(A) \times P(B) ?")
        
        p_A_times_p_B = p_A * p_B
        st.write(f"**P(A ∩ B) calculada:** {p_A_and_B:.6f}")
        st.write(f"**P(A) × P(B) calculada:** {p_A_times_p_B:.6f}")
        
        # Comparamos usando una pequeña tolerancia por cuestiones de decimales en Python
        es_independiente = abs(p_A_and_B - p_A_times_p_B) < 0.0001
        
        if es_independiente:
            st.success(f"✅ **Conclusión:** Como P(A ∩ B) es igual a P(A)P(B), los eventos **'{val_A}'** y **'{val_B}'** son estadísticamente **INDEPENDIENTES**. Saber que ocurrió uno no afecta la probabilidad de que ocurra el otro.")
        else:
            st.error(f"🚨 **Conclusión:** Como P(A ∩ B) es diferente de P(A)P(B), los eventos **'{val_A}'** y **'{val_B}'** son estadísticamente **DEPENDIENTES**. Existe una correlación entre ellos en la muestra.")

    else:
        st.warning("No hay suficientes datos para realizar cálculos.")