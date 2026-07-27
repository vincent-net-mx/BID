import streamlit as st

# Configuración global de la página
st.set_page_config(
    page_title="Sistema de Inteligencia de Amenazas",
    page_icon="🛡️", # ¡Esto le pone el escudo en el menú y en la pestaña del navegador!
    layout="wide"
)

# Título y presentación
st.title("Sistema Estadístico de Inteligencia de Amenazas")
st.markdown("---")

st.markdown("""
### Bienvenido al portal de análisis de ciberseguridad.

Este proyecto universitario utiliza inteligencia estadística y probabilidad bayesiana para evaluar el riesgo de direcciones IP. 

👈 **Utiliza el menú lateral para navegar entre las distintas herramientas:**

*   **1. Analizador:** Búsqueda en tiempo real de IPs, contrastada con la API de AbuseIPDB.
*   **2. Dashboard:** Análisis visual del comportamiento y prevalencia de amenazas.
*   **3. Herramientas:** (Próximamente) Análisis de tráfico y pcap.
""")

st.info("Proyecto de Probabilidad y Estadística I - Universidad de Guadalajara")