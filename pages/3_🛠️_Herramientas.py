import streamlit as st
from consultas_api import reportar_ip_abuse

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Herramientas", page_icon="🛠️", layout="wide")
st.title("🛠️ Caja de Herramientas")
st.markdown("Acciones activas de Threat Intelligence y mitigación.")

# --- SECCIÓN: REPORTAR IP ---
st.subheader("🚨 Reportar IP a la Comunidad (AbuseIPDB)")
st.markdown("¿Detectaste tráfico anómalo? Utiliza este formulario para contribuir a la base de datos global. Tu reporte alimentará directamente la API.")

# Diccionario de categorías (Inverso, para que el usuario lea el texto y la API reciba el número)
categorias_map = {
    "Fraude (3)": 3, "DDoS (4)": 4, "Proxy Abierto (9)": 9, "Web Spam (10)": 10, 
    "Email Spam (11)": 11, "Escaneo de Puertos (14)": 14, "Hacking (15)": 15, 
    "Fuerza Bruta (18)": 18, "Bot Malicioso (19)": 19, "Host Comprometido (20)": 20, 
    "Ataque a App Web (21)": 21, "Ataque SSH (22)": 22, "Ataque IoT (23)": 23
}

# Creamos el formulario
with st.form("form_reporte"):
    col1, col2 = st.columns(2)
    
    with col1:
        ip_reportar = st.text_input("Dirección IP a reportar:", placeholder="Ej: 192.168.1.100")
    
    with col2:
        categorias_seleccionadas = st.multiselect(
            "Categorías del ataque (Obligatorio):",
            options=list(categorias_map.keys()),
            help="Selecciona el tipo de actividad maliciosa que detectaste."
        )
    
    comentario = st.text_area(
        "Comentario / Evidencia (Opcional):", 
        placeholder="Ej: Intentos de login fallidos repetitivos en el puerto 22. Proyecto UdeG.",
        max_chars=250
    )
    
    # El botón que envía el formulario
    submit_btn = st.form_submit_button("🚀 Enviar Reporte a AbuseIPDB")
    
    if submit_btn:
        if not ip_reportar:
            st.error("Debes ingresar una dirección IP válida.")
        elif not categorias_seleccionadas:
            st.error("Debes seleccionar al menos una categoría de ataque.")
        else:
            # Extraemos los números (IDs) de las categorías que eligió el usuario
            ids_categorias = [categorias_map[cat] for cat in categorias_seleccionadas]
            
            with st.spinner(f"Reportando la IP {ip_reportar}..."):
                resultado = reportar_ip_abuse(ip_reportar, ids_categorias, comentario)
                
                if "error" not in resultado:
                    st.success(f"¡Éxito! La IP {ip_reportar} fue reportada a la comunidad.")
                    st.balloons() # ¡Un toque festivo por hacer las cosas bien!
                    
                    # Mostrar la respuesta de la API (opcional, para que se vea técnico)
                    with st.expander("Ver detalles del servidor"):
                        st.json(resultado)
                else:
                    # En caso de error (ej. reportar la misma IP dos veces seguidas o IP inválida)
                    st.error(f"No se pudo completar el reporte. Código HTTP: {resultado.get('codigo', 'N/A')}")
                    st.write(f"Detalle: {resultado.get('mensaje', 'Error desconocido')}")