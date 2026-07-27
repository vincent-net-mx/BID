# Bad IP Detector (BID) 🛡️📊

Sistema de detección y análisis de riesgo para direcciones IPv4 basado en **Inteligencia Artificial** y **Estadística Bayesiana**.

---

## 📖 ¿Cómo nace BID?

Todo desarrollo tiene un origen, y **BID (Bad IP Detector)** no nació en un entorno corporativo, sino como una iniciativa analítica en las aulas. El proyecto surgió originalmente como un trabajo escolar para la materia de **Probabilidad y Estadística**. Impulsado por el interés de aterrizar la teoría matemática en desafíos técnicos y reales de ciberseguridad, el concepto tomó forma mediante la asesoría y guía del profesor **Edgar Gonzalo Cossio Franco**.

Lo que comenzó como un ejercicio académico enfocado en crear un modelo de riesgo probabilístico utilizando la teoría bayesiana, evolucionó significativamente hasta convertirse en una herramienta que integra *Machine Learning* e inteligencia de amenazas para identificar direcciones IPv4 maliciosas.

---

## 📝 ¿Qué es BID?

**BID** es una solución de ciberseguridad que utiliza un enfoque híbrido. Al no depender únicamente de listas negras estáticas, el sistema combina el poder predictivo del algoritmo **CatBoost** con el rigor del análisis estadístico (**Teorema de Bayes**). Su objetivo es proporcionar una puntuación de riesgo dinámica y confiable que permita tomar decisiones proactivas frente a posibles intrusiones o tráfico anómalo.

### 🚀 Características Principales

* **Análisis de Riesgo:** Calcula la probabilidad de que una dirección IPv4 sea maliciosa o benigna mediante el uso combinado de Machine Learning (CatBoost) y cálculo probabilístico (Teorema de Bayes).
* **Obtención de Inteligencia:** Realiza consultas automatizadas a fuentes abiertas (AbuseIPDB, IP-API) para extraer datos enriquecidos del reporte (confianza de abuso, tipo de uso, ataques reportados) y de la dirección IP (país, ASN, ISP).
* **Denuncia Automatizada:** Gestión de incidentes mediante una interfaz dedicada para la sumisión de IPs hacia AbuseIPDB, permitiendo categorizar el tipo de ataque de forma directa a través de su API.
* **Análisis Estadístico Avanzado:** Clasificación de variables según su naturaleza (nominal, numérica, etc.) para el cálculo de medidas de tendencia central, dispersión y posición relativa.
* **Dashboard de Amenazas:** Tablero interactivo con mapas de calor sobre el origen geográfico de los incidentes, gráficos de barras para vectores de ataque y análisis de la infraestructura proveedora (ISP).
* **Prueba de Independencia:** Módulo especializado para seleccionar dos variables del dataset y analizar estadísticamente si son dependientes o independientes.
* **Pipeline de Preparación:** Implementación de scripts dedicados a la depuración y limpieza de datos (*data cleaning*) para asegurar la estructura y calidad de la información antes de inyectarla al modelo.

## 📂 Estructura del Proyecto

* `Inicio.py`: Archivo principal que lanza la interfaz gráfica de la plataforma.
* `data/`:
    * `BID_dataset.csv`: Dataset con 970 direcciones IPv4.
    * `training_BID_dataset.cbm`: Archivo de entrenamiento para la IA.
* `pages/`:
    * `1_📡_Analizador.py`: Buscador que ejecuta el modelo bayesiano.
    * `2_📊_Dashboard.py`: Análisis descriptivo e interactivo usando One-Hot Encoding.
    * `3_🛠️_Herramientas.py`: Herramientas de reporte a bases de datos externas.
* `predictor.py`: Modelo predictor con IA (CatBoost)
* `motor_bayesiano.py`: Núcleo estadístico (Teorema de Bayes, Suavizado de Laplace).

## ℹ️ Origen de los Datos

A continuación se detallan las fuentes utilizadas tanto para el enriquecimiento en tiempo real como para la consolidación del dataset base de entrenamiento:

| Tipo de Información | Fuente / Repositorio | Propósito |
| :--- | :--- | :--- |
| **Información de Reportes** | [AbuseIPDB API](https://api.abuseipdb.com/api/v2/check) | Inteligencia y Reporte de Amenazas |
| **Datos Geográficos / ISP** | [IP-API](http://ip-api.com/batch) | Geolocalización y Datos de Red |
| **Listas Negras (Blacklists)** | [CIRCL OSINT Feed](https://www.circl.lu/doc/misp/feed-osint) | Datos de Entrenamiento Histórico |
| **Listas Blancas (Whitelists)** | [MISP Warninglists](https://misp.github.io/misp-warninglists/) | Filtrado de Falsos Positivos |

### 🔍 Desglose de Listas Blancas (MISP):
* [Microsoft Office 365](https://raw.githubusercontent.com/MISP/misp-warninglists/main/lists/microsoft-office365-ip/list.json) | [Cloudflare](https://raw.githubusercontent.com/MISP/misp-warninglists/main/lists/cloudflare/list.json) | [Googlebot](https://raw.githubusercontent.com/MISP/misp-warninglists/main/lists/googlebot/list.json)
* [Amazon AWS](https://raw.githubusercontent.com/MISP/misp-warninglists/main/lists/amazon-aws/list.json) | [Public DNS v4](https://raw.githubusercontent.com/MISP/misp-warninglists/main/lists/public-dns-v4/list.json) | [Microsoft Azure](https://raw.githubusercontent.com/MISP/misp-warninglists/main/lists/microsoft-azure/list.json)


---

# 🚀 Guía de Instalación y Despliegue

> ⚠️ **ADVERTENCIA:** Es estrictamente necesario contar con una API Key válida de [AbuseIPDB](https://www.abuseipdb.com/) para el despliegue del proyecto. Puedes registrarte y obtenerla de forma gratuita en su sitio oficial.

### 📋 Prerrequisitos Generales
* **API Key** de AbuseIP.
* **Python 3.9 o superior**.
* **Git** (opcional).
* **Docker** (opcional).
## Opciones para el despliegue
### A) Despliegue con Docker 🐋 (recomendado)
1. **Enciende Docker**.
2. **Abre la consola (Windows) o terminal (Linux/Mac OS)**.
3. **Instala el proyecto**.
    Ejecuta el siguiente comando:
    ```powershell
    # Comentario: Cambia la letra X de este comando por la API Key de AbuseIP y ejecútalo.
    docker run -d -p 8501:8501 -e ABUSEIP_API_KEY=X amzk12/bid-project:latest
    ```
    Verifica si ya está corriendo.
    ```powershell
    docker ps
    ```
    Para detener la ejecución sólo escribe:
    ```powershell
    # Comentario: Sustituye el <ID_o_nombre_del_contenedor> por el ID o nombre que aparezca en docker ps.
    docker stop <ID_o_nombre_del_contenedor>
    ```
4. **Navega hacia [http://localhost:8501](http://localhost:8501)**.
### B) Despliegue con Python 🐍
1. **Descarga el repositorio**.
    - A través de este [enlace](https://github.com/ArteAlex09/BID/archive/refs/heads/main.zip) (es necesario descomprimir el archivo).
    - O mediante Git: `git clone https://github.com/ArteAlex09/BID.git`
2. **Abre la consola (Windows) o terminal (Linux/Mac OS)**.
3. **Dirígete hacia la ruta/carpeta del repositorio**.
4. **Crea un entorno virtual**.
    *Windows:*
    ```powershell
    1 python3 -m venv venv
    2 .\venv\Scripts\Activate.ps1
    ```
    *Linux/Mac OS:*
    ```powershell
    1 python3 -m venv venv
    2 source/bin/activate
    ```
5. **Instala los requerimientos**.
    ```powershell
    pip install -r requirements.txt
    ```
6. **Crea un archivo de configuración del entorno (.env).**
- Cambia el nombre del archivo `.env.example` por `.env`.
- Abre el archivo `.env` con cualquier editor de texto.
- Sustituye la letra `x` por la **API Key** de AbuseIP que generaste.
7. **Enciende la interfaz gráfica**.
    ```powershell
    streamlit run Inicio.py
    ```
8. **Navega hacia [http://localhost:8501](http://localhost:8501)**.

