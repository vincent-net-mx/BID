# 1. Usar una imagen oficial de Python ligera
FROM python:3.11-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar primero el archivo de dependencias (para aprovechar la caché de Docker)
COPY requirements.txt .

# 4. Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar todo el resto del proyecto al contenedor
COPY . .

# 6. Exponer el puerto que usa Streamlit por defecto
EXPOSE 8501

# 7. Comando para arrancar la aplicación
CMD ["streamlit", "run", "Inicio.py", "--server.port=8501", "--server.address=0.0.0.0"]