# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Копируем все зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Streamlit слушает на 0.0.0.0:8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
