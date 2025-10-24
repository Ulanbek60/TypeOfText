# Используем подходящий Python (например 3.12)
FROM python:3.12-slim

# Создаем рабочую директорию
WORKDIR /app

# Копируем весь проект внутрь контейнера
COPY . /app

# Устанавливаем зависимости через pip
RUN pip install --no-cache-dir torch==2.3.0 \
    torchtext==0.18.0 torchdata==0.9.0 streamlit==1.50.0 \
    numpy==2.3.4 pandas==2.3.3 altair==5.5.0

# Открываем порт Streamlit
EXPOSE 8501

# Команда для запуска
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
