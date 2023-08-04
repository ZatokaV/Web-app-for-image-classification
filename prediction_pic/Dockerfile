# Використовуємо офіційний базовий образ Python
FROM python:3.8

# Встановлюємо залежності
RUN pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Створюємо робочий каталог /app
WORKDIR /app

# Копіюємо файли вашого Django-проекту в контейнер
COPY . /app

# Запускаємо сервер Django
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
