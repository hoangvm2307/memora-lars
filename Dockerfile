FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt --no-deps
EXPOSE 5000
ENV FLASK_ENV=development
CMD ["python", "API/app.py"]