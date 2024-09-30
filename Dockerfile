FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt --no-deps
EXPOSE 5000
ENV FLASK_ENV=production
ENV FLASK_APP=api/app.py
CMD ["flask", "run", "--host=0.0.0.0"]