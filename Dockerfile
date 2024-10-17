FROM python:3.11-slim

WORKDIR /app

# Cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Cài đặt các dependencies từ requirements.txt, loại bỏ torch và cài đặt phiên bản CPU
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y torch \
    && pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

EXPOSE 5000
ENV FLASK_ENV=development
CMD ["python", "API/app.py"]