FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

WORKDIR /app

# Tesseract, görseldeki yazıyı okumak için gerekli (bkz. ocr.py). Bir sistem programıdır, pip ile
# gelmez; Türkçe dil verisi de ayrı bir paket. Kurulu olmasa program çalışmaya devam eder ama
# YAZILAR bölümü hiç çıkmaz, o yüzden imaja dahil ediliyor.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-tur \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY image_captioning.py ocr.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# entrypoint.sh önce Ollama sunucusunu bekler ve model yoksa indirir, sonra CMD'yi çalıştırır.
ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "image_captioning.py"]

