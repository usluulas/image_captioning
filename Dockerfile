FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY image_captioning.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# entrypoint.sh önce Ollama sunucusunu bekler ve model yoksa indirir, sonra CMD'yi çalıştırır.
ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "image_captioning.py"]
