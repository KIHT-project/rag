FROM python:3.11.9

WORKDIR /service/

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/service/src
ENV LOG_LEVEL=INFO

CMD ["python", "src/app/python/main.py"]
