FROM python:3.14.2-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /service/scheduler_pubmed

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /service/scheduler_pubmed/requirements.txt
RUN pip install --no-cache-dir -r /service/scheduler_pubmed/requirements.txt

COPY __init__.py /service/scheduler_pubmed/__init__.py
COPY src /service/scheduler_pubmed/src
COPY config /service/scheduler_pubmed/config
COPY alembic /service/scheduler_pubmed/alembic
COPY alembic.ini /service/scheduler_pubmed/alembic.ini

ENV PYTHONPATH=/service
ENV PUBMED_SCHEDULER_CONFIG_DIR=/service/scheduler_pubmed/config

CMD ["python", "-m", "uvicorn", "scheduler_pubmed.src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
