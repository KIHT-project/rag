FROM python:3.14.2-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /service

# Install OS deps only if you really need them.
# libatlas-base-dev is huge and slow, remove unless you have a hard dependency.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first for caching
COPY requirements.txt /service/requirements.txt
RUN pip install --no-cache-dir -r /service/requirements.txt

COPY src /service/src
COPY configs /service/configs

# Needed so "import biomed_platform" works
ENV PYTHONPATH=/service/src

# Run with uvicorn, not python file execution
CMD ["python", "-m", "uvicorn", "biomed_platform.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
