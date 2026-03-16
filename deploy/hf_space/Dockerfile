FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AI_PLATFORM_CLI_INIT=0 \
    PLATFORM_STORAGE_ROOT=/data/platform_projects \
    PORT=7860

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deploy/hf_space/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

EXPOSE 7860

CMD ["python", "-m", "api.main"]
