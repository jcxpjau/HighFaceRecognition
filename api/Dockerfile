FROM python:3.10-slim

# Instalar dependências de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    python3-dev \
    libjpeg-dev \
    libpng-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /api

COPY requirements.txt .

# Atualizar pip e instalar requirements
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir gunicorn uvicorn
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Rodar API com Gunicorn + Uvicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--workers", "2"]
