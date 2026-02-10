# 🧠 Face Recognition Microservice (Async & Scalable)

Sistema de reconhecimento facial assíncrono utilizando **FastAPI**,
**RabbitMQ**, **Redis** e **Qdrant**, com cache inteligente e
processamento distribuído via workers.

Projetado para cenários de alta concorrência, como: - controle de
acesso - PDV - SaaS - filas de processamento de imagens

------------------------------------------------------------------------

## 🚀 Funcionalidades

-   📤 Upload de imagem via API
-   🧵 Processamento assíncrono com RabbitMQ
-   🧑‍🦰 Extração de embeddings faciais (`face_recognition`)
-   🔍 Busca vetorial no Qdrant
-   ⚡ Cache de resultados no Redis
-   📊 Contador de reconhecimentos
-   🔁 Retry automático em caso de falha
-   🧩 Arquitetura desacoplada (API + Worker)

------------------------------------------------------------------------

## 🏗️ Arquitetura

\[ Client \] → \[ FastAPI \] → \[ RabbitMQ \] → \[ Worker \] → \[ Redis
Cache \] → \[ Qdrant \]

------------------------------------------------------------------------

## 🧰 Tecnologias

-   FastAPI
-   RabbitMQ
-   Redis
-   Qdrant (Vector Database)
-   Docker & Docker Compose
-   face_recognition / dlib
-   Pillow / NumPy

------------------------------------------------------------------------

## 📂 Estrutura do Projeto

    .
    ├── api/
    ├── worker/
    ├── system/
    ├── docker-compose.yml
    └── README.md

------------------------------------------------------------------------

## ⚙️ Variáveis de Ambiente

``` env
#DEFAULT CONFIGS
APP_ENV=local
MAX_IMAGE_SIZE=1000
CACHE_SCORE_THRESHOLD_QDRANT=0.25
CACHE_DISTANCE_THRESHOLD_LOCAL=0.55
RECOGNITION_COUNTER_KEY=recognition_counter
MAX_RETRIES=3

#REDIS
REDIS_HOST=redis
REDIS_PORT=6379
QUEUE_NAME=face_recognition_jobs

#RabbitMQ
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_MANAGEMENT_PORT=15672
RABBITMQ_DEFAULT_USER=guest
RABBITMQ_DEFAULT_PASS=guest
RABBITMQ_VHOST="/"

#qDrant Database
QDRANT_HOST=qdrant
QDRANT_PORT=6333
COLLECTION_NAME=faces
```

------------------------------------------------------------------------

## ▶️ Executando

``` bash
docker-compose up --build
```

------------------------------------------------------------------------

## 📡 Endpoints

POST /async-recognition\
POST /sync-recognition\
GET /stats\
GET /users\
DELETE /stats\

------------------------------------------------------------------------

## 👨‍💻 Autor

Julio Xavier\
Software Engineer
