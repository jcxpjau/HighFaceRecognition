import os
import asyncio
import aio_pika
import aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.upload import router as upload_router
from routes.recognition import router as recognition_router
from routes.stats import router as stats_router
from routes.users import router as users_router
from routes.reset import router as reset_router
from qdrant import init_qdrant_collection
from urllib.parse import quote_plus

app = FastAPI(
    title="Face Recognition App",
    description="API for face recognition with high performance",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ENV CONFIG
# =========================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_URL = (
    f"{REDIS_HOST}://{REDIS_HOST}:{REDIS_PORT}"
)
QUEUE_NAME = os.getenv("QUEUE_NAME", "face_recognition_jobs")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "faces")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", "5672")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = quote_plus(os.getenv("RABBITMQ_PASSWORD", "guest"))
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "0")
RABBITMQ_URL = (
    f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}"
    f"@{RABBITMQ_HOST}:{RABBITMQ_PORT}/{quote_plus(RABBITMQ_VHOST)}"
)
# =========================
# STARTUP
# =========================
@app.on_event("startup")
async def startup_event():
    for i in range(10):
        try:
            connection = await aio_pika.connect_robust(RABBITMQ_URL)
            channel = await connection.channel()
            await channel.declare_queue(QUEUE_NAME, durable=True)
            app.state.rabbitmq_connection = connection
            app.state.rabbitmq_channel = channel
            print("Connected to RabbitMQ successfully")
            break
        except Exception as e:
            print(f"RabbitMQ not ready, retry {i+1}/10... Error: {e}")
            await asyncio.sleep(3)
    else:
        raise RuntimeError("Cannot connect to RabbitMQ after multiple attempts")

    redis_async = await aioredis.from_url(REDIS_URL, decode_responses=True)
    app.state.redis_async = redis_async

    init_qdrant_collection()

# =========================
# SHUTDOWN
# =========================
@app.on_event("shutdown")
async def shutdown_event():
    connection = getattr(app.state, "rabbitmq_connection", None)
    if connection and not connection.is_closed:
        await connection.close()

    redis = getattr(app.state, "redis_async", None)
    if redis:
        await redis.close()

# =========================
# ROUTES
# =========================
app.include_router(upload_router)
app.include_router(recognition_router)
app.include_router(stats_router)
app.include_router(users_router)
app.include_router(reset_router)
