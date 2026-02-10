import pika
import aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.upload import router as upload_router
from app.routes.recognition import router as recognition_router
from app.routes.stats import router as stats_router
from app.routes.users import router as users_router
from app.routes.reset import router as reset_router
import aio_pika
import asyncio
from app.qdrant import init_qdrant_collection, get_qdrant_client

app = FastAPI(
    title="Face Recognition App",
    description="API for face recognition for high performance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REDIS_URL = "redis://redis:6379"

COLLECTION_NAME = "faces"

@app.on_event("startup")
async def startup_event():
    for i in range(10):
        try:
            connection = await aio_pika.connect_robust("amqp://guest:guest@rabbitmq/")
            channel = await connection.channel()
            await channel.declare_queue("face_recognition_jobs", durable=True)

            app.state.rabbitmq_connection = connection
            app.state.rabbitmq_channel = channel
            print("Connected to RabbitMQ async")
            break
        except Exception as e:
            print(f"RabbitMQ not ready, retry {i+1}/10...")
            await asyncio.sleep(3)
    else:
        raise RuntimeError("Cannot connect to RabbitMQ after multiple attempts")

    redis_async = await aioredis.from_url(REDIS_URL, decode_responses=True)
    app.state.redis_async = redis_async

    init_qdrant_collection()

@app.on_event("shutdown")
async def shutdown_event():
    connection = getattr(app.state, "rabbitmq_connection", None)
    if connection and connection.is_open:
        connection.close()

    redis = getattr(app.state, "redis_async", None)
    if redis:
        await redis.close()


app.include_router(upload_router)
app.include_router(recognition_router)
app.include_router(stats_router)
app.include_router(users_router)
app.include_router(reset_router)
