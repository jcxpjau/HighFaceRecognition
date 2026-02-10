from PIL import Image, ImageOps
import asyncio
import json
import numpy as np
from aio_pika import connect_robust, IncomingMessage, Message, DeliveryMode
import aioredis
from qdrant_client import QdrantClient
import os
import face_recognition
import base64
from io import BytesIO

RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'rabbitmq')
RABBITMQ_USER = os.getenv('RABBITMQ_DEFAULT_USER', 'guest')
RABBITMQ_PASS = os.getenv('RABBITMQ_DEFAULT_PASS', 'guest')
RABBITMQ_PORT = os.getenv('RABBITMQ_PORT', '5672')

RABBITMQ_URL = f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT}/"
REDIS_URL = f"redis://{os.getenv('REDIS_HOST', 'redis')}:{os.getenv('REDIS_PORT', '6379')}"

QDRANT_HOST = os.getenv('QDRANT_HOST', 'qdrant')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "faces")

CACHE_DISTANCE_THRESHOLD_LOCAL = float(os.getenv("CACHE_DISTANCE_THRESHOLD_LOCAL", 0.45))
CACHE_SCORE_THRESHOLD_QDRANT = float(os.getenv("CACHE_SCORE_THRESHOLD_QDRANT", 0.85))
RECOGNITION_COUNTER_KEY = os.getenv("RECOGNITION_COUNTER_KEY", "recognition_counter")
CACHE_TTL_SECONDS = 3600

MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
RETRY_PREFIX = "retry:"

# ==========================
# PROCESS MESSAGE
# ==========================
async def process_message(message: IncomingMessage, redis, qdrant, channel):
    async with message.process():
        try:
            data = json.loads(message.body.decode())
            job_id = data.get("job_id")
            image_b64 = data.get("image_base64")

            if not job_id or not image_b64:
                print("Invalid message format")
                return

            print(f"Processing job {job_id}")

            image_bytes = base64.b64decode(image_b64)

            pil_image = Image.open(BytesIO(image_bytes))
            pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")

            max_size = 1000
            if max(pil_image.size) > max_size:
                scale = min(max_size / pil_image.size[0], max_size / pil_image.size[1])
                pil_image = pil_image.resize(
                    (int(pil_image.size[0] * scale), int(pil_image.size[1] * scale)),
                    Image.LANCZOS
                )

            image = np.array(pil_image)
            face_locations = face_recognition.face_locations(image)

            if not face_locations:
                print(f"No faces found for job {job_id}")
                return

            unknown_encoding = face_recognition.face_encodings(image, face_locations)[0]
            unknown_encoding = np.array(unknown_encoding)

        except Exception as e:
            await handle_retry(message, redis, channel, str(e))
            return

        try:
            async for key in redis.scan_iter(match="face_cache:*"):
                cached_json = await redis.get(key)
                if not cached_json:
                    continue

                cached_data = json.loads(cached_json)
                cached_encoding = np.array(cached_data["encoding"])
                distance = np.linalg.norm(unknown_encoding - cached_encoding)

                if distance <= CACHE_DISTANCE_THRESHOLD_LOCAL:
                    print(f"Cache hit for job {job_id}")
                    await redis.incr(RECOGNITION_COUNTER_KEY)
                    await publish_success(channel, job_id, cached_data["identifier"], cached_data["photo"], True)
                    return

            print(f"Searching vector in Qdrant for job {job_id}")

            def qdrant_search():
                return qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=unknown_encoding.tolist(),
                limit=1,
                with_payload=True,
            )

            search_result = await asyncio.to_thread(qdrant_search)

            if search_result and search_result.points:
                point = search_result.points[0]
                score = point.score

                if score <= CACHE_SCORE_THRESHOLD_QDRANT:
                    payload = point.payload
                    print(f"Face recognized for job {job_id} (distance={distance:.4f})")

                    await redis.incr(RECOGNITION_COUNTER_KEY)
                    await redis.set(
                        f"face_cache:{payload['identifier']}",
                        json.dumps({
                            "encoding": unknown_encoding.tolist(),
                            "identifier": payload["identifier"],
                            "photo": payload["photo"]
                        }),
                        ex=CACHE_TTL_SECONDS
                    )

                    await publish_success(channel, job_id, payload["identifier"], payload["photo"], False)
                    return

            print(f"No match found for job {job_id}")
            await publish_success(channel, job_id, "Unknown", "", False)

        except Exception as e:
            await handle_retry(message, redis, channel, str(e))

# ==========================
# RETRY
# ==========================
async def handle_retry(message, redis, channel, reason):
    body = json.loads(message.body.decode())
    job_id = body.get("job_id", "unknown")

    retry_key = f"{RETRY_PREFIX}{job_id}"
    retries = await redis.incr(retry_key)
    await redis.expire(retry_key, 3600)

    print(f"Retry {retries}/{MAX_RETRIES} for job {job_id} | Reason: {reason}")

    if retries >= MAX_RETRIES:
        print(f"Job {job_id} failed")
        return

    await channel.default_exchange.publish(
        Message(body=message.body, delivery_mode=DeliveryMode.PERSISTENT),
        routing_key="face_recognition_jobs"
    )

# ==========================
# PUBLISH SUCCESS
# ==========================
async def publish_success(channel, job_id, identifier, photo, cached):
    payload = {
        "job_id": job_id,
        "identifier": identifier,
        "photo": photo,
        "cached": cached
    }

    await channel.default_exchange.publish(
        Message(body=json.dumps(payload).encode(), delivery_mode=DeliveryMode.PERSISTENT),
        routing_key="face_recognition_success"
    )


async def main():
    print("Worker started")

    try:
        print(f"Connecting to RabbitMQ at {RABBITMQ_URL} ...")
        connection = await connect_robust(RABBITMQ_URL)
        print("Connected to RabbitMQ")

        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        print("Channel created")

        queue = await channel.declare_queue("face_recognition_jobs", durable=True)
        await channel.declare_queue("face_recognition_success", durable=True)
        print("Queues declared")

        print(f"Connecting to Redis at {REDIS_URL} ...")
        redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
        print("Connected to Redis")

        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} ...")
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print("Connected to Qdrant")

        print("Waiting for mensages ...")
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                try:
                    print("MESSAGE RECEIVED", message.message_id, flush=True)
                    await process_message(message, redis, qdrant, channel)
                except Exception as e:
                    print("ERROR ON LOOP:", e, flush=True)

    except Exception as e:
        print(f"Startup failed: {e}")
        return


if __name__ == "__main__":
    asyncio.run(main())
