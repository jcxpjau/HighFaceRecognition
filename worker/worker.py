from PIL import Image, ImageOps
import asyncio
import json
import numpy as np
from aio_pika import connect_robust, IncomingMessage, Message, DeliveryMode
import aioredis
from qdrant_client import QdrantClient
import os
import face_recognition

RABBITMQ_URL = f"amqp://guest:guest@{os.getenv('RABBITMQ_HOST', 'localhost')}:{os.getenv('RABBITMQ_PORT', '5672')}/"
REDIS_URL = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "faces")

CACHE_DISTANCE_THRESHOLD = 0.5
CACHE_TTL_SECONDS = 3600
RECOGNITION_COUNTER_KEY = "recognition_count"

MAX_RETRIES = 3
RETRY_PREFIX = "retry:"

async def process_message(message: IncomingMessage, redis, qdrant, channel):
    async with message.process(ignore_processed=True):
        try:
            data = json.loads(message.body.decode())
            job_id = data.get("job_id")
            image_path = data.get("path")

            if not job_id or not image_path:
                print("Invalid message format: job_id or path missing")
                return

            print(f"Processing job {job_id}")

            pil_image = Image.open(image_path)
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

                if distance <= CACHE_DISTANCE_THRESHOLD:
                    print(f"Cache hit for job {job_id}")
                    await redis.incr(RECOGNITION_COUNTER_KEY)

                    await publish_success(channel, job_id, cached_data["identifier"], cached_data["photo"], True)
                    return

            print(f"🔍 Searching vector in Qdrant for job {job_id}")

            search_result = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=unknown_encoding.tolist(),
                limit=1,
                with_payload=True,
            )

            if search_result:
                point = search_result[0]
                distance = point.score

                if distance <= CACHE_DISTANCE_THRESHOLD:
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

                    os.remove(image_path)
                    await publish_success(channel, job_id, payload["identifier"], payload["photo"], False)
                    return

            print(f"No match found for job {job_id}")

            os.remove(image_path)
            await publish_success(channel, job_id, "Unknown", "", False)

        except Exception as e:
            await handle_retry(message, redis, channel, str(e))


async def handle_retry(message, redis, channel, reason):
    body = json.loads(message.body.decode())
    job_id = body.get("job_id", "unknown")

    retry_key = f"{RETRY_PREFIX}{job_id}"
    retries = await redis.incr(retry_key)
    await redis.expire(retry_key, 3600)

    print(f"Retry {retries}/{MAX_RETRIES} for job {job_id} | Reason: {reason}")

    if retries >= MAX_RETRIES:
        print(f"☠️ Job {job_id} failed after {MAX_RETRIES} retries")
        return

    await channel.default_exchange.publish(
        Message(
            body=message.body,
            delivery_mode=DeliveryMode.PERSISTENT
        ),
        routing_key="face_recognition_jobs"
    )


async def publish_success(channel, job_id, identifier, photo, cached):
    payload = {
        "job_id": job_id,
        "identifier": identifier,
        "photo": photo,
        "cached": cached
    }

    await channel.default_exchange.publish(
        Message(
            body=json.dumps(payload).encode(),
            delivery_mode=DeliveryMode.PERSISTENT
        ),
        routing_key="face_recognition_success"
    )


async def main():
    print("Worker started")

    connection = await connect_robust(RABBITMQ_URL)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    queue = await channel.declare_queue("face_recognition_jobs", durable=True)
    await channel.declare_queue("face_recognition_success", durable=True)

    redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            await process_message(message, redis, qdrant, channel)


if __name__ == "__main__":
    asyncio.run(main())
