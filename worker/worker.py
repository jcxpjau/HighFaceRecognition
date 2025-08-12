import asyncio
import json
import numpy as np
from aio_pika import connect_robust, IncomingMessage, Message, DeliveryMode
import aioredis
from qdrant_client import QdrantClient
import os

RABBITMQ_URL = f"amqp://guest:guest@{os.getenv('RABBITMQ_HOST', 'localhost')}:{os.getenv('RABBITMQ_PORT', '5672')}/"
REDIS_URL = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
COLLECTION_NAME = "faces"
CACHE_DISTANCE_THRESHOLD = 0.5

async def process_message(message: IncomingMessage, redis, qdrant, channel):
    async with message.process():
        try:
            data = json.loads(message.body.decode())
        except Exception as e:
            print(f"❌ Error decoding message: {e}")
            return

        job_id = data.get("job_id")
        encoding = data.get("encoding")

        if not job_id or not encoding:
            print("⚠️ Missing job_id or encoding in message")
            return

        async for key in redis.scan_iter(match="face_cache:*"):
            data_json = await redis.get(key)
            if not data_json:
                continue
            cached_data = json.loads(data_json)
            cached_encoding = np.array(cached_data["encoding"])
            distance = np.linalg.norm(np.array(encoding) - cached_encoding)
            if distance <= CACHE_DISTANCE_THRESHOLD:
                identifier = cached_data["identifier"]
                photo = cached_data["photo"]
                await channel.default_exchange.publish(
                    Message(
                        body=json.dumps({
                            "job_id": job_id,
                            "identifier": identifier,
                            "photo": photo,
                            "cached": True
                        }).encode(),
                        delivery_mode=DeliveryMode.PERSISTENT
                    ),
                    routing_key="face_recognition_success"
                )
                return


        try:
            search_result = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=encoding,
                limit=1,
                with_payload=True,
            )
        except Exception as e:
            return

        if search_result:
            point = search_result[0]
            distance = point.score
            if distance <= CACHE_DISTANCE_THRESHOLD:
                payload = point.payload
                await redis.set(
                    f"face_cache:{payload['identifier']}",
                    json.dumps({
                        "encoding": encoding,
                        "identifier": payload["identifier"],
                        "photo": payload["photo"]
                    }),
                    ex=3600
                )
                await channel.default_exchange.publish(
                    Message(
                        body=json.dumps({
                            "job_id": job_id,
                            "identifier": payload["identifier"],
                            "photo": payload["photo"],
                            "cached": False
                        }).encode(),
                        delivery_mode=DeliveryMode.PERSISTENT
                    ),
                    routing_key="face_recognition_success"
                )
                return

        await redis.set(
            f"face_job_result:{job_id}",
            json.dumps({
                "identifier": None,
                "photo": None,
                "cached": False
            }),
            ex=300
        )


async def main():
    connection = await connect_robust(RABBITMQ_URL)
    channel = await connection.channel()
    queue = await channel.declare_queue("face_recognition_jobs", durable=True)
    await channel.declare_queue("face_recognition_success", durable=True)

    redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            try:
                await process_message(message, redis, qdrant, channel)
            except Exception as e:
                print(f"❌ Error processing message: {e}")

if __name__ == "__main__":
    asyncio.run(main())
