import io
import json
import os
import uuid
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from PIL import Image, ImageOps

import face_recognition 
from aio_pika import Message, DeliveryMode

from dependencies import get_redis_async, get_rabbitmq_channel
from qdrant import get_qdrant_client

# ========================
# CONFIGS
# ========================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
RABBITMQ_URL = (
    f"{REDIS_HOST}://{REDIS_HOST}:{REDIS_PORT}"
)
QUEUE_NAME = os.getenv("QUEUE_NAME", "face_recognition_jobs")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "faces")
CACHE_DISTANCE_THRESHOLD = float(os.getenv("CACHE_DISTANCE_THRESHOLD", 0.45))
RECOGNITION_COUNTER_KEY = os.getenv("RECOGNITION_COUNTER_KEY", "recognition_counter")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 1000))
FOTOS_DIR = os.path.abspath(os.getenv("FOTOS_DIR", "system/photos/recognition"))

os.makedirs(FOTOS_DIR, exist_ok=True)
router = APIRouter()

# ========================
# Pydantic Models
# ========================
class RecognitionResponse(BaseModel):
    name: Optional[str] = None
    photo: Optional[str] = None
    cached: Optional[bool] = None
    message: Optional[str] = None
    job_id: Optional[str] = None
    status: Optional[str] = None


# ========================
# Utils
# ========================
def preprocess_image(image_bytes: bytes) -> Image.Image:
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")

    if max(pil_image.size) > MAX_IMAGE_SIZE:
        scale = min(MAX_IMAGE_SIZE / pil_image.size[0], MAX_IMAGE_SIZE / pil_image.size[1])
        pil_image = pil_image.resize(
            (int(pil_image.size[0] * scale), int(pil_image.size[1] * scale)),
            Image.LANCZOS
        )
    return pil_image


async def encode_face(pil_image: Image.Image) -> Optional[list]:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=100)
    buffer.seek(0)

    image_array = face_recognition.load_image_file(buffer)
    face_locations = await run_in_threadpool(face_recognition.face_locations, image_array)
    if not face_locations:
        return None
    encodings = await run_in_threadpool(face_recognition.face_encodings, image_array, face_locations)
    return encodings[0] if encodings else None


async def search_qdrant(qdrant, encoding: list):
    def search_point():
        results = qdrant.points.search(
            collection_name=COLLECTION_NAME,
            query_vector=encoding.tolist(),
            limit=1,
            with_payload=True
        )
        return results

    results = await run_in_threadpool(search_point)
    return results[0] if results else None


async def increment_redis(redis_client, key: str):
    if redis_client:
        try:
            await redis_client.incr(key)
        except Exception as e:
            print(f"Redis increment failed: {e}")


# ========================
# Endpoints
# ========================
@router.post("/async-recognition", response_model=RecognitionResponse)
async def async_recognition(
    file: UploadFile = File(...),
    rabbitmq_channel=Depends(get_rabbitmq_channel)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    image_bytes = await file.read()
    job_id = str(uuid.uuid4())
    file_path = os.path.join(FOTOS_DIR, f"{job_id}.jpg")

    
    def write_file():
        with open(file_path, "wb") as f:
            f.write(image_bytes)
    await run_in_threadpool(write_file)

    
    message_body = json.dumps({"job_id": job_id, "path": file_path}).encode()
    await rabbitmq_channel.default_exchange.publish(
        Message(body=message_body, delivery_mode=DeliveryMode.PERSISTENT),
        routing_key=QUEUE_NAME
    )

    return RecognitionResponse(status="pending", job_id=job_id)


@router.post("/sync-recognition", response_model=RecognitionResponse)
async def sync_recognition(
    file: UploadFile = File(...),
    qdrant=Depends(get_qdrant_client),
    redis_client=Depends(get_redis_async)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    image_bytes = await file.read()
    pil_image = preprocess_image(image_bytes)

    encoding = await encode_face(pil_image)
    if encoding is None:
        raise HTTPException(status_code=400, detail="No faces found.")

    result = await search_qdrant(qdrant, encoding)
    if result:
        distance = result.score
        if distance <= CACHE_DISTANCE_THRESHOLD:
            payload = result.payload
            await increment_redis(redis_client, RECOGNITION_COUNTER_KEY)
            return RecognitionResponse(
                name=payload.get("identifier"),
                photo=payload.get("photo"),
                cached=True
            )

    return RecognitionResponse(message="Unrecognized face.", cached=False)
