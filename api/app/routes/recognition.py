import face_recognition
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.dependencies import get_redis_async, get_rabbitmq_channel
from aio_pika import Message, DeliveryMode
import json
import uuid
import io
import numpy as np
from PIL import Image, ImageOps
from app.qdrant import get_qdrant_client
import os


router = APIRouter()

CACHE_DISTANCE_THRESHOLD = 0.5
RECOGNITION_COUNTER_KEY = "recognition_count"
COLLECTION_NAME  = "faces"
FOTOS_DIR = os.path.abspath("system/photos/recognition")
os.makedirs(FOTOS_DIR, exist_ok=True)

@router.post("/async-recognition")
async def asyncRecognitionSimple(rabbitmq_channel=Depends(get_rabbitmq_channel), file: UploadFile = File(...)):
    image_bytes = await file.read()
    job_id = str(uuid.uuid4())
    file_path = os.path.join(FOTOS_DIR, f"{job_id}.jpg")
    def write_file():
        with open(file_path, "wb") as f:
            f.write(image_bytes)

    await run_in_threadpool(write_file)
    
    message_body = json.dumps({
        "job_id": job_id,
        "path": file_path
    }).encode()

    await rabbitmq_channel.default_exchange.publish(
        Message(body=message_body, delivery_mode=DeliveryMode.PERSISTENT),
        routing_key="face_recognition_jobs"
    )

    return {"status": "pending", "job_id": job_id}

@router.post("/sync-recognition")
async def syncRecognition(
    qdrant=Depends(get_qdrant_client),
    redis_client=Depends(get_redis_async),
    rabbitmq_channel=Depends(get_rabbitmq_channel),
    file: UploadFile = File(...)
):
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")

    max_size = 1000
    if max(pil_image.size) > max_size:
        scale = min(max_size / pil_image.size[0], max_size / pil_image.size[1])
        pil_image = pil_image.resize(
            (int(pil_image.size[0] * scale), int(pil_image.size[1] * scale)),
            Image.LANCZOS
        )

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=100)
    buffer.seek(0)

    image = face_recognition.load_image_file(buffer)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        raise HTTPException(status_code=400, detail="No faces found.")

    unknown_encoding = face_recognition.face_encodings(image, face_locations)[0]
        
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=unknown_encoding.tolist(),
        limit=1,
        with_payload=True,
    )

    if results:
        point = results[0]
        distance = point.score
        if distance <= CACHE_DISTANCE_THRESHOLD:
            payload = point.payload
            return {
                "nome": payload["identifier"],
                "foto": payload["photo"],
                "cached": False
            }
    
    return {"message": "Unrecognized face."}
    
