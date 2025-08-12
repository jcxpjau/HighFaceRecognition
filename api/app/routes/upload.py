from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from PIL import Image, ImageOps
import os
import io
import numpy as np
import uuid
import face_recognition
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from app.qdrant import get_qdrant_client

router = APIRouter()

FOTOS_DIR = os.path.abspath("system/photos")
os.makedirs(FOTOS_DIR, exist_ok=True)
COLLECTION_NAME = "faces"

@router.post("/upload")
async def upload(
    qdrant: QdrantClient = Depends(get_qdrant_client), 
    identifier: str = Form(...), 
    file: UploadFile = File(...)
    ):
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")

    original_width, original_height = pil_image.size
    max_size = 1000
    if max(original_width, original_height) > max_size:
        scale = min(max_size / original_width, max_size / original_height)
        new_size = (int(original_width * scale), int(original_height * scale))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)

    save_path = os.path.join(FOTOS_DIR, f"{identifier}.jpg")
    pil_image.save(save_path, format="JPEG", quality=100)

    image_np = np.array(pil_image)
    face_encodings_list = face_recognition.face_encodings(image_np)
    if len(face_encodings_list) == 0:
        raise HTTPException(status_code=400, detail="No faces found")

    encoding = face_encodings_list[0]

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=encoding.tolist(),
                payload={"identifier": identifier, "photo": save_path}
            )
        ]
    )
    return {"message": f"{identifier} registered successfully!"}
