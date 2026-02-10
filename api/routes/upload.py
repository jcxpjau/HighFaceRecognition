from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from PIL import Image, ImageOps
from fastapi.concurrency import run_in_threadpool
import os
import io
import numpy as np
import uuid
import face_recognition
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant import get_qdrant_client

router = APIRouter()

FOTOS_DIR = os.path.abspath(os.getenv("FOTOS_DIR", "system/photos"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "faces")
os.makedirs(FOTOS_DIR, exist_ok=True)


class UploadResponse(BaseModel):
    message: str
    identifier: str
    photo_path: str

# ========================
# Endpoint Upload
# ========================
@router.post("/upload", response_model=UploadResponse)
async def upload_face(
    qdrant: QdrantClient = Depends(get_qdrant_client),
    identifier: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")

    max_size = 1000
    original_width, original_height = pil_image.size
    if max(original_width, original_height) > max_size:
        scale = min(max_size / original_width, max_size / original_height)
        new_size = (int(original_width * scale), int(original_height * scale))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)

    save_path = os.path.join(FOTOS_DIR, f"{identifier}.jpg")
    pil_image.save(save_path, format="JPEG", quality=100)

    def get_encoding():
        image_np = np.array(pil_image)
        encodings = face_recognition.face_encodings(image_np)
        return encodings

    face_encodings_list = await run_in_threadpool(get_encoding)

    if not face_encodings_list:
        raise HTTPException(status_code=400, detail="No faces found in the image.")

    if len(face_encodings_list) > 1:
        print(f"Warning: More than one face detected in {identifier}")

    encoding = face_encodings_list[0]

    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert into Qdrant: {e}")

    return UploadResponse(
        message=f"{identifier} registered successfully!",
        identifier=identifier,
        photo_path=save_path
    )
