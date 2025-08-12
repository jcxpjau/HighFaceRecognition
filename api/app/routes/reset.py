from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
import os
from qdrant_client.http.models import VectorParams
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance
from app.qdrant import get_qdrant_client

router = APIRouter()

FOTOS_DIR = os.path.abspath("system/photos")
os.makedirs(FOTOS_DIR, exist_ok=True)
COLLECTION_NAME = "faces"

@router.delete("/reset/")
async def reset(
    qdrant: QdrantClient = Depends(get_qdrant_client)
):
    if os.path.exists(FOTOS_DIR):
        for filename in os.listdir(FOTOS_DIR):
            file_path = os.path.join(FOTOS_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                return JSONResponse(status_code=500, content={"message": f"Error on delete {file_path}: {e}"})

    try:
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=128,
                distance=Distance.EUCLID
            )
        )
    except UnexpectedResponse as e:
        if e.status_code != 404:
            return JSONResponse(status_code=500, content={"message": f"Error on delete collection: {e}"})

    return {"message": "Reset completed successfully."}
