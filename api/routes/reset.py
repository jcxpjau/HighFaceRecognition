from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.http.exceptions import UnexpectedResponse
from fastapi.concurrency import run_in_threadpool
import os

from qdrant import get_qdrant_client

router = APIRouter()

# ========================
# Configs
# ========================
FOTOS_DIR = os.path.abspath(os.getenv("FOTOS_DIR", "system/photos"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "faces")
os.makedirs(FOTOS_DIR, exist_ok=True)

# ========================
# Endpoint Reset
# ========================
@router.delete("/reset/")
async def reset(qdrant: QdrantClient = Depends(get_qdrant_client)):

    # Função síncrona para deletar arquivos
    def delete_files():
        errors = []
        for filename in os.listdir(FOTOS_DIR):
            file_path = os.path.join(FOTOS_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                errors.append(f"{file_path}: {e}")
        return errors

    # Rodar em thread pool para não bloquear o event loop
    errors = await run_in_threadpool(delete_files)

    if errors:
        return JSONResponse(
            status_code=500,
            content={"message": "Some files could not be deleted", "errors": errors}
        )

    # Resetar a coleção do Qdrant
    try:
        try:
            qdrant.delete_collection(collection_name=COLLECTION_NAME)
        except UnexpectedResponse as e:
            # Ignora erro 404 (coleção não existe)
            if e.status_code != 404:
                raise

        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=128,
                distance=Distance.EUCLID
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting Qdrant collection: {e}"
        )

    return {"message": "Reset completed successfully."}
