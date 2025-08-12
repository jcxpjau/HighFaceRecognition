from fastapi import APIRouter, Depends
import redis
from qdrant_client import QdrantClient
from app.qdrant import get_qdrant_client
router = APIRouter()

COLLECTION_NAME = "faces"

@router.get("/users/")
async def get_users(qdrant: QdrantClient = Depends(get_qdrant_client)):
    response = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=[0.0]*128,
        limit=1000,
        with_payload=True
    )
    
    users = []
    for point in response:
        users.append({
            "id": point.id,
            "identifier": point.payload.get("identifier"),
            "photo": point.payload.get("photo"),
        })
    return {"users": users}
