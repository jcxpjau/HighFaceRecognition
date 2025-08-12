from fastapi import APIRouter, Depends
import redis
from app.dependencies import get_redis_async
from qdrant_client import QdrantClient

router = APIRouter()


@router.get("/stats/")
async def get_stats(redis=Depends(get_redis_async)):
    count = await redis.get("recognition_count")
    return {"total_recognitions": int(count) if count else 0}
