from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import os
from dependencies import get_redis_async

router = APIRouter()

RECOGNITION_COUNTER_KEY = os.getenv("RECOGNITION_COUNTER_KEY", "recognition_counter")

class StatsResponse(BaseModel):
    total_recognitions: int

# ========================
# Endpoint Stats
# ========================
@router.get("/stats/", response_model=StatsResponse)
async def get_stats(redis=Depends(get_redis_async)):
    if not redis:
        raise HTTPException(status_code=500, detail="Redis client not available")

    try:
        count = await redis.get(RECOGNITION_COUNTER_KEY)
        total = int(count) if count else 0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading from Redis: {e}")

    return StatsResponse(total_recognitions=total)
