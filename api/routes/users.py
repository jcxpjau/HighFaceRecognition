from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant import get_qdrant_client
from fastapi.concurrency import run_in_threadpool
import os

router = APIRouter()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "faces")

class UserPayload(BaseModel):
    id: str
    identifier: Optional[str]
    photo: Optional[str]

class UsersResponse(BaseModel):
    users: List[UserPayload]

@router.get("/users/", response_model=UsersResponse)
async def get_users(qdrant: QdrantClient = Depends(get_qdrant_client)):
    def fetch_users():
        users = []

        points, next_page = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000
        )

        for point in points:
            users.append(UserPayload(
                id=str(point.id),
                identifier=point.payload.get("identifier") if point.payload else None,
                photo=point.payload.get("photo") if point.payload else None
            ))

        while next_page:
            points, next_page = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                offset=next_page
            )
            for point in points:
                users.append(UserPayload(
                    id=str(point.id),
                    identifier=point.payload.get("identifier") if point.payload else None,
                    photo=point.payload.get("photo") if point.payload else None
                ))

        return users

    users = await run_in_threadpool(fetch_users)
    return UsersResponse(users=users)
