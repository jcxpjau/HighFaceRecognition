from fastapi import Request
from redis import Redis
from qdrant_client import QdrantClient

def get_redis_async(request: Request):
    return request.app.state.redis_async

def get_rabbitmq_channel(request: Request):
    return request.app.state.rabbitmq_channel

def get_redis() -> Redis:
    return Redis(host="redis", port=6379, decode_responses=True)