from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import aioredis

router = APIRouter()


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    pubsub = redis_async.pubsub()
    await pubsub.subscribe(f"face_job_done:{job_id}")

    try:
        async for message in pubsub.listen():
            if message['type'] == 'message':
                data = message['data']
                await websocket.send_text(data)
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(f"face_job_done:{job_id}")
        await pubsub.close()
        await websocket.close()
