import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# =========================
# ENV CONFIG
# =========================
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "faces")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 128))
DISTANCE_METRIC = os.getenv("DISTANCE_METRIC", "EUCLID")

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def init_qdrant_collection():
    collections = qdrant_client.get_collections()
    existing = [c.name for c in collections.collections]
    if COLLECTION_NAME not in existing:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance[DISTANCE_METRIC]
            )
        )

def get_qdrant_client() -> QdrantClient:
    return qdrant_client
