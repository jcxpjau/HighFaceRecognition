from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

COLLECTION_NAME = "faces"

qdrant_client = QdrantClient(host="qdrant", port=6333)

def init_qdrant_collection():
    collections = qdrant_client.get_collections()
    existing = [c.name for c in collections.collections]
    if COLLECTION_NAME not in existing:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=128, distance=Distance.EUCLID)
        )


def get_qdrant_client() -> QdrantClient:
    return qdrant_client