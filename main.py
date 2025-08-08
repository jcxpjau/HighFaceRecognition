from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import face_recognition
import numpy as np
import os
import io
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
import shutil
import redis
import hashlib
import json



app = FastAPI(
    title="Avent7 Face Recognition App",
    description="API for face recognition powerd by Avent7",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diretórios
FOTOS_DIR = "photos"
os.makedirs(FOTOS_DIR, exist_ok=True)

redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

def encoding_to_key(encoding: list[float]) -> str:
    # Cria uma chave hash para o vetor de encoding
    m = hashlib.sha256()
    m.update(np.array(encoding, dtype=np.float32).tobytes())
    return m.hexdigest()

# Qdrant
qdrant = QdrantClient(host="qdrant", port=6333)
COLLECTION_NAME = "faces"

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=128,
            distance=Distance.EUCLID
        )
    )



@app.post("/upload/")
async def upload(identifier: str = Form(...), file: UploadFile = File(...)):
    # Carrega imagem
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")

    # Redimensiona se necessário
    original_width, original_height = pil_image.size
    max_size = 1000
    if max(original_width, original_height) > max_size:
        scale = min(max_size / original_width, max_size / original_height)
        new_size = (int(original_width * scale), int(original_height * scale))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)

    # Salva a imagem
    save_path = os.path.join(FOTOS_DIR, f"{identifier}.jpg")
    pil_image.save(save_path, format="JPEG", quality=100)
    image_np = np.array(pil_image)
    # Extrai encoding
    face_encodings_list = face_recognition.face_encodings(image_np)
    if len(face_encodings_list) == 0:
        return JSONResponse(status_code=400, content={"message": "No faces found"})

    encoding = face_encodings_list[0]
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=encoding.tolist(),
                payload={"identifier": identifier, "photo": save_path}
            )
        ]
    )
    return {"message": f"{identifier} registered successfully!"}


# Conexão Redis (ajuste host, port e senha conforme seu ambiente)
redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

def image_to_hash(pil_img):
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()
    return hashlib.sha256(img_bytes).hexdigest()

@app.post("/recognition/")
async def recognition(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")

    original_width, original_height = pil_image.size
    max_size = 1000
    if max(original_width, original_height) > max_size:
        scale = min(max_size / original_width, max_size / original_height)
        new_size = (int(original_width * scale), int(original_height * scale))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=100)
    buffer.seek(0)

    image = face_recognition.load_image_file(buffer)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        return JSONResponse(status_code=400, content={"message": "No faces found."})

    encodings = face_recognition.face_encodings(image, face_locations)
    unknown_encoding = encodings[0]
    key = f"face_cache:{image_to_hash(pil_image)}"

    # Tenta pegar do Redis
    cached = redis_client.get(key)
    if cached:
        result = json.loads(cached)
        return {
            "nome": result["identifier"],
            "foto": result["photo"],
            "cached": True
        }

    # Se não achou no cache, busca no Qdrant
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=unknown_encoding.tolist(),
        limit=1
    )

    if search_result and search_result[0].score <= 0.6:
        result = search_result[0].payload
        # Salva no Redis com TTL, ex 1h (3600 segundos)
        redis_client.set(key, json.dumps(result), ex=3600)

        return {
            "nome": result["identifier"],
            "foto": result["photo"],
            "cached": False
        }

    return {"message": "Unrecognized face."}


@app.delete("/reset/")
async def reset():
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
    except UnexpectedResponse as e:
        if e.status_code == 404:
            pass
        else:
            return JSONResponse(status_code=500, content={"message": f" Error on delete collection: {e}"})
    
    return {"message": "Reset completed successfully."}
