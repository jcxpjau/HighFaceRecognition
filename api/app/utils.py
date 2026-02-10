import io
import hashlib
import numpy as np
from PIL import Image
import pika
import json
import face_recognition

def publish_job_to_rabbitmq(channel, job_id: str, encoding: list[float]):
    message = json.dumps({
        "job_id": job_id,
        "encoding": encoding
    })
    channel.basic_publish(
        exchange='',
        routing_key='face_recognition_jobs',
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=2,
        )
    )