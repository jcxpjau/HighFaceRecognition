import io
import json
import pika
from urllib.parse import quote_plus

# =========================
# ENV CONFIG
# =========================
QUEUE_NAME = os.getenv("QUEUE_NAME", "face_recognition_jobs")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", "5672")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = quote_plus(os.getenv("RABBITMQ_PASSWORD", "guest"))
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")
RABBITMQ_URL = (
    f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}"
    f"@{RABBITMQ_HOST}:{RABBITMQ_PORT}/{quote_plus(RABBITMQ_VHOST)}"
)

def publish_job_to_rabbitmq(channel, job_id: str, encoding: list[float], queue_name: str = QUEUE_NAME):
    message = json.dumps({
        "job_id": job_id,
        "encoding": encoding
    })
    channel.basic_publish(
        exchange='',
        routing_key=queue_name,
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=2,
        )
    )