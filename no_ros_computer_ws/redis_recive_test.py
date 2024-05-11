import redis
import json
import base64
from PIL import Image
from io import BytesIO


def subscribe_and_process(redis_client, channel_name):
    pubsub = redis_client.pubsub()
    pubsub.subscribe(channel_name)
    print(f"Subscribed to {channel_name}")

    for message in pubsub.listen():
        if message['type'] == 'message':
            data = json.loads(message['data'])
            # decode the image from base64
            image_data = base64.b64decode(data["image"])
            image = Image.open(BytesIO(image_data))
            image.show()

            print(f"Received command {data['user_input']}")


if __name__ == "__main__":
    r = redis.Redis(host="localhost", port=6379, db=0)
    subscribe_and_process(r, 'llm_cmd_input')
