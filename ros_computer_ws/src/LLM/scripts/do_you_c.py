import os
from openai import OpenAI

# TODO : add key to script
client = OpenAI(api_key=os.getenv("openaikey"))

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What you think the person in the image is doing?"},
                {
                    "type": "image_url",
                    "image_url": "https://img-s-msn-com.akamaized.net/tenant/amp/entityid/AA18Lnc8.img?w=1920&h=1080&q=60&m=2&f=jpg",
                },
            ],
        }
    ],
    max_tokens=300,
)