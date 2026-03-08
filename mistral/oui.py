import os
from mistralai import Mistral
from dotenv import load_dotenv
import mimetypes


load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-medium-latest"

client = Mistral(api_key=api_key)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": "https://docs.mistral.ai/img/eiffel-tower-paris.jpg",
            },
        ],
    }
]


def load_image(image_path):
    import base64

    mime_type, _ = mimetypes.guess_type(image_path)
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    base64_encoded = base64.b64encode(image_data).decode("utf-8")
    base64_url = f"data:{mime_type};base64,{base64_encoded}"
    return base64_url


chat_response = client.chat.complete(
    model=model,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extrais le nom de cette carte Magic the Gathering",
                },
                {"type": "image_url", "image_url": load_image("./test.jpg")},
            ],
        }
    ],
)

chat_response = client.chat.complete(model=model, messages=messages)
