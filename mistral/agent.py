import os
import threading
import time
import itertools
from mistralai import Mistral
from dotenv import load_dotenv
import boto3
import mimetypes
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

load_dotenv()
console = Console()
current_messages = ""

# MISTRAL SIDE

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-medium-latest"

client = Mistral(api_key=api_key)

# END OF MISTRAL SIDE

# AWS SIDE

access_key = os.environ["ACCESS_KEY"]
secret_key = os.environ["SECRET_ACCESS_KEY"]
region = os.environ["REGION"]
agent_id = os.environ["AGENT_ID"]
alias_id = os.environ["ALIAS_ID"]

client_aws = boto3.client(
    "bedrock-agent-runtime",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=region,
)

# END OF AWS SIDE


def chat_with_style(msg_complete):
    md = Markdown(msg_complete)
    return md


def load_image(image_path):
    import base64

    mime_type, _ = mimetypes.guess_type(image_path)
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    base64_encoded = base64.b64encode(image_data).decode("utf-8")
    base64_url = f"data:{mime_type};base64,{base64_encoded}"
    return base64_url


def get_cards():
    user_input = input(
        "Enter the path of the cards you want to check separated by ','. "
    )
    return user_input.split(", ")


def get_name(image_path: str) -> str:
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": 'Extrais le nom de cette carte Magic the Gathering et ne réponds que avec le nom de la carte magic comme par exemple "black lotus"',
                    },
                    {"type": "image_url", "image_url": load_image(image_path)},
                ],
            }
        ],
    )
    return chat_response


def chat_with_agent(message):
    global current_messages
    response = client_aws.invoke_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId="uniqueID",
        inputText=message,
    )

    event_stream = response.get("completion")
    first_chunk_received = threading.Event()

    with Live(console=console, refresh_per_second=10) as live:
        def loading_animation():
            frames = itertools.cycle([". ", ".. ", "... "])
            while not first_chunk_received.is_set():
                live.update(next(frames))
                time.sleep(0.4)

        loader = threading.Thread(target=loading_animation, daemon=True)
        loader.start()

        for event in event_stream:
            first_chunk_received.set()
            chunk = event.get("chunk")
            if chunk:
                messages = chunk.get("bytes").decode()
                current_messages += messages
                live.update(chat_with_style(current_messages))

        loader.join()
    print("\n")


if __name__ == "__main__":
    chatting = True
    cards_path = get_cards()
    print(cards_path)
    card_names = []
    for card in cards_path:
        card_names.append(get_name(card).choices[0].message.content)
    print(f"Voici votre liste de carte : {', '.join(card_names)}.")
    message = input("Indiquez votre question pour l'agent : ")
    full_message = f"Voici les cartes que j'ai : {', '.join(card_names)}. \n\n{message}"
    chat_with_agent(full_message)
    while chatting:
        msg = input(">> ")
        if msg == "stop":
            chatting = False
            break
        chat_with_agent(msg)
