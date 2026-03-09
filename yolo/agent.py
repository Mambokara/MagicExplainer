import os
from dotenv import load_dotenv
import boto3
import mimetypes
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from ultralytics import YOLO
import cv2
import easyocr

reader = easyocr.Reader(['fr', 'en'])
load_dotenv()
console = Console()
current_messages = ""

# YOLO SIDE

MODEL_YOLO = 'model/best.pt'
model = YOLO(MODEL_YOLO)

# END OF YOLO SIDE

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
    image = get_title_crop(image_path)
    result = reader.readtext(image)
    return result[0][1]



def chat_with_agent(message):
    global current_messages
    response = client_aws.invoke_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId="uniqueID-3",
        inputText=message,
    )

    event_stream = response.get("completion")
    with Live(console=console, refresh_per_second=10) as live:
        for event in event_stream:
            chunk = event.get("chunk")
            if chunk:
                messages = chunk.get("bytes").decode()
                current_messages += messages
                live.update(chat_with_style(current_messages))
    print("\n")

def get_title_crop(image_path: str):
    img = cv2.imread(image_path)
    results = model(img)
    found = False
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            target_class_index = 0
            if cls == target_class_index and conf >= 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                titre_crop = img[y1:y2, x1:x2]
                found = True
                break
        if found:
            break
    return titre_crop


if __name__ == "__main__":
    chatting = True
    cards_path = get_cards()
    print(cards_path)
    card_names = []
    for card in cards_path:
        card_names.append(get_name(card))
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
