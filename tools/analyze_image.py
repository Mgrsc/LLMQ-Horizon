import os
import base64
import requests
from openai import OpenAI
from langchain_core.tools import tool
from .config import config

img_config = config.get("img_analysis", {})

client = OpenAI(
    api_key=img_config.get("new_api_key"),
    base_url=img_config.get("new_base_url"),
)

@tool
def analyze_image(query: str, image_input: str) -> str:
    """Get and return the content and information in the image according to the query requirements.  It can also analyze the image.

    Args:
        query: The image information to be retrieved. e.g., "What's in the image" "Describe the image in detail" "Where is the sphere in the image" "How many people are in the image" "Give life advice based on the astrolabe in the image"
        image_input: The image source, which can be an image URL (http:// or https://), a Base64 encoded image string, or a Base64 image string with the "image/" prefix.
    """

    img_folder = img_config.get("img_folder")
    os.makedirs(img_folder, exist_ok=True)
    image_url = None

    if image_input.startswith(("http://", "https://")):
        try:
            if "multimedia.nt.qq.com.cn" in image_input and image_input.startswith("https"):
                image_input = image_input.replace("https", "http", 1)
            file_path = os.path.join(img_folder, "downloaded_image.jpg")
            response = requests.get(image_input, timeout=10)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(response.content)
            with open(file_path, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{encoded_image}"
        except requests.exceptions.RequestException as e:
            print(f"下载失败，错误：{e}")
            raise
    else:
        if image_input.startswith(("data:image/", "data:application/")):
            image_url = image_input
        else:
            image_url = f"data:image/jpeg;base64,{image_input}"

    completion = client.chat.completions.create(
        model=img_config.get("model"),
        messages=[
            {
                "role": "system",
                "content": """You are now a pair of keen eyes. Your task is to observe and understand the images uploaded by the user and, based on the user's instructions, return the required information and a detailed description of the image.

Please note the following requirements:
1. You can only output in plain text format. You cannot use any text rendering formats, such as Markdown, LaTeX, **bold**, *italics*, etc.
2. You need to extract key information from the image based on the user's instructions.
3. You need to provide a detailed description of the image, including but not limited to the overall content, colors, composition, details, etc.
4. Your description should be objective and accurate, avoiding any personal subjective assumptions.
5. Your language should be fluent, natural, and easy to understand.
6. You need to respond in the language used by the user. If the user uses Chinese, you need to respond in Chinese; if the user uses English, you need to respond in English, and so on.

After understanding the image uploaded by the user and their instructions, please begin your observation and description.""" 
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        },
                    },
                ],
            }
        ],
    )
    return f"tool result: {completion}"

tools = [analyze_image]