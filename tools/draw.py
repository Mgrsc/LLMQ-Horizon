import fal_client
from openai import OpenAI
import os
from pathlib import Path
import base64
import requests
from datetime import datetime
from .config import config
from .prompt.prompt import prompt_all

root_path = Path(__file__).resolve().parents[1]
temp_server_dir = root_path / "temp_server"
temp_server_dir.mkdir(parents=True, exist_ok=True)

draw_config = config.get("draw", {})
os.environ.update({
    "FAL_KEY": draw_config.get("fal_key"),
    "OPENAI_API_KEY": draw_config.get("openai_api_key"),
    "OPENAI_BASE_URL": draw_config.get("openai_base_url")
})

def save_image(url: str) -> None:
    """Save image from URL or base64 data to temp directory"""
    filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_path = temp_server_dir / filename
    
    try:
        if url.startswith("data:image"):
            image_data = base64.b64decode(url.split(",")[1])
            save_path.write_bytes(image_data)
        else:
            response = requests.get(url)
            response.raise_for_status()
            save_path.write_bytes(response.content)
        print(f"Image saved to {save_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

def fal_draw(prompt: str, image_size: str = "square_hd", style: str = "any") -> str:
    client = OpenAI()
    completion = client.chat.completions.create(
        model=draw_config.get("model"),
        messages=[
            {"role": "system", "content": prompt_all.get("draw")},
            {"role": "user", "content": prompt}
        ]
    )
    optimized_prompt = completion.choices[0].message.content.strip()
    print(f"Optimized prompt: [{optimized_prompt}]")

    result = fal_client.submit(
        "fal-ai/recraft-v3",
        arguments={
            "prompt": optimized_prompt,
            "image_size": image_size,
            "output_format": "png",
            "style": style,
            "sync_mode": True
        }
    )
    
    result = fal_client.result("fal-ai/recraft-v3", result.request_id)
    if result and result.get('images'):
        url = result['images'][0].get('url', '')
        if url:
            save_image(url)
            if url.startswith("data:image"):
                return "画好了: " + url
            return "画好了: " + url 
    return None

from langchain_core.tools import tool

@tool
def draw(prompt: str, image_size: str = "square_hd", style: str = "any"):
    """根据要求进行绘画然后返回图片链接
    Args:
         prompt: 要画的内容描述
         image_size: 图片尺寸，可选值为 "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"
         style: 图片风格，可选值为 "any", "realistic_image", "digital_illustration", "realistic_image/b_and_w", "realistic_image/hard_flash", "realistic_image/hdr", "realistic_image/natural_light", "realistic_image/studio_portrait", "realistic_image/enterprise", "realistic_image/motion_blur", "digital_illustration/pixel_art", "digital_illustration/hand_drawn", "digital_illustration/grain", "digital_illustration/infantile_sketch", "digital_illustration/2d_art_poster", "digital_illustration/handmade_3d", "digital_illustration/hand_drawn_outline", "digital_illustration/engraving_color", "digital_illustration/2d_art_poster_2"
    """
    return fal_draw(prompt, image_size, style)

tools = [draw]