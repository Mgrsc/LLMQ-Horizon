import fal_client
from openai import OpenAI
import os
from pathlib import Path
import base64
import requests
from datetime import datetime
import json
from .config import config
from .prompt.prompt import prompt_all
from langchain_core.tools import tool

root_path = Path(__file__).resolve().parents[1]
temp_server_dir = root_path / "temp_server"
temp_server_dir.mkdir(parents=True, exist_ok=True)

create_art_config = config.get("create_art", {})
os.environ.update({
    "FAL_KEY": create_art_config.get("fal_key"),
    "OPENAI_API_KEY": create_art_config.get("openai_api_key"),
    "OPENAI_BASE_URL": create_art_config.get("openai_base_url")
})

def _optimize_prompt(prompt: str) -> str:
    """使用 AI 优化绘图提示词"""
    client = OpenAI()
    completion = client.chat.completions.create(
        model=create_art_config.get("model"),
        messages=[
            {"role": "system", "content": prompt_all.get("create_art")},
            {"role": "user", "content": prompt}
        ]
    )
    optimized_prompt = completion.choices[0].message.content.strip()
    print(f"优化后Prompt: [{optimized_prompt}]")
    return optimized_prompt

def _draw_via_fal(prompt: str, image_size: str = "square_hd", style: str = "any") -> str:
    """使用 FAL.ai 生成图片"""
    optimized_prompt = _optimize_prompt(prompt)
    
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
            _save_image(url)
            return url
    return None

def _convert_size_for_glm(fal_size: str) -> str:
    """将 FAL 的尺寸格式转换为 GLM 支持的尺寸格式"""
    size_mapping = {
        # 方形尺寸
        "square_hd": "1024x1024",
        "square": "1024x1024",
        # 纵向尺寸
        "portrait_4_3": "864x1152",
        "portrait_16_9": "768x1344",
        # 横向尺寸
        "landscape_4_3": "1152x864",
        "landscape_16_9": "1344x768",
        # 默认尺寸
        "default": "1024x1024"
    }
    return size_mapping.get(fal_size, size_mapping["default"])

def _draw_via_glm(prompt: str, size: str = "square_hd") -> str:
    """使用智谱 GLM 生成图片"""
    optimized_prompt = _optimize_prompt(prompt)
    glm_size = _convert_size_for_glm(size)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {create_art_config.get('glm_key')}"
    }
    
    payload = {
        "model": "cogview-3-plus",
        "prompt": optimized_prompt,
        "size": glm_size,
        "user_id": "default_user"
    }
    
    try:
        print(f"使用GLM尺寸: {glm_size}")
        response = requests.post(
            "https://open.bigmodel.cn/api/paas/v4/images/generations",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        
        result = response.json()
        if "data" in result and isinstance(result["data"], list) and result["data"]:
            url = result["data"][0].get("url")
            if url:
                _save_image(url)
                return url
            else:
                return f"未找到图片URL, 响应数据: {result}"
        else:
            return f"响应格式错误: {result}"
            
    except requests.exceptions.RequestException as e:
        error_msg = f"GLM 绘图请求错误: {str(e)}"
        print(error_msg)
        return error_msg
    except json.JSONDecodeError as e:
        error_msg = f"GLM 响应解析错误: {str(e)}"
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"GLM 绘图其他错误: {str(e)}"
        print(error_msg)
        return error_msg

def _save_image(url: str) -> None:
    """保存图片到临时目录"""
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
        print(f"图像已保存到 {save_path}")
    except Exception as e:
        print(f"保存图像出错: {e}")


@tool
def create_art(prompt: str, image_size: str = "square_hd", style: str = "any", provider: str = "glm"):
    """Create artwork based on the requirements and return an image link.

    Args:
        prompt: Description of the content to be drawn.
        image_size: Image size.  Available values for FAL: "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9".  Available values for GLM: "1024x1024", "1024x1792", "1792x1024".
        style: Image style for FAL. Available values: "any", "realistic_image", "digital_illustration", "realistic_image/b_and_w", "realistic_image/hard_flash", "realistic_image/hdr", "realistic_image/natural_light", "realistic_image/studio_portrait", "realistic_image/enterprise", "realistic_image/motion_blur", "digital_illustration/pixel_art", "digital_illustration/hand_drawn", "digital_illustration/grain", "digital_illustration/infantile_sketch", "digital_illustration/2d_art_poster", "digital_illustration/handmade_3d", "digital_illustration/hand_drawn_outline", "digital_illustration/engraving_color", "digital_illustration/2d_art_poster_2".
        provider: Drawing provider. Available values: "fal" or "glm".
    """
    if provider == "fal":
        result = _draw_via_fal(prompt, image_size, style)
        return f"Tool Response: {result}"
    elif provider == "glm":
        result = _draw_via_glm(prompt, image_size)
        return f"Tool Response: {result}"
    else:
        return "不支持的绘图提供商"

tools = [create_art]