import fal_client
from pathlib import Path
import base64
import requests
from datetime import datetime
import json
from .config import config
from .prompt.prompt import prompt_all
from langchain_core.tools import tool
from abc import ABC, abstractmethod
from enum import Enum
import os

root_path = Path(__file__).resolve().parents[1]
temp_server_dir = root_path / "temp_server" / "images"
temp_server_dir.mkdir(parents=True, exist_ok=True)

# 统一从配置中获取必要信息
create_art_config = config.get("create_art", {})
fal_key = create_art_config.get("fal_key")
glm_key = create_art_config.get("glm_key")
cloudflare_account_id = create_art_config.get("cloudflare_account_id")
cloudflare_api_token = create_art_config.get("cloudflare_api_token")
openai_base_url = create_art_config.get("openai_base_url")
openai_model = create_art_config.get("model")

# 配置需要环境变量的提供商
os.environ.update({
    "FAL_KEY": fal_key
})

class ImageSize(Enum):
    SQUARE = "square"        # 1024x1024
    PORTRAIT = "portrait"    # 768x1344
    LANDSCAPE = "landscape"  # 1344x768

class ImageSizeConverter:
    """统一图片尺寸转换器"""
    
    @staticmethod
    def to_fal_size(size: ImageSize) -> str:
        """转换为FAL支持的尺寸格式"""
        size_mapping = {
            ImageSize.SQUARE: "square_hd",
            ImageSize.PORTRAIT: "portrait_16_9",
            ImageSize.LANDSCAPE: "landscape_16_9"
        }
        return size_mapping.get(size, "square_hd")
    
    @staticmethod
    def to_glm_size(size: ImageSize) -> str:
        """转换为GLM支持的尺寸格式"""
        size_mapping = {
            ImageSize.SQUARE: "1024x1024",
            ImageSize.PORTRAIT: "768x1344",
            ImageSize.LANDSCAPE: "1344x768"
        }
        return size_mapping.get(size, "1024x1024")
    
    @staticmethod
    def to_cloudflare_params(size: ImageSize) -> dict:
        """转换为Cloudflare支持的尺寸参数"""
        size_mapping = {
            ImageSize.SQUARE: {"width": 1024, "height": 1024},
            ImageSize.PORTRAIT: {"width": 768, "height": 1344},
            ImageSize.LANDSCAPE: {"width": 1344, "height": 768}
        }
        return size_mapping.get(size, {"width": 1024, "height": 1024})

def _optimize_prompt(prompt: str) -> str:
    """使用 AI 优化绘图提示词"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {create_art_config.get('openai_api_key')}"
    }
    
    payload = {
        "model": openai_model,
        "messages": [
            {
                "role": "system",
                "content": prompt_all.get("create_art")
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{openai_base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        if result.get("choices") and len(result["choices"]) > 0:
            optimized_prompt = result["choices"][0]["message"]["content"].strip()
            print(f"Optimize Prompt: [{optimized_prompt}]")
            return optimized_prompt
        else:
            print("提示词优化失败,使用原始提示词")
            return prompt
            
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {str(e)}")
        return prompt
    except Exception as e:
        print(f"其他错误: {str(e)}")
        return prompt

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



class ModelProvider(ABC):
    """模型提供商抽象基类"""
    @abstractmethod
    def get_model_names(self) -> list:
        """返回该提供商支持的模型列表"""
        pass
    
    @abstractmethod
    def generate_image(self, model_name: str, prompt: str, image_size: ImageSize = ImageSize.SQUARE, style: str = "any") -> str:
        """生成图片的抽象方法"""
        pass

class FalProvider(ModelProvider):
    """FAL.ai 模型提供商"""
    def __init__(self, fal_key: str):
        self.api_key = fal_key
    
    def get_model_names(self) -> list:
        return ["rfv3"]
    
    def generate_image(self, model_name: str, prompt: str, image_size: ImageSize = ImageSize.SQUARE, style: str = "any") -> str:
        optimized_prompt = _optimize_prompt(prompt)
        if model_name == "rfv3":
            return self._generate_recraft_v3(optimized_prompt, image_size, style)
        return "不支持的FAL模型"
    
    def _generate_recraft_v3(self, optimized_prompt: str, image_size: ImageSize = ImageSize.SQUARE, style: str = "any") -> str:
        """ReCraft v3 模型实现"""
        fal_size = ImageSizeConverter.to_fal_size(image_size)
        result = fal_client.submit(
            "fal-ai/recraft-v3",
            arguments={
                "prompt": optimized_prompt,
                "image_size": fal_size,
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

class GlmProvider(ModelProvider):
    """智谱 GLM 模型提供商"""
    def __init__(self, glm_key: str):
        self.api_key = glm_key
    
    def get_model_names(self) -> list:
        return ["cv3p"]
    
    def generate_image(self, model_name: str, prompt: str, image_size: ImageSize = ImageSize.SQUARE, style: str = "any") -> str:
        optimized_prompt = _optimize_prompt(prompt)
        if model_name == "cv3p":
            return self._generate_cogview3(optimized_prompt, image_size)
        return "不支持的GLM模型"
    
    def _generate_cogview3(self, optimized_prompt: str, image_size: ImageSize = ImageSize.SQUARE) -> str:
        """CogView-3 模型实现"""
        glm_size = ImageSizeConverter.to_glm_size(image_size)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
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

class CloudflareProvider(ModelProvider):
    """Cloudflare AI 模型提供商"""
    def __init__(self, account_id: str, api_token: str):
        self.account_id = account_id
        self.api_token = api_token
    
    def get_model_names(self) -> list:
        return ["sdxlx1"]
    
    def generate_image(self, model_name: str, prompt: str, image_size: ImageSize = ImageSize.SQUARE, style: str = "any") -> str:
        optimized_prompt = _optimize_prompt(prompt)
        if model_name == "sdxlx1":
            return self._generate_sdxl(optimized_prompt, image_size)
        return "不支持的Cloudflare模型"
    
    def _generate_sdxl(self, optimized_prompt: str, image_size: ImageSize = ImageSize.SQUARE) -> str:
        """SDXL 模型实现"""
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/@cf/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        size_params = ImageSizeConverter.to_cloudflare_params(image_size)
        data = {
            "prompt": optimized_prompt,
            "num_steps": 20,
            "guidance": 7.5,
            **size_params
        }
            
        try:
            print(f"SDXL请求参数: {data}")
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            resp.raise_for_status()
            
            # 处理返回的二进制图片数据
            if resp.headers.get("content-type") == "image/png":
                # 生成唯一文件名
                filename = f"sdxl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                save_path = temp_server_dir / filename
                
                # 保存图片
                save_path.write_bytes(resp.content)
                print(f"SDXL图像已保存到 {save_path}")

                return f"file://{temp_server_dir}/{filename}"
            else:
                error_data = resp.json()
                return f"生成图片失败: {error_data}"
                
        except requests.exceptions.RequestException as e:
            error_msg = f"SDXL请求错误: {str(e)}"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"SDXL其他错误: {str(e)}"
            print(error_msg)
            return error_msg

class ProviderFactory:
    """模型提供商工厂类"""
    # 初始化时注入统一管理的密钥
    _providers = {
        "fal": FalProvider(fal_key),
        "glm": GlmProvider(glm_key),
        "cloudflare": CloudflareProvider(cloudflare_account_id, cloudflare_api_token)
    }
    
    _model_to_provider = {
        "rfv3": "fal",
        "cv3p": "glm",
        "sdxlx1": "cloudflare"
    }
    
    @classmethod
    def get_provider(cls, model_name: str) -> ModelProvider:
        provider_name = cls._model_to_provider.get(model_name)
        return cls._providers.get(provider_name)

@tool(parse_docstring=True)
def create_art(prompt: str, image_size: str = "square", style: str = "any", model: str = "sdxlx1"):
    """Create artwork based on the requirements and return an image link.

    Args:
        prompt: Description of the content to be drawn.
        image_size: Image size. Available values: "square", "portrait", "landscape"
        style: Image style for rfv3. Available values: "any", "realistic_image", "digital_illustration"
        model: Model name. Available values:  "sdxlx1", "cv3p","rfv3"
    """
    try:
        size_enum = ImageSize(image_size)
    except ValueError:
        size_enum = ImageSize.SQUARE
        
    provider = ProviderFactory.get_provider(model)
    if not provider:
        return "不支持的模型名称"
        
    try:
        result = provider.generate_image(model, prompt, size_enum, style)
        return result
    except Exception as e:
        return f"生成图片失败: {str(e)}"

tools = [create_art]

# print(create_art(prompt="A beautiful landscape painting"))