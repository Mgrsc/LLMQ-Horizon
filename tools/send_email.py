import resend
import requests
import json
from langchain_core.tools import tool
from .config import config
from .prompt.prompt import prompt_all

smtp_config = config.get("send_email", {})
resend_api_key = smtp_config.get("resend_api_key")
from_email = smtp_config.get("from_email")
openai_api_key = smtp_config.get("openai_api_key")
openai_base_url = smtp_config.get("openai_base_url", "https://api.openai.com/v1")
model = smtp_config.get("model", "deepseek-chat")
format_json = smtp_config.get("format_json", True)
timeout = smtp_config.get("timeout", 60)

def optimize_content(subject: str = "", content: str = "", draft_desc: str = "") -> tuple:
    """使用大模型优化邮件主题和内容"""
    # 如果未配置API key，直接返回原始内容
    if not openai_api_key:
        print("未配置OpenAI API Key，跳过内容优化")
        return subject, content
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    
    # 根据是否有主题和内容构建提示词
    prompt_parts = []
    if subject:
        prompt_parts.append(f"主题：{subject}")
    if content:
        prompt_parts.append(f"内容：{content}")
    if draft_desc:
        prompt_parts.append(f"代写要求：{draft_desc}")
    
    prompt = "\n".join(prompt_parts)
    
    def parse_response(response_text: str) -> tuple:
        """解析模型响应，返回主题和内容"""
        print(f"[DEBUG] 原始响应: {response_text}")
        try:
            response_data = json.loads(response_text)
            subject = response_data.get("subject", "")
            content = response_data.get("content", "")
            print(f"[DEBUG] 解析后的主题: {subject}")
            print(f"[DEBUG] 解析后的内容长度: {len(content)} 字符")
            print(f"[DEBUG] 解析后的内容预览: {content[:200]}...")
            if subject and content:
                return subject.replace('"', '\\"'), content.replace('"', '\\"') 
        except json.JSONDecodeError:
            pass
        return None, None
    
    def make_api_call() -> tuple:
        """发起API调用并处理响应"""
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_all.get("send_email")
                },
                {
                    "role": "assistant",
                    "content": "好的，我将严格遵循你的要求编写邮件，并且严格遵守输出格式使用json格式进行输出"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.6,
            "max_tokens": 8192,
            "top_p": 0.8,
        }
        
        if format_json:
            data["response_format"] = {
                "type": "json_object"
            }
            
        api_url = f"{openai_base_url}/chat/completions"
        response = requests.post(api_url, headers=headers, json=data, timeout=timeout)
        result = response.json()
        response_text = result['choices'][0]['message']['content']
        return parse_response(response_text)
    
    # 主处理逻辑，包含重试机制
    try:
        result_subject, result_content = make_api_call()
        if result_subject and result_content:
             return result_subject, result_content
        return subject, content
            
    except Exception as e:
        print(f"内容优化失败: {e}")
        return subject, content

@tool(parse_docstring=True)
def send_email(email: str, subject: str = "", content: str = "", optimize: bool = True, draft_desc: str = "") -> str:
    """Send emails or draft and send emails

    Args:
        email (str): Email address of the recipient
        subject (str): Email subject (optional if using draft_desc)
        content (str): Email body content (optional if using draft_desc)
        optimize (bool): Whether to optimize the content using LLM, defaults to True
        draft_desc (str): Description and reference content for AI-drafted emails (e.g. "写一封邀请朋友周末聚餐的邮件，语气要温和，500字左右", "写一封邀请朋友周末聚餐的邮件，语气要温和，参考内容：xxxxx")
    """
    try:
        if draft_desc:
            subject, content = optimize_content(draft_desc=draft_desc)
            print(f"邮件代写成功: {subject}")
        elif optimize:
            subject, content = optimize_content(subject, content)
            print(f"邮件优化成功: {subject}")
            
        if not content or content.strip() == "":
            raise Exception("邮件内容为空")
            
        resend.api_key = resend_api_key
        to_emails = email if isinstance(email, list) else [email]
        
        params = {
            "from": from_email,
            "to": to_emails,
            "subject": subject,
            "html": content,
        }
        
        response = resend.Emails.send(params)
        print("邮件发送成功!")
        return response
        
    except Exception as e:
        error_msg = f"邮件发送失败: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

tools = [send_email]