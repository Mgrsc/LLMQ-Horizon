import requests
from langchain_core.tools import tool
from .config import config

jina_api_key = config.get('jina', {}).get('api_key', '')
top_n = config.get('jina', {}).get('top_n', 5)
min_length = config.get('jina', {}).get('min_length', 10)

@tool
def jina_fact_checking(query: str):
    """待查询或确认的事实，例如“OpenAI 最新模型 o1-pro 的订阅价格是多少？”
    Args:
        query: 要查询的内容
    """
    url = f'https://g.jina.ai/{query}'
    headers = {
        'Accept': 'application/json'
    }
    
    if jina_api_key:
        headers['Authorization'] = f'Bearer {jina_api_key}'

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        lines = response.text.splitlines()
        filtered_lines = [line for line in lines if len(line.strip()) >= min_length]
        truncated_lines = filtered_lines[:top_n]
        result = '\n'.join(truncated_lines)

        return result
    except requests.exceptions.Timeout:
        return "搜索超时"
    except requests.exceptions.RequestException:
        return "搜索失败"

tools = [jina_fact_checking]