import requests
from langchain_core.tools import tool
from .config import config

jina_api_key = config.get('jina', {}).get('api_key', '')
top_n = config.get('jina', {}).get('top_n', 5)
min_length = config.get('jina', {}).get('min_length', 10)

@tool
def jina_search(query: str):
    """Query in the search engine
    Args:
        query: The content to query/search for/look up/inquire about
    """
    url = f'https://s.jina.ai/{query}'
    headers = {
        'X-Retain-Images': 'none'
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

        return f"tool result: {result}"
    except requests.exceptions.Timeout:
        return "搜索超时"
    except requests.exceptions.RequestException:
        return "搜索失败"

tools = [jina_search]