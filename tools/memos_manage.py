from .config import config
from langchain_core.tools import tool
import requests
from typing import Union, List

def _get_headers(memos_config):
    """获取 HTTP 请求头."""
    headers = {
        "Content-Type": "application/json",
    }
    auth_token = memos_config.get("memos_token")
    if (auth_token):
        headers["Authorization"] = f"Bearer {auth_token}"
    return headers

def _create_memo(base_url, headers, content, visibility):
    """创建备忘录."""
    url = f"{base_url}/api/v1/memos"
    payload = {
        "content": content,
        "visibility": visibility,
    }
    response = requests.post(url, headers=headers, json=payload)
    if (response.status_code == 200):
        data = response.json()
        return {
            "name": data.get("name", "Unknown"),
            "time": data.get("createTime", "").replace("T", " ").replace("Z", ""),
            "content": data.get("content", "")
        }
    else:
        return {"error": f"Create failed: {response.text}"}


def _search_memos(base_url, headers, page_size, user_id=None, search_keyword=None, limit=None):
    """检索备忘录."""
    url = f"{base_url}/api/v1/memos"
    params = {
        "pageSize": page_size,
    }
    if (user_id):
        params["filter"] = f"creator == 'users/{user_id}'"

    result_limit = limit if limit else page_size
    
    search_terms = []
    
    if search_keyword:
         if isinstance(search_keyword, str):
            search_terms.extend([kw.strip() for kw in search_keyword.split(',')])
         elif isinstance(search_keyword, list):
            search_terms.extend(search_keyword)
         else:
            search_terms.append(str(search_keyword))    
            
    if not search_terms:
        response = requests.get(url, headers=headers, params=params)
        if (response.status_code == 200):
            data = response.json()
            if ("memos" in data):
                filtered_memos = [{
                    "name": memo["name"],
                    "updateTime": memo["updateTime"].replace("T", " ").replace("Z", ""),
                    "content": memo["content"]
                } for memo in data["memos"]]
                return {"memos": filtered_memos}
            else:
                print("No memos found.")
                return {"memos": []}
        else:
            return {"error": f"Search failed: {response.text}"}
    else:
        combined_memos = []
        for kw in search_terms:
            all_memos = []
            page_token = None
            while True:
                response = requests.get(url, headers=headers, params=params)
                if (response.status_code == 200):
                    data = response.json()
                    if ("memos" in data):
                        for memo in data["memos"]:
                            if (kw.lower() in memo.get("content", "").lower()):
                                all_memos.append(memo)

                    page_token = data.get("nextPageToken")
                    if (not page_token):
                        break
                    params["pageToken"] = page_token
                else:
                    return {"error": f"Search failed: {response.text}"}

            for memo in all_memos:
                combined_memos.append({
                    "name": memo["name"],
                    "updateTime": memo["updateTime"].replace("T", " ").replace("Z", ""),
                    "content": memo["content"].replace(
                        kw, f"\033[91m{kw}\033[0m"
                    )
                })
        unique_memos = {m["name"]: m for m in combined_memos}
        result_list = list(unique_memos.values())[:result_limit]
        return {"memos": result_list}

def _delete_memo(base_url, headers, memo_ids):
    """删除备忘录."""
    if not memo_ids:
        return {"error": "Memo not found."}
    
    results = []
    for memo_id in memo_ids:
        memo_name = f"memos/{memo_id}"
        url = f"{base_url}/api/v1/{memo_name}"
        response = requests.delete(url, headers=headers)
        results.append({
            "id": memo_id,
            "status": "success" if response.status_code == 200 else "failed",
            "message": "Delete successful." if response.status_code == 200 else f"Delete failed: {response.text}"
        })
    
    return {"results": results}

memos_config = config.get("memos_manage", {})
@tool(parse_docstring=True)
def memos_manage(operation: str, create_content: str = None, search_keyword: str = None, delete_id: List[int] = None, limit: int = None) -> str:
    """Create, retrieve, and delete memos, operate on memos, and use memos.

    Args:
        operation: "create", "search", or "delete".
        create_content: Memo content, including user name in the format "user_name: content" (e.g., "blaze: Today eat natto"). Multiple memos separated by "###%%&". Required for "create".
        search_keyword: Memo retrieval keywords. Retrieve the latest memo of a user using the username. If it is empty, return all the latest memos.  Only used for "search".
        delete_id: The ID of the memo(s) to be deleted.  It can be a single ID (e.g., [300]) or multiple IDs (e.g., [300, 200, 100]).
        limit: Limits the number of search results (only used when the operation is "search").
    """
    global memos_config
    if (memos_config is None):
        memos_config = {}

    base_url = memos_config.get("url")
    default_visibility = memos_config.get("default_visibility", "PRIVATE")
    default_page_size = memos_config.get("page_size", 10)
    user_id = memos_config.get("user_id")

    if (not base_url):
        return {"error": "Missing 'url' in memos_config."}

    headers = _get_headers(memos_config)

    if (operation == "create"):
        if (not create_content):
            return {"error": "Missing 'create_content' for create operation."}
        MEMO_SEPARATOR = "###%%&"
        if isinstance(create_content, str):
            contents_list = [cnt.strip() for cnt in create_content.split(MEMO_SEPARATOR) if cnt.strip()]
        elif isinstance(create_content, list):
            contents_list = create_content
        else:
            contents_list = [str(create_content)]

        results = []
        for cnt in contents_list:
            create_result = _create_memo(base_url, headers, cnt, default_visibility)
            results.append(create_result)
        return {"results": results}

    elif (operation == "search"):
        return _search_memos(base_url, headers, default_page_size, user_id, search_keyword, limit)

    elif (operation == "delete"):
        if (not delete_id):
            return {"error": "Missing 'delete_id' for delete operation."}
        return _delete_memo(base_url, headers, delete_id)

    else:
        return {"error": f"Invalid operation: {operation}"}

tools = [memos_manage]