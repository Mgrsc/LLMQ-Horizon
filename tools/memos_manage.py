from .config import config
from langchain_core.tools import tool
import requests

def _get_headers(memos_config):
    """获取 HTTP 请求头."""
    headers = {
        "Content-Type": "application/json",
    }
    auth_token = memos_config.get("memos_token")
    if (auth_token):
        headers["Authorization"] = f"Bearer {auth_token}"
    return headers

def _create_memo(base_url, headers, content, visibility, user_name=None):
    """创建备忘录."""
    url = f"{base_url}/api/v1/memos"
    
    formatted_content = f"{user_name}: {content}" if user_name else content
    
    payload = {
        "content": formatted_content,
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

def _search_memos(base_url, headers, page_size, user_id=None, search_keyword=None, limit=None, user_name=None):
    """检索备忘录."""
    url = f"{base_url}/api/v1/memos"
    params = {
        "pageSize": page_size,
    }
    if (user_id):
        params["filter"] = f"creator == 'users/{user_id}'"

    # 使用传入的limit，如果没有则使用page_size
    result_limit = limit if limit else page_size
    
    # 合并所有搜索关键词
    search_terms = []
    
    # 处理search_keyword
    if search_keyword:
        if isinstance(search_keyword, str):
            search_terms.extend([kw.strip() for kw in search_keyword.split(',')])
        elif isinstance(search_keyword, list):
            search_terms.extend(search_keyword)
        else:
            search_terms.append(str(search_keyword))
    
    # 添加user_name作为搜索关键词
    if user_name:
        if isinstance(user_name, str):
            search_terms.extend([name.strip() for name in user_name.split(',')])
        elif isinstance(user_name, list):
            search_terms.extend(user_name)
        else:
            search_terms.append(str(user_name))
            
    if not search_terms:
        # 无搜索关键词时的逻辑保持不变
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
                # 修改添加到combined_memos的数据结构
                combined_memos.append({
                    "name": memo["name"],
                    "updateTime": memo["updateTime"].replace("T", " ").replace("Z", ""),
                    "content": memo["content"].replace(
                        kw, f"\033[91m{kw}\033[0m"
                    )
                })
        # 去重并只取指定数量
        unique_memos = {m["name"]: m for m in combined_memos}
        result_list = list(unique_memos.values())[:result_limit]
        return {"memos": result_list}

def _delete_memo(base_url, headers, memo_ids):
    """删除备忘录.
    
    Args:
        memo_ids: 可以是单个ID或多个ID的列表
    """
    if not memo_ids:
        return {"error": "Memo not found."}
    
    # 将输入统一转换为列表
    if isinstance(memo_ids, str):
        # 支持逗号分隔的字符串
        ids = [id.strip() for id in memo_ids.split(',')]
    elif isinstance(memo_ids, list):
        ids = memo_ids
    else:
        ids = [str(memo_ids)]
    
    results = []
    for memo_id in ids:
        memo_name = f"memos/{memo_id}"
        url = f"{base_url}/api/v1/{memo_name}"
        response = requests.delete(url, headers=headers)
        results.append({
            "id": memo_id,
            "status": "success" if response.status_code == 200 else "failed",
            "message": "Delete successful." if response.status_code == 200 else f"Delete failed: {response.text}"
        })
    
    return {"results": results}

memos_config = config.get("memos", {})
@tool
def memos_manage(operation: str, create_content: str = None, search_keyword: str = None, 
                delete_id: str = None, limit: int = None, user_name: str = None):
    """Create, retrieve, and delete memos, operate on memos, and use memos.

    Args:
        operation: The type of operation, which can be "create", "search", or "delete".
        create_content: The content to be recorded in the memo. Use ###%%& as a separator for multiple memos (only used when the operation is "create"). Optional
        search_keyword: The keyword(s) to search for memos. Multiple keywords are separated by commas. If not provided, the latest memos will be returned (only used when the operation is "search"). Optional.
        delete_id: The ID(s) of the memo(s) to be deleted. Multiple IDs are supported and separated by commas (only used when the operation is "delete"). Optional.
        limit: Limits the number of search results (only used when the operation is "search"). Optional.
        user_name: When the operation is "create", it is added as a prefix to the content; when the operation is "search", it is used as a search keyword. Optional.
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
            create_result = _create_memo(base_url, headers, cnt, default_visibility, user_name)
            results.append(create_result)
        return {"results": results}

    elif (operation == "search"):
        return _search_memos(base_url, headers, default_page_size, user_id, search_keyword, limit, user_name)

    elif (operation == "delete"):
        if (not delete_id):
            return {"error": "Missing 'delete_id' for delete operation."}
        return _delete_memo(base_url, headers, delete_id)

    else:
        return {"error": f"Invalid operation: {operation}"}

tools = [memos_manage]

# 创建备忘录
# create_result = memos("create", create_content="今天我最喜欢小猫")
# print("Create Result:", create_result)

# 检索备忘录
# search_result = memos("search", search_keyword=["南瓜", "test", "小猫"])
# print("Search Result:", search_result)

# # 删除备忘录
# delete_result = memos("delete", delete_id="93")
# print("Delete Result:", delete_result)

# 删除多个备忘录
# delete_result = memos_manage("delete", delete_id="93,94,95")
# print("Delete Result:", delete_result)

# 示例：创建多条备忘录 
# """
# create_result = memos_manage("create", create_content='''今天我最喜欢小猫,它真可爱###$$$%%%明天我最喜欢小狗,它也很可爱###$$$%%%后天我最喜欢小兔子''')
# print("Create Result:", create_result)
# """