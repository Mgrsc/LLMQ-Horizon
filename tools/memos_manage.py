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
        # 格式化返回信息
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

    # 使用传入的limit，如果没有则使用page_size
    result_limit = limit if limit else page_size

    if (not search_keyword):
        response = requests.get(url, headers=headers, params=params)
        if (response.status_code == 200):
            data = response.json()
            if ("memos" in data):
                # 过滤返回字段
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
        # 处理关键词输入
        if isinstance(search_keyword, str):
            # 支持逗号分隔的字符串
            search_keywords = [kw.strip() for kw in search_keyword.split(',')]
        elif isinstance(search_keyword, list):
            search_keywords = search_keyword
        else:
            search_keywords = [str(search_keyword)]

        combined_memos = []
        for kw in search_keywords:
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
def memos_manage(operation: str, create_content: str = None, search_keyword: str = None, delete_id: str = None, limit: int = None):
    """创建、检索、删除备忘录, 对memos进行操作

    Args:
        operation: 操作类型，可选值为 "create", "search", "delete".
        create_content: 创建备忘录的内容，使用 ###$$$%%% 作为多条备忘录的分隔符 (仅当 operation 为 "create" 时使用).可选
        search_keyword: 检索备忘录的关键词,多个关键词以逗号分隔,不传则返回最新备忘录 e.g.我喜欢吃南瓜饼->"喜欢吃","南瓜饼" (仅当 operation 为 "search" 时使用). 可选
        delete_id: 要删除的备忘录的 id,支持多个id以逗号分隔 e.g. "1,2,3" (仅当 operation 为 "delete" 时使用). 可选
        limit: 限制搜索结果数量 (仅当 operation 为 "search" 时使用). 可选
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
        MEMO_SEPARATOR = "###$$$%%%"
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