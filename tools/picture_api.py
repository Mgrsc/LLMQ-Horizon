# tools/picture_api.py
import requests
from langchain_core.tools import tool
from .config import config
import os

picture = config.get("picture_api", {})
api = picture.get("api")

@tool
def picture_api(select_type: str):
  """Select a suitable image classification request random image API from the variable api_type_list based on the request, store the image on the server, and return the image link.

  Args:
    select_type: Match the image type based on the user's request with priority. For example, "beautiful pictures" will match "beautiful", "anime pictures" will match "anime", and "shesh pictures" will match "shesh".
  """
  try:
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取上级目录的路径
    parent_dir = os.path.dirname(current_dir)
    #请求url
    pic_response = requests.get(api+select_type)
    if pic_response:
      pic_type = pic_response.url.split('.')[-1]
      print(pic_response.url)
      with open(parent_dir+"/temp_server/random."+pic_type, 'wb') as f:
        f.write(pic_response.content)
      return pic_response.url
  except Exception as e:
    print(f"picture_api出现错误: {e}")
    return None
tools = [picture_api]
