# tools/picture_api.py
import requests
from langchain_core.tools import tool
import os


@tool
def picture_api(api: str="https://picture.netqianye.com/api.php?type=",type: str="动漫"):
  """请求随机图片api,保存图片,并且返回图片链接。
  Args:
      api: 请求图片所需的api，需要和type元拼接然后请求。
      type: 根据用户选择请求图片的类型，类型有"动漫":动漫图片之类的、"风景":大自然风景图之类，如果不存在就使用默认类型。
  """
  try:
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取上级目录的路径
    parent_dir = os.path.dirname(current_dir)
    #请求url
    pic_response = requests.get(api)
    if pic_response:
      pic_type = pic_response.url.split('.')[-1]  # 获取后缀
      pic_response = requests.get(api+type)
      print(pic_response.url)
      with open(parent_dir+"/temp_server/random."+pic_type, 'wb') as f:
        f.write(pic_response.content)
      return pic_response.url
    else:
      return None
  except Exception as e:
    print(f"picture_api出现错误: {e}")
    return None

tools = [picture_api]
