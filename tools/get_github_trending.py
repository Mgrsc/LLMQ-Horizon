import requests
from pyquery import PyQuery as pq
import datetime

def get_github_trending():
    """Fetch the GitHub Trending page and return the content as a Markdown formatted list."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; rv:11.0) Gecko/20100101 Firefox/11.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip,deflate,sdch',
            'Accept-Language': 'zh-CN,zh;q=0.8'
        }

        url = 'https://github.com/trending'
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        doc = pq(response.content)
        items = doc('div.Box article.Box-row')

        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        trending_content = f"## {current_date}\n"

        for item in items:
            item_pq = pq(item)
            title = item_pq(".lh-condensed a").text()
            description = item_pq("p.col-9").text()
            relative_url = item_pq(".lh-condensed a").attr("href")
            absolute_url = f"https://github.com{relative_url}"

            trending_content += f"* [{title}]({absolute_url}): {description}\n"

        return trending_content

    except requests.exceptions.RequestException as e:
        error_message = f"网络请求出错了: {e}"
        return error_message
    except Exception as e:
        error_message = f"抓取 GitHub Trending 时发生了未知错误: {e}"
        return error_message
    
tools = [get_github_trending]