import datetime
import pytz
from langchain_core.tools import tool

@tool
def get_time(timezone: str, format: str = "%Y-%m-%d %H:%M:%S"):
    """Get the current time in the specified time zone and return the time string according to the specified format.

    Args:
        timezone: Time zone name string.  For example "Asia/Shanghai", "America/New_York".
        format: Time format, for example %Y-%m-%d %H:%M:%S, Optional.
    """
    try:
        tz = pytz.timezone(timezone)
        now = datetime.datetime.now(tz)
        return now.strftime(format)
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"警告：无效的时区名称 '{timezone}'，将使用 UTC 时间。")
        tz = pytz.timezone("UTC")
        now = datetime.datetime.now(tz)
        return now.strftime(format)

tools = [get_time]
