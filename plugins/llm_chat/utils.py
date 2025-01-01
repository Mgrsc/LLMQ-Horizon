from nonebot.adapters.onebot.v11 import (
    Message,
    MessageEvent,
    GroupMessageEvent,
    Event,
    MessageSegment,
)
from nonebot.adapters.onebot.v11.exception import ActionFailed
from datetime import datetime, timedelta, timezone
from nonebot.exception import MatcherException
from typing import List, Dict
from nonebot import get_bot
from .config import Config
from random import choice
import asyncio
import re

plugin_config = Config.load_config()


async def get_user_name(event: MessageEvent) -> str:
    """获取用户昵称，如果获取失败则使用用户ID"""
    if not plugin_config.plugin.enable_username:
        return ""

    user_name = event.sender.nickname if event.sender.nickname else event.sender.card
    if not user_name:
        try:
            if isinstance(event, GroupMessageEvent):
                user_info = await event.bot.get_group_member_info(
                    group_id=event.group_id, user_id=event.user_id
                )
                user_name = user_info.get("nickname") or user_info.get("card") or str(event.user_id)
            else:
                user_info = await event.bot.get_stranger_info(user_id=event.user_id)
                user_name = user_info.get("nickname") or str(event.user_id)
        except Exception as e:
            print(f"获取用户信息失败: {e}")
            user_name = str(event.user_id)
    return user_name

async def extract_media_urls(message: Message, reply_message: Message = None) -> Dict[str, Dict[str, List[str]]]:
    """提取消息和引用中的媒体URL"""
    media_urls = {
        "main": {"image": [], "video": [], "audio": []},
        "reply": {"image": [], "video": [], "audio": []}
    }

    def _extract_from_segments(segments: Message, media_type: str):
        return [
            seg.data["url"]
            for seg in segments
            if seg.type == media_type and seg.data.get("url")
        ]

    # 提取主消息中的媒体URL
    media_urls["main"]["image"].extend(_extract_from_segments(message, "image"))
    media_urls["main"]["video"].extend(_extract_from_segments(message, "video"))
    media_urls["main"]["audio"].extend(_extract_from_segments(message, "audio"))

    # 提取引用消息中的媒体URL
    if reply_message:
        media_urls["reply"]["image"].extend(_extract_from_segments(reply_message, "image"))
        media_urls["reply"]["video"].extend(_extract_from_segments(reply_message, "video"))
        media_urls["reply"]["audio"].extend(_extract_from_segments(reply_message, "audio"))

    return media_urls

def get_utc8_time():
    """获取当前时间的UTC+8时间"""
    utc_now = datetime.now(timezone.utc)
    utc8_time = utc_now + timedelta(hours=8)
    return utc8_time.strftime("%Y-%m-%d %H:%M:%S")

async def remove_trigger_words(message: Message, event: MessageEvent) -> str:
    """移除命令前缀,并将@CQ码转换为@真实昵称格式"""
    text = ""
    bot = get_bot()
    for seg in message:
        if seg.type == "text":
            text += seg.data["text"]
        elif seg.type == "at":
            try:
                if isinstance(event, GroupMessageEvent):
                    # 获取被@用户的群成员信息
                    user_info = await bot.get_group_member_info(
                        group_id=event.group_id,
                        user_id=int(seg.data["qq"])
                    )
                    # 优先使用群名片,其次是昵称,最后是QQ号
                    at_name = user_info.get("card") or user_info.get("nickname") or str(seg.data["qq"])
                else:
                    # 私聊场景下获取陌生人信息
                    user_info = await bot.get_stranger_info(user_id=int(seg.data["qq"]))
                    at_name = user_info.get("nickname") or str(seg.data["qq"])
            except Exception as e:
                print(f"获取被@用户信息失败: {e}")
                at_name = str(seg.data["qq"])
            text += f"@{at_name}"
            
    text = text.strip()
    
    # 移除命令前缀
    if hasattr(plugin_config.plugin, 'trigger_words'):
        for cmd in plugin_config.plugin.trigger_words:
            if text.startswith(cmd):
                text = text[len(cmd):].strip()
                break
    
    return text


def calculate_typing_delay(text: str) -> float:
    """
    计算模拟打字延迟
    基于配置的每秒处理字符数计算延迟
    """
    delay = len(text) / plugin_config.plugin.chunk.char_per_s
    return min(delay, plugin_config.plugin.chunk.max_time)



async def send_in_chunks(response: str, chat_handler) -> bool:
    """
    分段发送逻辑, 返回True表示已完成发送, 否则False
    """
    for sep in plugin_config.plugin.chunk.words:
        if sep in response:
            chunks = response.split(sep)
            for i, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if not chunk:
                    continue
                for word in plugin_config.plugin.chunk.words:
                    chunk = chunk.replace(word, "")
                chunk = chunk.strip()
                if i == len(chunks) - 1:
                    await chat_handler.finish(Message(chunk))
                else:
                    await chat_handler.send(Message(chunk))
                    await asyncio.sleep(calculate_typing_delay(chunk))
            return True
    return False



def remove_cq_codes(message: Message) -> str:
    """从消息中移除CQ码"""
    text = str(message)
    for seg in message:
        if seg.type in ["image", "audio", "video"]:
            text = text.replace(str(seg), "").strip()
    return text


async def build_message_content(
    message: Message,
    media_urls: Dict[str, Dict[str, List[str]]],
    event: MessageEvent,
    user_name: str,
) -> str:
    """构建最终发送给大模型的消息内容"""
    
    # 获取纯文本消息
    full_content = await remove_trigger_words(message, event)

    # 如果纯文本为空，则使用随机回复
    if not full_content.strip():
        full_content = choice(plugin_config.responses.empty_message_replies)

    # 添加主消息的媒体URL
    for media_type, urls in media_urls["main"].items():
        if urls:
            full_content += f"\n{media_type.capitalize()} URL：" + "\n".join(urls)

    if event.reply:
        reference_text = remove_cq_codes(event.reply.message)
        # 添加引用消息的媒体URL
        for media_type, urls in media_urls["reply"].items():
            if urls:
                reference_text += f"\n{media_type.capitalize()} URL：" + "\n".join(urls)
        if reference_text:
            full_content = f"「Quote_message：{reference_text}」\n{full_content}"

    if user_name:
        current_time = get_utc8_time()
        message_content = f"Time_now(UTC+8): {current_time}\nuser_name: {user_name}\nuser_message: {full_content}"
    else:
        message_content = full_content

    return message_content




