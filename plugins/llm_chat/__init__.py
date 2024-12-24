from nonebot.adapters.onebot.v11 import (
    Message,
    MessageEvent,
    GroupMessageEvent,
    Event,
    MessageSegment,
)
from nonebot.permission import SUPERUSER
from nonebot import on_message, on_command
from nonebot.params import CommandArg, EventMessage, EventPlainText
from nonebot.exception import MatcherException
from nonebot.plugin import PluginMetadata
from nonebot.adapters.onebot.v11.exception import ActionFailed
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from .graph import build_graph, get_llm, format_messages_for_print
from typing import Dict
from datetime import datetime
from random import choice
from .config import Config
import asyncio
import os
import re




__plugin_meta__ = PluginMetadata(
    name="LLM Chat",
    description="基于 LangGraph 的chatbot",
    usage="@机器人 或关键词，或使用命令前缀触发对话",
    config=Config,
)

plugin_config = Config.load_config()

os.environ["OPENAI_API_KEY"] = plugin_config.llm.api_key
os.environ["OPENAI_BASE_URL"] = plugin_config.llm.base_url
os.environ["GOOGLE_API_KEY"] = plugin_config.llm.google_api_key

# 会话模板
class Session:
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.memory = MemorySaver()
        # 最后访问时间
        self.last_accessed = datetime.now()
        self.graph = None

# "group_123456_789012": Session对象1
sessions: Dict[str, Session] = {}

# 添加异步锁保护sessions字典
sessions_lock = asyncio.Lock()

async def get_or_create_session(thread_id: str) -> Session:
    """异步获取或创建会话"""
    async with sessions_lock:
        if thread_id not in sessions:
            sessions[thread_id] = Session(thread_id)
        session = sessions[thread_id]
        session.last_accessed = datetime.now()
    return session

async def cleanup_old_sessions():
    """异步清理过期的会话"""
    async with sessions_lock:
        if len(sessions) > plugin_config.plugin.max_sessions:
            # 按最后访问时间排序，删除最旧的会话
            sorted_sessions = sorted(
                sessions.items(),
                key=lambda x: x[1].last_accessed,
                reverse=True
            )
            # 保留配置指定数量的会话
            for thread_id, _ in sorted_sessions[plugin_config.plugin.max_sessions:]:
                del sessions[thread_id]

# 初始化模型和对话图
llm = get_llm()
graph_builder = build_graph(plugin_config, llm)


def chat_rule(event: Event) -> bool:
    """定义触发规则"""
    trigger_mode = plugin_config.plugin.Trigger_mode
    trigger_words = plugin_config.plugin.Trigger_words
    msg = str(event.get_message())

    if "at" in trigger_mode and event.is_tome():
        return True
    if "keyword" in trigger_mode:
        for word in trigger_words:
            if word in msg:
                return True
    if "prefix" in trigger_mode:
        for word in trigger_words:
            if msg.startswith(word):
                return True
    if not trigger_mode:
        return event.is_tome()
    return False

chat_handler = on_message(rule=chat_rule, priority=10, block=True)

def remove_trigger_words(text: str, message: Message) -> str:
    """移除命令前缀(包括@和昵称)，保留关键词"""
    # 删除所有@片段
    text = str(message).strip()
    for seg in message:
        if seg.type == "at":
            text = text.replace(str(seg), "").strip()
    
    # 移除命令前缀
    if hasattr(plugin_config.plugin, 'Trigger_words'):
        for cmd in plugin_config.plugin.Trigger_words:
            if text.startswith(cmd):
                text = text[len(cmd):].strip()
                break
    
    return text

@chat_handler.handle()
async def handle_chat(
    # 提取消息全部对象
    event: MessageEvent,
    # 提取各种消息段
    message: Message = EventMessage(),
    # 提取纯文本
    plain_text: str = EventPlainText()
):
    
    # 检查群聊/私聊开关，判断消息对象是否是群聊/私聊的实例
    if (isinstance(event, GroupMessageEvent) and not plugin_config.plugin.enable_group) or \
       (not isinstance(event, GroupMessageEvent) and not plugin_config.plugin.enable_private):
        await chat_handler.finish(plugin_config.plugin.disabled_message)
        
    # 获取用户名
    user_name = ""  # 初始化为空字符串
    if plugin_config.plugin.enable_username:
        user_name = event.sender.nickname if event.sender.nickname else event.sender.card
        if not user_name:
            try:
                if isinstance(event, GroupMessageEvent):
                    user_info = await event.bot.get_group_member_info(group_id=event.group_id, user_id=event.user_id)
                    user_name = user_info.get("nickname") or user_info.get("card") or str(event.user_id)
                else:
                    user_info = await event.bot.get_stranger_info(user_id=event.user_id)
                    user_name = user_info.get("nickname") or str(event.user_id)
            except Exception as e:
                print(f"获取用户信息失败: {e}")
                user_name = str(event.user_id)
    print(user_name)
    image_urls = [
        seg.data["url"]
        for seg in message
        if seg.type == "image" and seg.data.get("url")
    ]
    if event.reply:
        image_urls.extend(
            seg.data["url"]
            for seg in event.reply.message
            if seg.type == "image" and seg.data.get("url")
        )
    
    # 提取视频链接
    video_urls = [
        seg.data["url"]
        for seg in message
        if seg.type == "video" and seg.data.get("url")
    ]
    if event.reply:
       video_urls.extend(
            seg.data["url"]
            for seg in event.reply.message
            if seg.type == "video" and seg.data.get("url")
       )

    # 处理消息内容,移除触发词
    full_content = remove_trigger_words(plain_text, message)
    
    # 如果全是空白字符,使用配置中的随机回复
    if not full_content.strip():
        if hasattr(plugin_config.plugin, "empty_message_replies"):
            reply = choice(plugin_config.plugin.empty_message_replies)
            await chat_handler.finish(Message(reply))
        else:
            await chat_handler.finish("您想说什么呢?")
    
    if image_urls:
        full_content += "\n图片URL：" + "\n".join(image_urls)
    if video_urls:
        full_content += "\n视频URL：" + "\n".join(video_urls)
    
    # 构建会话ID
    if isinstance(event, GroupMessageEvent):
        if plugin_config.plugin.group_chat_isolation:
            thread_id = f"group_{event.group_id}_{event.user_id}"
        else:
            thread_id = f"group_{event.group_id}"
    else:
        thread_id = f"private_{event.user_id}"
    print(f"Current thread: {thread_id}")
    await cleanup_old_sessions()
    session = await get_or_create_session(thread_id)
    # 如果当前会话没有图，则创建一个
    if session.graph is None:
        session.graph = graph_builder.compile(checkpointer=session.memory)
    try:
        # 在发送给 LangGraph 的消息内容中添加用户名
        if plugin_config.plugin.enable_username and user_name:
            message_content = f"{user_name}: {full_content}"
        else:
            message_content = full_content
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            session.graph.invoke,
            {"messages": [HumanMessage(content=message_content)]},
            {"configurable": {"thread_id": thread_id}},
        )
        formatted_output = format_messages_for_print(result["messages"])
        print(formatted_output)
        if not result["messages"]:
            response = "对不起，我现在无法回答。"
        else:
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                if last_message.invalid_tool_calls:
                    if isinstance(last_message.invalid_tool_calls, list) and last_message.invalid_tool_calls:
                        response = f"工具调用失败: {last_message.invalid_tool_calls[0]['error']}"
                    else:
                        response = "工具调用失败，但没有错误信息"
                elif last_message.content:
                    response = last_message.content.strip()
                else:
                    response = "对不起，我没有理解您的问题。"
            elif isinstance(last_message, ToolMessage) and last_message.content:
                response = (
                    last_message.content
                    if isinstance(last_message.content, str)
                    else str(last_message.content)
                )
            else:
                response = "对不起，我没有理解您的问题。"
    except Exception as e:
        print(f"调用 LangGraph 时发生错误: {e}")
        async with sessions_lock:
            if thread_id in sessions:
                del sessions[thread_id]
        response = f"""卧槽，报错了：{e}\n尝试自行修复中，聊聊别的吧！"""
        
    # 检查是否有图片或视频链接，并发送图片或视频或文本消息
    image_match = re.search(r'https?://[^\s]+?\.(?:png|jpg|jpeg|gif|bmp|webp|svg)', response, re.IGNORECASE)
    video_match = re.search(r'https?://[^\s]+?\.(?:mp4|avi|mov|mkv)', response, re.IGNORECASE)
    if image_match:
        image_url = image_match.group(0)
        message_content = response.replace(image_url, "")
        if image_url.endswith(".svg"):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(image_url)
                    resp.raise_for_status()
                    svg_data = resp.content
                filename = f"{uuid.uuid4().hex}.png"
                output_path = Path("temp_server") / filename
                
                subprocess.run(
                    ["rsvg-convert", "-f", "png", "-o", str(output_path)],
                    input=svg_data,
                    check=True,
                )
                
                with open(output_path,"rb") as f:
                    image_data = f.read()
                base64_data = base64.b64encode(image_data).decode()
                image_segment = MessageSegment.image(f"base64://{base64_data}")
                await chat_handler.finish(Message(message_content) + image_segment)
                
            except MatcherException:
                raise
            except Exception as e:
                await chat_handler.finish(Message(message_content) + MessageSegment.text(f" (未知错误: {e})"))
        else:
            try:
                await chat_handler.finish(Message(message_content) + MessageSegment.image(image_url))
            except ActionFailed:
                await chat_handler.finish(Message(message_content) + MessageSegment.text(" (图片发送失败)"))
            except MatcherException:
                raise
            except Exception as e :
                 await chat_handler.finish(Message(message_content) + MessageSegment.text(f" (未知错误： {e})"))
    elif video_match:
        video_url = video_match.group(0)
        message_content = response.replace(video_url, "")
        try:
            await chat_handler.finish(Message(message_content) + MessageSegment.video(video_url))
        except ActionFailed:
            await chat_handler.finish(Message(message_content) + MessageSegment.text(" (视频发送失败)"))
        except MatcherException:
            raise
        except Exception as e:
            await chat_handler.finish(Message(message_content) + MessageSegment.text(f" (未知错误： {e})"))
    else:
        await chat_handler.finish(Message(response))





group_chat_isolation = on_command(
    "chat group", 
    priority=5, 
    block=True, 
    permission=SUPERUSER,
)

@group_chat_isolation.handle()
async def handle_group_chat_isolation(args: Message = CommandArg(), event: Event = None):
    global plugin_config, sessions
    
    # 切换群聊会话隔离
    isolation_str = args.extract_plain_text().strip().lower()
    if not isolation_str:
        current_group = plugin_config.plugin.group_chat_isolation
        await group_chat_isolation.finish(f"当前群聊会话隔离: {current_group}")
    
    if isolation_str == "true":
        plugin_config.plugin.group_chat_isolation = True
    elif isolation_str == "false":
        plugin_config.plugin.group_chat_isolation = False
    else:
        await group_chat_isolation.finish("请输入 true 或 false")

    # 清理对应会话
    keys_to_remove = []
    if isinstance(event, GroupMessageEvent):
        prefix = f"group_{event.group_id}"
        if plugin_config.plugin.group_chat_isolation:
            keys_to_remove = [key for key in sessions if key.startswith(f"{prefix}_")]
        else:
            keys_to_remove = [key for key in sessions if key == prefix]
    else:
        keys_to_remove = [key for key in sessions if key.startswith("private_")]

    async with sessions_lock:
        for key in keys_to_remove:
            del sessions[key]


    await group_chat_isolation.finish(
        f"已{'禁用' if not plugin_config.plugin.group_chat_isolation else '启用'}群聊会话隔离，已清理对应会话"
    )





# 模型切换和清理历史会话
chat_command = on_command(
    "chat",
    priority=5,
    block=True,
    permission=SUPERUSER,
)

@chat_command.handle()
async def handle_chat_command(args: Message = CommandArg()):
    """处理 chat model 和 chat clear 命令"""
    global llm, graph_builder, sessions, plugin_config

    command_args = args.extract_plain_text().strip().split(maxsplit=1)
    if not command_args:
        await chat_command.finish(
            """请输入有效的命令：
            'chat model <模型名字>' 切换模型 
            'chat clear' 清理会话"""
            )
    command = command_args[0].lower()
    if command == "model":
        # 处理模型切换
        if len(command_args) < 2:
            try:
                current_model = llm.model_name
            except AttributeError:
                current_model = llm.model
            await chat_command.finish(f"当前模型: {current_model}")
        model_name = command_args[1]
        try:
            llm = get_llm(model_name)
            graph_builder = build_graph(plugin_config, llm)
            async with sessions_lock:
                sessions.clear()
            await chat_command.finish(f"已切换到模型: {model_name}")
        except MatcherException:
            raise
        except Exception as e:
            await chat_command.finish(f"切换模型失败: {str(e)}")
            
    elif command == "clear":
        # 处理清理历史会话
        async with sessions_lock:
            sessions.clear()
        await chat_command.finish("已清理所有历史会话。")
    
    elif command == "down":
        plugin_config.plugin.enable_private = False
        plugin_config.plugin.enable_group = False
        await chat_command.finish("已关闭对话功能")
    elif command == "up":
        plugin_config.plugin.enable_private = True
        plugin_config.plugin.enable_group = True
        await chat_command.finish("已开启对话功能")
    else:
        await chat_command.finish("无效的命令，请使用 'chat model <模型名字>' 或 'chat clear'.")
