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
from datetime import datetime
from typing import Dict
from random import choice
from .config import Config
from .utils import (
    extract_media_urls,
    send_in_chunks,
    get_user_name,
    build_message_content,
)
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
        self.lock = asyncio.Lock()  # 添加会话锁
        self.processing = False  # 添加处理状态标志
        self.processing_start_time = None  # 处理开始时间

    async def try_acquire_lock(self) -> bool:
        """尝试获取锁,返回是否成功"""
        if self.processing:
            # 检查是否超时(60秒)
            if self.processing_start_time and \
               (datetime.now() - self.processing_start_time).seconds > 60:
                self.processing = False
                self.processing_start_time = None
            else:
                return False
        return True

    async def start_processing(self):
        """开始处理"""
        self.processing = True
        self.processing_start_time = datetime.now()

    async def end_processing(self):
        """结束处理"""
        self.processing = False
        self.processing_start_time = None

# "group_123456_789012": Session对象1
sessions: Dict[str, Session] = {}

# 添加异步锁保护sessions字典
sessions_lock = asyncio.Lock()

CLEANUP_INTERVAL = 600  # 会话清理间隔(秒) 例:10分钟
LAST_CLEANUP_TIME = datetime.now()

async def cleanup_sessions():
    """定期清理长时间未使用(超过 CLEANUP_INTERVAL)的会话"""
    now = datetime.now()
    async with sessions_lock:
        expired_keys = []
        for k, s in sessions.items():
            if (now - s.last_accessed).total_seconds() > CLEANUP_INTERVAL:
                expired_keys.append(k)
        for k in expired_keys:
            del sessions[k]
    return len(expired_keys)


async def get_or_create_session(thread_id: str) -> Session:
    """获取或创建会话"""
    global LAST_CLEANUP_TIME
    # 定期清理
    if (datetime.now() - LAST_CLEANUP_TIME).total_seconds() > CLEANUP_INTERVAL:
        await cleanup_sessions()
        LAST_CLEANUP_TIME = datetime.now()

    # 获取锁,保证线程安全
    async with sessions_lock:
        if thread_id not in sessions:
            # 构建字典{""group_123456_789012": Session对象1}
            sessions[thread_id] = Session(thread_id)
        # 获取Session对象
        session = sessions[thread_id]
        session.last_accessed = datetime.now()
    return session


# 初始化模型和对话图
llm = None
graph_builder = None

async def initialize_resources():
    global llm, graph_builder
    if llm is None:
        llm = await get_llm()
        graph_builder = await build_graph(plugin_config, llm)


async def _process_llm_response(result: dict) -> str:
    """处理LLM返回的消息，提取回复内容"""
    if not result["messages"]:
        print("警告: 结果消息列表为空")
        return plugin_config.responses.assistant_empty_reply
        
    last_message = result["messages"][-1]
    
    if isinstance(last_message, AIMessage):
        if last_message.invalid_tool_calls:
            if isinstance(last_message.invalid_tool_calls, list) and last_message.invalid_tool_calls:
                error_msg = last_message.invalid_tool_calls[0]['error']
                print(f"工具调用错误: {error_msg}")
                return f"工具调用失败: {error_msg}"
            print("工具调用错误: 未知错误(无错误信息)")
            return "工具调用失败，但没有错误信息"
            
        if last_message.content:
            return last_message.content.strip()
            
        print("警告: AI消息内容为空")
        return "对不起，我没有理解您的问题。"
        
    if isinstance(last_message, ToolMessage) and last_message.content:
        return (last_message.content 
                if isinstance(last_message.content, str)
                else str(last_message.content))
                
    print(f"警告: 未知的消息类型或内容为空: {type(last_message)}")
    return "对不起，我没有理解您的问题。"


async def _handle_langgraph_error(e: Exception, thread_id: str) -> str:
    """处理 LangGraph 调用的异常"""
    error_message = str(e)
    print(f"调用 LangGraph 时发生错误: {error_message}")
    print(f"错误类型: {type(e)}")
    print(f"完整异常信息: {e}")
    
    if "insufficient tool messages following tool_calls message" in error_message:
        print("工具调用消息序列不完整错误，重置会话状态")
        async with sessions_lock:
            if thread_id in sessions:
                del sessions[thread_id]
        await chat_handler.finish("对话状态异常已重置，请重试")
        return
        
    return (plugin_config.responses.token_limit_error 
            if "'list' object has no attribute 'strip'" in error_message
            else plugin_config.responses.general_error)



def _chat_rule(event: Event) -> bool:
    """定义触发规则"""
    trigger_mode = plugin_config.plugin.trigger_mode
    trigger_words = plugin_config.plugin.trigger_words
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

chat_handler = on_message(rule=_chat_rule, priority=10, block=True)

@chat_handler.handle()
async def handle_chat(
    # 提取消息全部对象
    event: MessageEvent,
    # 提取各种消息段
    message: Message = EventMessage(),
    # 提取纯文本
    plain_text: str = EventPlainText(),
):
    global llm, graph_builder
    # 确保 llm 已初始化
    if llm is None:
        await initialize_resources()
    
    # 检查群聊/私聊开关，判断消息对象是否是群聊/私聊的实例
    if (isinstance(event, GroupMessageEvent) and not plugin_config.plugin.enable_group) or \
       (not isinstance(event, GroupMessageEvent) and not plugin_config.plugin.enable_private):
        await chat_handler.finish(plugin_config.responses.disabled_message)


    # ----------------- 对话消息体构建START -----------------
    # 获取用户名
    user_name = await get_user_name(event)

    # 提取媒体链接
    media_urls = await extract_media_urls(message, event.reply.message if event.reply else None)

    # 构建消息内容
    message_content = await build_message_content(message, media_urls, event, user_name)
    
    # ----------------- 对话消息体构建END -----------------

    # 构建会话ID，创建或获取Session对象
    if isinstance(event, GroupMessageEvent):
        if plugin_config.plugin.group_chat_isolation:
            thread_id = f"group_{event.group_id}_{event.user_id}"
        else:
            thread_id = f"group_{event.group_id}"
    else:
        thread_id = f"private_{event.user_id}"
    print(f"Current thread: {thread_id}")
    session = await get_or_create_session(thread_id)


    # ---------- 判断当前会话ID对应的会话是否在处理中，如无则调用langgraph START ----------
    
    # 如果锁已被占用,说明已有请求在处理,丢弃本次请求
    if session.lock.locked():
        await chat_handler.finish(Message(plugin_config.responses.session_busy_message))
        return

    # 显式获取锁,防止后续请求排队，获取失败则丢弃请求
    locked = await session.lock.acquire()
    if not locked:
        await chat_handler.finish(Message(plugin_config.responses.session_busy_message))
        return

    try:
        # 二次判断processing，获取锁之前检查当前会话是否在处理
        if session.processing:
            await chat_handler.finish(Message(plugin_config.responses.session_busy_message))
            return

        await session.start_processing()
        # 如果当前会话没有图，则创建一个
        if session.graph is None:
            session.graph = graph_builder.compile(checkpointer=session.memory)

        # 调用 LangGraph
        result = await session.graph.ainvoke(
            {"messages": [HumanMessage(content=message_content)]},
            {"configurable": {"thread_id": thread_id}},
        )
        
        print(format_messages_for_print(result["messages"]))
        
        response = await _process_llm_response(result)
    except Exception as e:
        response = await _handle_langgraph_error(e, thread_id)
    finally:
        await session.end_processing()
        # 释放锁
        session.lock.release()

    # 定义媒体类型的正则和处理函数的映射
    MEDIA_PATTERNS = {
        "image": {
            "pattern": r'(?:https?://|file:///)[^\s]+?\.(?:png|jpg|jpeg|gif|bmp|webp)',
            "segment_func": MessageSegment.image,
            "error_msg": "图片"
        },
        "video": {
            "pattern": r'https?://[^\s]+?\.(?:mp4|avi|mov|mkv)',
            "segment_func": MessageSegment.video,
            "error_msg": "视频"
        },
        "audio": {
            "pattern": r'https?://[^\s]+?\.(?:mp3|wav|ogg|aac|flac)',
            "segment_func": MessageSegment.record, 
            "error_msg": "音频"
        }
    }

    async def process_media_message(response: str, media_type: str, url: str) -> Message:
        """处理包含媒体的消息"""
        media_info = MEDIA_PATTERNS[media_type]
        print(plugin_config.plugin.media_include_text)
        if plugin_config.plugin.media_include_text:
            # 清理markdown链接语法
            message_content = re.sub(r'!?\[.*?\]\((.*?)\)', r'\1', response)
            message_content = message_content.replace(url, "").strip()
            try:
                return Message(message_content) + media_info["segment_func"](url)
            except ActionFailed:
                return Message(message_content) + MessageSegment.text(f" ({media_info['error_msg']}发送失败)")
            except MatcherException:
                raise
            except Exception as e:
                return Message(message_content) + MessageSegment.text(f" (未知错误: {e})")
        else:
            # 仅发送媒体
            try:
                return Message(media_info["segment_func"](url))
            except ActionFailed:
                return Message(f"{media_info['error_msg']}发送失败")
            except MatcherException:
                raise
            except Exception as e:
                return Message(f"未知错误: {e}")

    # 在handle_chat函数中替换原来的媒体处理代码:
    for media_type, info in MEDIA_PATTERNS.items():
        if match := re.search(info["pattern"], response, re.IGNORECASE):
            result = await process_media_message(response, media_type, match.group(0))
            await chat_handler.finish(result)

    # 处理纯文本消息
    if plugin_config.plugin.chunk.enable:
        if await send_in_chunks(response, chat_handler):
            return
    await chat_handler.finish(Message(response))












# cmd
chat_command = on_command(
    "chat",
    priority=5,
    block=True,
    permission=SUPERUSER,
)

@chat_command.handle()
async def handle_chat_command(args: Message = CommandArg(), event: Event = None):
    """处理 chat model、chat clear、chat group 等命令"""
    global llm, graph_builder, sessions, plugin_config

    command_args = args.extract_plain_text().strip().split(maxsplit=1)
    if not command_args:
        await chat_command.finish(
            """请输入有效的命令：
            'chat model <模型名字>' 切换模型 
            'chat clear' 清理会话
            'chat group <true/false>' 切换群聊会话隔离
            'chat down' 关闭对话功能
            'chat up' 开启对话功能
            'chat chunk <true/false>' 切换分开发送功能"""
            )
    command = command_args[0].lower()
    if command == "model":
        # 处理模型切换
        if len(command_args) < 2:
            try:
                current_model = llm.model_name if hasattr(llm, 'model_name') else llm.model
                await chat_command.finish(f"当前模型: {current_model}")
            except Exception as e:
                await chat_command.finish(f"获取当前模型失败: {str(e)}")
                
        model_name = command_args[1]
        try:
            new_llm = await get_llm(model_name)
            new_graph_builder = await build_graph(plugin_config, new_llm)
            # 成功创建新实例后才更新全局变量
            llm = new_llm
            graph_builder = new_graph_builder
            # 清理所有会话
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
    
    elif command == "group":
        # 处理群聊会话隔离设置
        if len(command_args) < 2:
            current_group = plugin_config.plugin.group_chat_isolation
            await chat_command.finish(f"当前群聊会话隔离: {current_group}")
        
        isolation_str = command_args[1].strip().lower()
        if isolation_str == "true":
            plugin_config.plugin.group_chat_isolation = True
        elif isolation_str == "false":
            plugin_config.plugin.group_chat_isolation = False
        else:
            await chat_command.finish("请输入 true 或 false")

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

        await chat_command.finish(
            f"已{'禁用' if not plugin_config.plugin.group_chat_isolation else '启用'}群聊会话隔离，已清理对应会话"
        )
    elif command == "down":
        plugin_config.plugin.enable_private = False
        plugin_config.plugin.enable_group = False
        await chat_command.finish("已关闭对话功能")
    elif command == "up":
        plugin_config.plugin.enable_private = True
        plugin_config.plugin.enable_group = True
        await chat_command.finish("已开启对话功能")
    elif command == "chunk":
        if len(command_args) < 2:
            await chat_command.finish(f"当前分开发送开关: {plugin_config.plugin.chunk.enable}")
        chunk_str = command_args[1].strip().lower()
        if chunk_str == "true":
            plugin_config.plugin.chunk.enable = True
            await chat_command.finish("已开启分开发送回复功能")
        elif chunk_str == "false":
            plugin_config.plugin.chunk.enable = False
            await chat_command.finish("已关闭分开发送回复功能")
        else:
            await chat_command.finish("请输入 true 或 false")
    else:
        await chat_command.finish("无效的命令，请使用 'chat model <模型名字>'、'chat clear' 或 'chat group <true/false>'。")
