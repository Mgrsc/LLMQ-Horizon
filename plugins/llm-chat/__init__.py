from typing import Annotated, Dict
from nonebot import on_message, on_command
from nonebot.adapters.onebot.v11 import Message, MessageEvent, GroupMessageEvent
from nonebot.plugin import PluginMetadata
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from .config import Config
import os
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
from nonebot.params import CommandArg, EventMessage, EventPlainText, EventToMe
from nonebot.exception import MatcherException
from random import choice

__plugin_meta__ = PluginMetadata(
    name="LLM Chat",
    description="基于 LangGraph 的chatbot",
    usage="@机器人 进行对话，或使用命令前缀触发对话",
    config=Config,
)

plugin_config = Config.load_config()

os.environ["OPENAI_API_KEY"] = plugin_config.llm.api_key
os.environ["OPENAI_BASE_URL"] = plugin_config.llm.base_url
os.environ["GOOGLE_API_KEY"] = plugin_config.llm.google_api_key

class State(TypedDict):
    messages: Annotated[list, add_messages]

# 会话模板
class Session:
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.memory = MemorySaver()
        self.last_accessed = datetime.now()
        self.graph = None

sessions: Dict[str, Session] = {}

def get_or_create_session(thread_id: str) -> Session:
    """获取或创建会话"""
    if thread_id not in sessions:
        sessions[thread_id] = Session(thread_id)
    session = sessions[thread_id]
    session.last_accessed = datetime.now()
    return session

def cleanup_old_sessions():
    """清理过期的会话"""
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

def cleanup_old_messages(session: Session):
    """清理过期的消息"""
    max_messages = plugin_config.plugin.max_messages_per_session
    if hasattr(session.memory, 'messages') and len(session.memory.messages) > max_messages:
        # 只保留最新的消息
        session.memory.messages = session.memory.messages[-max_messages:]

graph_builder = StateGraph(State)

def get_llm():
    """根据配置获取适当的 LLM 实例"""
    if plugin_config.llm.provider == "google":
        return ChatGoogleGenerativeAI(
            model=plugin_config.llm.model,
            temperature=plugin_config.llm.temperature,
            max_tokens=plugin_config.llm.max_tokens,
        )
    else:  # 默认使用 OpenAI
        return ChatOpenAI(
            model=plugin_config.llm.model,
            temperature=plugin_config.llm.temperature,
            max_tokens=plugin_config.llm.max_tokens,
        )

# 替换原有的 llm 初始化
llm = get_llm()

from .tools import load_tools
tools = load_tools()  # 直接从tools.py加载工具，不需要传参
llm_with_tools = llm.bind_tools(tools)

from langchain_core.messages import trim_messages
trimmer = trim_messages(
    strategy="last",
    max_tokens=plugin_config.llm.max_context_messages,
    token_counter=len,
    include_system=True,
    allow_partial=False,
    start_on="human",
    end_on=("human", "tool"),
)


def chatbot(state: State):
    messages = state["messages"]
    if plugin_config.llm.system_prompt:
        messages = [SystemMessage(content=plugin_config.llm.system_prompt)] + messages
    
    trimmed_messages = trimmer.invoke(messages)
    print(trimmed_messages)
    response = llm_with_tools.invoke(trimmed_messages)
    
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

def check_trigger(event: MessageEvent) -> bool:
    """检查是否触发对话
    支持三种触发方式：
    1. 艾特触发（默认）
    2. 关键词触发
    3. 前缀触发
    """

    msg_text = event.get_plaintext()
    
    # 命令触发
    if hasattr(plugin_config.plugin, 'command_start') and plugin_config.plugin.command_start:
        if any(msg_text.startswith(cmd) for cmd in plugin_config.plugin.command_start):
            return True
    
    # 关键词触发
    if hasattr(plugin_config.plugin, 'keywords') and plugin_config.plugin.keywords:
        if any(keyword in msg_text for keyword in plugin_config.plugin.keywords):
            return True
    
    # 默认艾特触发
    if (not hasattr(plugin_config.plugin, 'command_start') and 
        not hasattr(plugin_config.plugin, 'keywords')) or \
        plugin_config.plugin.need_at:
        return event.is_tome()
    
    return False

chat_handler = on_message(rule=check_trigger, priority=5, block=True)

def remove_trigger_words(text: str, message: Message) -> str:
    """移除所有触发词,包括@和昵称"""
    # 删除所有@片段
    text = str(message).strip()  # 转换成字符串处理
    for seg in message:
        if seg.type == "at":
            text = text.replace(str(seg), "").strip()
    
    # 移除命令前缀
    if hasattr(plugin_config.plugin, 'command_start'):
        for cmd in plugin_config.plugin.command_start:
            if text.startswith(cmd):
                text = text[len(cmd):].strip()
                break
    
    # 移除关键词
    if hasattr(plugin_config.plugin, 'keywords'):
        for keyword in plugin_config.plugin.keywords:
            text = text.replace(keyword, "").strip()
    
    return text

@chat_handler.handle()
async def handle_chat(
    event: MessageEvent,
    is_tome: bool = EventToMe(),
    message: Message = EventMessage(),
    plain_text: str = EventPlainText()
):
    # 检查群聊/私聊开关
    if isinstance(event, GroupMessageEvent) and not plugin_config.plugin.enable_group:
        return
    if not isinstance(event, GroupMessageEvent) and not plugin_config.plugin.enable_private:
        return
    
    image_urls = []
    for seg in message:
        if seg.type == "image" and seg.data.get("url"):
            image_urls.append(seg.data["url"])
    
    if event.reply:
        reply_message = event.reply.message
        for seg in reply_message:
            if seg.type == "image" and seg.data.get("url"):
                image_urls.append(seg.data["url"])

    # 处理消息内容,移除触发词
    full_content = remove_trigger_words(plain_text, message)
    
    # 如果全是空白字符,使用配置中的随机回复
    if not full_content.strip():
        if hasattr(plugin_config.plugin, 'empty_message_replies'):
            print("ok")
            reply = choice(plugin_config.plugin.empty_message_replies)
            await chat_handler.finish(Message(reply))
        else:
            print("no")
            await chat_handler.finish("您想说什么呢?")
    
    if image_urls:
        full_content += "\n图片：" + "\n".join(image_urls)
    
    # 构建会话ID
    thread_id = (
        f"group_{event.group_id}_{event.user_id}" 
        if isinstance(event, GroupMessageEvent)
        else f"private_{event.user_id}"
    )
    
    cleanup_old_sessions()
    session = get_or_create_session(thread_id)
    cleanup_old_messages(session)

    if session.graph is None:
        session.graph = graph_builder.compile(checkpointer=session.memory)

    result = session.graph.invoke(
        {"messages": [HumanMessage(content=full_content)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    response = result["messages"][-1].content.strip() if result["messages"] else "对不起，我现在无法回答。"
    await chat_handler.finish(Message(response))

# 添加模型切换命令处理器
change_model = on_command("chat model", priority=1, block=True)

@change_model.handle()
async def handle_change_model(args: Message = CommandArg()):
    global llm, llm_with_tools
    model_name = args.extract_plain_text().strip()
    if not model_name:
        current_model = getattr(llm, "model_name", llm.model)
        await change_model.finish(f"当前模型: {current_model}")
    
    try:
        if "gemini" in model_name.lower():
            plugin_config.llm.provider = "google"
        else:
            plugin_config.llm.provider = "openai"
        
        plugin_config.llm.model = model_name
        llm = get_llm()
        llm_with_tools = llm.bind_tools(tools)
        await change_model.finish(f"已切换到模型: {model_name}")
    except MatcherException:
        raise
    except Exception as e:
        await change_model.finish(f"切换模型失败: {str(e)}")