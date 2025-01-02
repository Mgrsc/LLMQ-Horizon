from typing import Annotated, List, Union, Any, Optional, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, trim_messages, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.language_models import LanguageModelInput
from langchain_google_genai import ChatGoogleGenerativeAI
from .tools import load_tools
from functools import wraps
from .config import Config
from .config import plugin_config
import asyncio
import json

groq_models = {
    "llama3-groq-70b-8192-tool-use-preview",
    "llama-3.3-70b-versatile"
}

class MyOpenAI(ChatOpenAI):
    @property
    def _default_params(self) -> Dict[str, Any]:
        params = super()._default_params
        if "max_completion_tokens" in params:
            params["max_tokens"] = params.pop("max_completion_tokens")
        return params

    def _get_request_payload(self, input_: LanguageModelInput, *, stop: Optional[List[str]] = None, **kwargs: Any) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload

async def get_llm(model=None):
    """异步获取适当的 LLM 实例"""
    model = model.lower() if model else plugin_config.llm.model
    print(f"使用模型: {model}")

    try:
        if model in groq_models:
            print("使用groq")
            return ChatGroq(
                model=model,
                temperature=plugin_config.llm.temperature,
                max_tokens=plugin_config.llm.max_tokens,
                api_key=plugin_config.llm.groq_api_key,
            )
        elif "gemini" in model:
            print("使用google")
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=plugin_config.llm.temperature,
                max_tokens=plugin_config.llm.max_tokens,
                google_api_key=plugin_config.llm.google_api_key,
                top_p=plugin_config.llm.top_p,
            )
        else:
            print("使用 OpenAI")
            return MyOpenAI(
                model=model,
                temperature=plugin_config.llm.temperature,
                max_tokens=plugin_config.llm.max_tokens,
                api_key=plugin_config.llm.api_key,
                base_url=plugin_config.llm.base_url,
                top_p=plugin_config.llm.top_p,
            )
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        raise

class State(TypedDict):
    messages: Annotated[list, add_messages]

async def build_graph(config: Config, llm):
    """构建并返回对话图"""
    tools = load_tools()
    llm_with_tools = llm.bind_tools(tools)
    
    trimmer = trim_messages(
        strategy="last",
        max_tokens=config.llm.max_context_messages,
        token_counter=len,
        include_system=True,
        allow_partial=False,
        start_on="human",
        end_on=("human", "tool"),
    )

    async def chatbot(state: State):
        messages = state["messages"]
        if config.llm.system_prompt:
            messages = [SystemMessage(content=config.llm.system_prompt)] + messages
        trimmed_messages = trimmer.invoke(messages)
        if not trimmed_messages:
            return {"messages": []}
        print("-" * 50)
        truncated_messages = trimmed_messages[-2:]
        print(format_messages_for_print(truncated_messages))
        response = await llm_with_tools.ainvoke(trimmed_messages) 
        print(f"chatbot: {response}")
        return {"messages": [response]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    return graph_builder



def format_messages_for_print(messages: List[Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]]) -> str:
    """格式化 LangChain 消息列表"""
    output = []
    for message in messages:
        # if isinstance(message, SystemMessage):
        #     output.append(f"SystemMessage: {message.content}\n")
        #     output.append("_" * 50 + "\n")
        if isinstance(message, HumanMessage):
            output.append("\n" + "_" * 50 + "\nHumanMessage: \n" +  f"{message.content}\n\n")
        elif isinstance(message, AIMessage):
            output.append(f"AIMessage: {message.content}\n")
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    output.append(f"  Tool Name: {tool_call['name']}\n")
                    try:
                        args = json.loads(tool_call['args'])
                    except (json.JSONDecodeError, TypeError):
                        args = tool_call['args']
                    output.append(f"  Tool Arguments: {args}\n\n")
        elif isinstance(message, ToolMessage):
            output.append(f"ToolMessage: \n  Tool Name: {message.name}\n  Tool content: {message.content}\n\n")
    return "".join(output)