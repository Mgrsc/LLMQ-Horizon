from typing import List, Dict, Optional
from langchain.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
import importlib.util
import sys
import os
from pathlib import Path
import tomli

def get_builtin_tools(config: dict) -> Dict[str, BaseTool]:
    """根据配置返回内置工具的初始化方法字典。"""
    return {
        "tavily_search": lambda: TavilySearchResults(
            max_results=config.get("tavily", {}).get("max_results", 2)
        )
    }

def load_tools(enabled_tools: Optional[List[str]] = None, tool_paths: Optional[List[str]] = None) -> List[BaseTool]:
    """加载启用的工具

    Args:
        enabled_tools: 可选的已启用工具列表,如果为None则从配置文件加载
        tool_paths: 可选的工具路径列表
    """
    tools_list = []

    if enabled_tools is None:
        root_path = Path(__file__).resolve().parents[2]
        config_path = root_path / "config-tools.toml"

        try:
            with open(config_path, "rb") as f:
                config = tomli.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Tools config file not found at {config_path}")

        builtin_tool_factories = get_builtin_tools(config)
        
        tavily_config = config.get("tavily")
        if tavily_config and "api_key" in tavily_config:
            os.environ["TAVILY_API_KEY"] = tavily_config["api_key"]

        builtin_tool_names = config.get("tools", {}).get("builtin", [])
        for name in builtin_tool_names:
            factory = builtin_tool_factories.get(name)
            if factory:
                tools_list.append(factory())
            else:
                print(f"Warning: Built-in tool {name} not found")

        enabled_tools = config.get("tools", {}).get("enabled", [])

    search_paths = [str(Path(__file__).resolve().parents[2])]
    if tool_paths:
        search_paths.extend(tool_paths)

    for path in search_paths:
        if path not in sys.path:
            sys.path.insert(0, path)

    for name in enabled_tools:
        try:
            tool_module = importlib.import_module(f"tools.{name.replace('-', '_')}")
            tools_list.extend(tool_module.tools if hasattr(tool_module, 'tools') else [])
        except (ModuleNotFoundError, ImportError) as e:
            print(f"Error loading tool {name}: {str(e)}")

    return tools_list