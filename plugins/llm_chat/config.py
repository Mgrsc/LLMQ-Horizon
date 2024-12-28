from typing import List, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import tomli

class LLMConfig(BaseModel):
    """LLM 配置"""
    model: str
    api_key: Optional[str]
    google_api_key: str = ""
    groq_api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 1000
    system_prompt: Optional[str] = None
    max_context_messages: int = 10

class ChunkConfig(BaseModel):
    """分段发送配置"""
    enable: bool = False
    words: List[str] = ["||"]
    max_time: float = 5.0
    char_per_s: int = 5

class PluginConfig(BaseModel):
    """插件配置"""
    trigger_words: List[str] = []
    trigger_mode: List[str] = ["keyword", "at"]
    group_chat_isolation: bool = True
    enable_private: bool = True
    enable_group: bool = True
    max_sessions: int = Field(default=1000, gt=0)
    enable_username: bool = False
    chunk: ChunkConfig = ChunkConfig()
    command_start: str = "?"
    superusers: str = ""

class ResponseConfig(BaseModel):
    """回复消息配置"""
    empty_message_replies: List[str] = ["你好", "在呢", "我在听"]
    token_limit_error: str = "太长了发不出来，换一个吧"
    general_error: str = "卧槽，报错了，尝试自行修复中，聊聊别的吧！"
    disabled_message: str = "Bot已禁用"  # 新增此行

class Config(BaseModel):
    llm: LLMConfig
    plugin: PluginConfig
    responses: ResponseConfig = ResponseConfig()

    @classmethod
    def load_config(cls) -> "Config":
        """从 config.toml 加载配置"""
        root_path = Path(__file__).resolve().parents[2]
        config_path = root_path / "config.toml"
        
        if not config_path.exists():
            raise RuntimeError(f"Config file not found at {config_path}")
            
        try:
            with open(config_path, "rb") as f:
                toml_config = tomli.load(f)

            llm_config = LLMConfig(
                model=toml_config["llm"]["model"],
                api_key=toml_config["llm"]["api_key"],
                base_url=toml_config["llm"]["base_url"],
                temperature=toml_config["llm"].get("temperature", 0.7),
                max_tokens=toml_config["llm"].get("max_tokens", 2000),
                system_prompt=toml_config["llm"]["system_prompt"],
                google_api_key=toml_config["llm"].get("google_api_key", ""),
                top_p=toml_config["llm"].get("top_p", 1.0),
                groq_api_key=toml_config["llm"].get("groq_api_key", ""),
            )
            
            plugin_config = PluginConfig(
                trigger_words=toml_config["plugin_settings"]["trigger_words"],
                trigger_mode=toml_config["plugin_settings"]["trigger_mode"],
                group_chat_isolation=toml_config["plugin_settings"]["group_chat_isolation"],
                enable_private=toml_config["plugin_settings"]["enable_private"],
                enable_group=toml_config["plugin_settings"]["enable_group"],
                max_sessions=toml_config["plugin_settings"].get("max_sessions", 1000),
                enable_username=toml_config["plugin_settings"].get("enable_username", False),
                command_start=toml_config["plugin_settings"].get("command_start", "?"),
                superusers=toml_config["plugin_settings"].get("superusers", ""),
                chunk=ChunkConfig(
                    enable=toml_config.get("chunk", {}).get("enable", False),
                    words=toml_config.get("chunk", {}).get("words", ["||"]),
                    max_time=toml_config.get("chunk", {}).get("max_time", 5.0),
                    char_per_s=toml_config.get("chunk", {}).get("char_per_s", 5)
                )
            )
            
            responses_config = ResponseConfig(
                empty_message_replies=toml_config["responses"].get("empty_message_replies", ResponseConfig().empty_message_replies),
                token_limit_error=toml_config["responses"].get("token_limit_error", ResponseConfig().token_limit_error),
                general_error=toml_config["responses"].get("general_error", ResponseConfig().general_error),
                disabled_message=toml_config["responses"].get("disabled_message", "Bot已禁用")
            )
            
            return cls(llm=llm_config, plugin=plugin_config, responses=responses_config)
        except Exception as e:
            raise RuntimeError(f"Failed to load config.toml: {str(e)}")


