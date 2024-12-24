from typing import List, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import tomli

class LLMConfig(BaseModel):
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


class PluginConfig(BaseModel):
    Trigger_words: List[str] = []
    Trigger_mode: List[str] = ["keyword", "at"]
    group_chat_isolation: bool = True
    enable_private: bool = True
    disabled_message: str = "Bot已禁用"
    enable_group: bool = True
    max_sessions: int = Field(default=1000, gt=0)
    enable_username: bool = False
    empty_message_replies: List[str] = ["你好", "在呢", "我在听"]

class Config(BaseModel):
    llm: LLMConfig
    plugin: PluginConfig

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
                Trigger_words=toml_config["plugin"]["llm_chat"]["Trigger_words"],
                Trigger_mode=toml_config["plugin"]["llm_chat"]["Trigger_mode"],
                group_chat_isolation=toml_config["plugin"]["llm_chat"]["group_chat_isolation"],
                enable_private=toml_config["plugin"]["llm_chat"]["enable_private"],
                disabled_message=toml_config["plugin"]["llm_chat"].get("disabled_message", "Bot已禁用"),
                enable_group=toml_config["plugin"]["llm_chat"]["enable_group"],
                max_sessions=toml_config["plugin"]["llm_chat"].get("max_sessions", 1000),
                enable_username=toml_config["plugin"]["llm_chat"].get("enable_username", False),
                empty_message_replies=toml_config["plugin"]["llm_chat"].get("empty_message_replies", ["你好", "在呢", "我在听"]),
            )
            
            return cls(llm=llm_config, plugin=plugin_config)
        except Exception as e:
            raise RuntimeError(f"Failed to load config.toml: {str(e)}")


