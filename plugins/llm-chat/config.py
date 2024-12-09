from typing import List, Optional
from pydantic import BaseModel, Field
import tomli
from pathlib import Path

class LLMConfig(BaseModel):
    model: str
    api_key: Optional[str]
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: Optional[str] = None
    max_context_messages: int = 10
    google_api_key: str = ""
    provider: str = "openai"

class PluginConfig(BaseModel):
    command_start: List[str] = []
    enable_private: bool = True
    enable_group: bool = True
    need_at: bool = True
    max_sessions: int = Field(default=1000, gt=0)
    max_messages_per_session: int = Field(default=50, gt=0)
    keywords: List[str] = []
    empty_message_replies: List[str] = ["你好", "在呢", "我在听"] 

class Config(BaseModel):
    llm: LLMConfig
    plugin: PluginConfig

    @classmethod
    def load_config(cls) -> "Config":
        """从 config.toml 加载配置"""
        root_path = Path(__file__).parent.parent.parent
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
                provider=toml_config["llm"].get("provider", "openai"),
            )
            
            plugin_config = PluginConfig(
                command_start=toml_config["plugin"]["llm_chat"]["command_start"],
                enable_private=toml_config["plugin"]["llm_chat"]["enable_private"],
                enable_group=toml_config["plugin"]["llm_chat"]["enable_group"],
                need_at=toml_config["plugin"]["llm_chat"]["need_at"],
                max_sessions=toml_config["plugin"]["llm_chat"].get("max_sessions", 1000),
                max_messages_per_session=toml_config["plugin"]["llm_chat"].get("max_messages_per_session", 50),
                empty_message_replies=toml_config["plugin"]["llm_chat"].get("empty_message_replies", ["你好", "在呢", "我在听"]),
            )
            
            return cls(
                llm=llm_config,
                plugin=plugin_config
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load config.toml: {str(e)}")