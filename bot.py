import nonebot
from nonebot.adapters.onebot.v11 import Adapter
import toml
from pathlib import Path

config_path = Path(__file__).parent / "config.toml"
config = toml.load(config_path)
nonebot.init(
    superusers={config['llm']['superusers']},
    host="0.0.0.0",
    port=8082,command_start={config['llm']['command_start']},
    command_sep={"."})

driver = nonebot.get_driver()
driver.register_adapter(Adapter)

# nonebot.load_builtin_plugins("echo")
# nonebot.load_plugin("thirdparty_plugin")
nonebot.load_plugins("plugins")
# nonebot.load_from_toml("config.toml", encoding="utf-8")

if __name__ == "__main__":
    nonebot.run()
