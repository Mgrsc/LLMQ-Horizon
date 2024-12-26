from pathlib import Path
import toml

def load_config() -> dict:
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / "config-tools.toml"
    try:
        return toml.load(config_path)
    except FileNotFoundError:
        print(f"错误: 找不到配置文件 {config_path}")
        return {}
    except toml.TomlDecodeError as e:
        print(f"错误: 配置文件格式不正确 - {e}")
        return {}

config = load_config()