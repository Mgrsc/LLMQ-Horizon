<div align="center">

<img src="static/LLMQ.webp" width="400" style="margin-bottom: 10px;">

# 🤖 LLMQ-Horizon QQ 聊天机器人

**基于 NoneBot2 和 LangGraph 的智能 QQ 机器人，支持多模型对话、工具调用和会话管理**

<br>

**工具均采用 Function-calling 编写, 不使用插件，参考 [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) , [LangChain Tools](https://python.langchain.com/docs/how_to/#tools)**

<br>

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FMgrsc%2FLLMQ-Horizon.svg?type=small)](https://app.fossa.com/projects/git%2Bgithub.com%2FMgrsc%2FLLMQ-Horizon?ref=badge_small)
[![Docker Release](https://img.shields.io/docker/pulls/bitfennec/llmq-horizon?color=%230077c8&label=Docker%20Pulls&logo=docker&logoColor=white&style=flat)](https://hub.docker.com/r/bitfennec/llmq-horizon)
[![License](https://img.shields.io/github/license/Mgrsc/LLMQ-Horizon?color=%2300c853&label=MIT%20License&style=flat)](https://github.com/Mgrsc/LLMQ-Horizon/blob/main/LICENSE)

<br>

[English](https://github.com/Mgrsc/LLMQ-Horizon/blob/main/readmes_i18n/README_en.md) | [Deutsch](https://github.com/Mgrsc/LLMQ-Horizon/blob/main/readmes_i18n/README_de.md) | [Español](https://github.com/Mgrsc/LLMQ-Horizon/blob/main/readmes_i18n/README_es.md) | [Français](https://github.com/Mgrsc/LLMQ-Horizon/blob/main/readmes_i18n/README_fr.md) | [日本語](https://github.com/Mgrsc/LLMQ-Horizon/blob/main/readmes_i18n/README_ja.md)

</div>

---

## ✨ 主要特性

-   **🔌 丰富的工具集成：** 代码运行、天气查询、占卜、绘画等
-   **🤖 支持多种大模型：** OpenAI、Google Gemini、Groq 等
-   **💬 完善的对话管理：** 群聊/私聊、多轮对话、会话隔离
-   **🎯 灵活的触发方式：** @、关键词、命令前缀
-   **🎨 多媒体能力：** 图片分析、音视频处理
-   **⚡ 自动的会话管理：** 超时清理、并发控制
-   **🦖 强大的扩展能力：** 可自行编写 tools，可使用 tools 控制 nonebot

---

## 🚀 快速开始

### 1. 部署环境准备

-   Docker 和 Docker Compose
-   稳定的网络环境
-   建议系统：Ubuntu 22.04 及以上, Debian 11 以上

> 注意: deepseek模型开启工具不要超过5个，并且提示词尽可能少，否则ds会疯狂调用工具给你刷爆，要不就不调用工具玩个寂寞。

### 2. 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/Mgrsc/LLMQ-Horizon.git
cd LLMQ-Horizon

# 2. 准备配置文件
cp config-tools.toml.example config-tools.toml
cp config.toml.example config.toml
cd napcat/config/
mv onebot11_qq.json onebot11_<你的QQ>.json  # 替换为实际 QQ 号

# 3. 修改配置(参考配置文件中的注释进行修改)
vim config.toml
vim config-tools.toml

# 4. 启动服务
docker compose up -d

# 5. 扫码登录
docker compose logs -f

# 重启 LLMQ 服务
docker compose restart llmq

# 停止所有服务
docker compose down
```

## 🛠️ 工具配置

<details>
<summary>💻 代码运行 (Code Runner - Judge0)</summary>

[Judge0 官方部署教程](https://github.com/judge0/judge0/blob/master/CHANGELOG.md)

1. **准备 Ubuntu 22.04 以上环境及 Docker，配置 cgroup v1：**

    ```bash
    sudo sed -i 's/GRUB_CMDLINE_LINUX=""/GRUB_CMDLINE_LINUX="systemd.unified_cgroup_hierarchy=0"/' /etc/default/grub
    sudo update-grub
    sudo reboot
    ```

2. **部署 Judge0：**

    ```bash
    wget https://github.com/judge0/judge0/releases/download/v1.13.1/judge0-v1.13.1.zip
    unzip judge0-v1.13.1.zip
    cd judge0-v1.13.1

    # 生成两个密码并设置密码
    openssl rand -hex 32

    # 使用生成的密码更新 judge0.conf 文件中的 REDIS_PASSWORD 和 POSTGRES_PASSWORD 变量。

    # 启动服务
    docker-compose up -d db redis
    sleep 10s
    docker-compose up -d
    sleep 5s
    ```

    您的 Judge0 CE v1.13.1 实例现已启动并运行；访问 http://<您的服务器 IP 地址>:2358/docs 获取文档。

3. **配置 config-tools.toml：**

    ```toml
    [code_generation_running]
    judge0_url = "http://your-server:2358"
    judge0_api_key = "your-api-key"
    ```

</details>

<details>
<summary>📝 备忘录 (memos_manage - Memos)</summary>

[Memos 官方部署教程](https://www.usememos.com/docs/install/container-install)

1. **准备环境：**
   - Ubuntu 22.04 及以上
   - Docker 和 Docker Compose

2. **编写 docker-compose.yaml 文件**

    ```yaml
    services:
      memos:
        image: neosmemo/memos:stable
        container_name: memos
        ports:
          - 5230:5230
        volumes:
          - ./memos:/var/opt/memos
        restart: always
    ```

3. **启动服务：**
```bash
docker compose up -d
```

此时就可以在 http://<您的服务器 IP 地址>:5230 访问到 memos，在 memos 中的 Settings 中获取 Tokens。

4. **配置 config-tools.toml：**

```toml
[memos_manage]
url = "http://your-server:5230"
memos_token = "your-memos-token"  # 从 Settings 页面获取的 Token
default_visibility = "PRIVATE"
page_size = 10
user_id = 6
```
</details>

<details>
<summary>📰 新闻获取 (get_news - SynapseNews)</summary>

[SynapseNews 项目地址](https://github.com/Mgrsc/SynapseNews)

1. **部署步骤：**
```bash
git clone https://github.com/Mgrsc/SynapseNews.git
cd synapsenews
# 配置 config.toml
docker compose up -d
```
</details>

## 📝 命令说明

| 命令                      | 说明                             |
| :------------------------ | :------------------------------- |
| `/chat model <模型名>`   | 切换对话模型                     |
| `/chat clear`             | 清理所有会话                     |
| `/chat group <true/false>` | 开关群聊隔离                     |
| `/chat down`              | 关闭对话功能                     |
| `/chat up`                | 开启对话功能                     |
| `/chat chunk <true/false>` | 开关分段发送                     |

## 🦊 提示词编写技巧

<details>
<summary>1. 基本原则</summary>

- 明确指令: 使用命令式语言明确陈述用户的需求,确保LLM能精确理解。
- 提供参考例子/文本：提供详尽例子和信息,构成Few-shot-Prompt,帮助LLM加强意图的理解。
- 结构化表达：使用标记符号(如XML标签、三引号、Markdown)增强可读性,让提示词表达清晰。
- 输出控制: 指定输出格式、语言风格等要求,确保LLM生成符合用户期望的输出。
- 布局优化: 精心安排Prompt的排版布局,便于LLM理解。
</details>

<details>
<summary>2. 其他技巧</summary>

- 列出可用工具，对于复杂工具进行解释和要求
  ```
  create_speech生成语音
    - 最大40字,不可加emoji
    - 支持语言：中文、英语、日语、德语、法语、西班牙语、韩语、阿拉伯语、俄语、荷兰语、意大利语、波兰语、葡萄牙语
    - 可用音色映射:
        可莉 = keli
        西格雯 = xigewen
        神子 = shenzi
        丁真 = dingzhen
        雷军 = leijun
        懒羊羊 = lanyangyang
  ```
- 要求发送tool返回的file://地址
  ```
    画图、获取音乐和tts必须把返回的链接或者文件路径地址发给用户
  ```
- tool返回内容的排版示例
  ```
      # 工具返回内容排版优化示例
    get_weather_data返回数据格式化示例：
    *   A: 告诉我今天长沙的天气
        T: 调用工具`get_weather_data`获取天气
        Q:
        🌤️ {地点}天气
        🌅 日出日落: {xx:xx}-{xx:xx不加年}
        ⏱️   时间: {时间}
        🌡️ 温度: {温度}℃
        💧 湿度: {湿度}%
        🧣 体感温度: {体感温度}℃
        🍃 风向风速度: {风向}-{风速}
        📋 综合状况: {综合分析}
        宝宝出门要多穿衣服哦~小心感冒
  ```
</details>

## 🤝 贡献指南

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

## 🤖 常见问题
所有tools均通过测试，如有问题请参考下面检查。

<details>
<summary>1. 登录失败</summary>

-   检查 QQ 号配置是否正确
-   确认 napcat 配置文件格式
-   查看 napcat 容器日志排查问题

</details>

<details>
<summary>2. 工具调用失败</summary>

-   确认模型支持函数调用能力
-   检查相关 API 密钥配置
-   查看 LLMQ 容器日志定位错误
-   在 docker 容器中加入 [LangSmith](https://smith.langchain.com/) 进行 debug

    ```yaml
    environment:
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
      - LANGCHAIN_API_KEY="<your_api_key>"
      - LANGCHAIN_PROJECT="<your_project_name>"
    ```

</details>

<details>
<summary>3. 其他问题</summary>

-   其他问题请加 QQ 群讨论
    ![qrcode](static/qrcode.jpg)

</details>

## 🔗 相关项目

-   [NoneBot2](https://github.com/nonebot/nonebot2)
-   [LangGraph](https://github.com/langchain-ai/langgraph)
-   [LangChain](https://github.com/langchain-ai/langchain)
-   [Judge0](https://github.com/judge0/judge0)
-   [Memos](https://github.com/usememos/memos)
-   [NapCat](https://github.com/NapNeko/NapCatQQ)

## 📄 许可证

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FMgrsc%2FLLMQ-Horizon.svg?type=large&issueType=license)](https://app.fossa.com/projects/git%2Bgithub.com%2FMgrsc%2FLLMQ-Horizon?ref=badge_large&issueType=license)
本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。
Copyright © 2024 Bitfennec.
---