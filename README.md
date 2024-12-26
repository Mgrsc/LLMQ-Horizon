# 🤖 LLMQ-Horizon QQ_Chatbot

一个基于 NoneBot2 和 LangGraph 的QQ_Chatbot。

## ✨ 特性

- 🔌 支持多种工具扩展
- 💬 支持群聊和私聊
- 🎯 多种触发方式:
  - @机器人
  - 关键词触发
  - 命令前缀触发
- 🧠 基于 LangGraph 的对话管理
- 📦 自动会话管理和清理

# 快速开始

## 部署
```
git clone https://github.com/Mgrsc/LLMQ-Horizon.git

# 把示例文件改名
mv config-tools.toml.example config-tools.toml
mv config.toml.example config.toml

# 参考下面编辑配置文件

# 启动
docker compose up -d
# 修改配置仅需重启llmq
docker compose down llmq
# 关闭
docker compose down
```

### 编辑napcat
```
cd napcat/config/
mv onebot11_qq.json onebot11_<你的QQ>.json #改为你的qq号
```

### 编辑compose

```
#填入超级用户QQ，用于命令执行
    environment:
      - ACCOUNT=<qq>
```

### 编辑config.toml

```
[llm]
model = "gpt-4o" # 模型必须支持fc否则无法使用tools
superusers = "1221212" # 超级用户QQ
groq_api_key = "xxxxxxxxxx"
google_api_key = "xxxxxxxxxxx"
api_key = "xxxxxxxx"
base_url = "https://xxx.xxx.com/v1"
temperature = 0.4 # 注意范围是0-1不要调太大
command_start = "?" # 触发命令前缀
system_prompt= """ """ #编写提示词，工具调用有问题请调节

[plugin.llm_chat]
# 触发命令
Trigger_words = ["小宝","qw",]
# 触发方式"prefix", "keyword", "at"
Trigger_mode = ["prefix","at",]
# 是否开启群对话隔离，群里每个人对话都是隔离开的
group_chat_isolation = false
# 是否传递用户名给LLM格式为 "用户名：消息"
enable_username = true
# 是否允许私聊
enable_private = true
# 是否允许群聊
enable_group = true
max_sessions = 1000
# 默认回复列表，空艾特，空触发回复
empty_message_replies = [
    "说",
    "？",
    "内容？",
    "问题？"
    ]
```



## 编辑config-tools.toml
```
- img_analysis：视觉能力，填写视觉模型，仅支持openai请求，可以使用new-api等项目转换
- code_runner：代码运行，需要安装judge0填写url和key
- divination：占卜，填写openai格式的api和url可以和主模型一致
- draw：绘画能力暂时仅支持fal(https://fal.ai/)的模型,还需填写一个openai格式模型用于提示词生成
- get_weather_data：天气信息，对接oenweather(https://openweathermap.org/api/one-call-3)
- jina_fact_checking/jina_reader/jina_search: jina的模型https://jina.ai/
- picture_api：随机图片
```


## tools

### 安装code_runner的judge0

参考https://github.com/judge0/judge0/blob/master/CHANGELOG.md
我们建议使用 Ubuntu 22.04，在此系统上您需要进行以下 GRUB 更新(改为cgroup v1)：
```
使用 sudo 打开文件 /etc/default/grub
在 GRUB_CMDLINE_LINUX 变量的值中添加 systemd.unified_cgroup_hierarchy=0。
应用更改：sudo update-grub
重启您的服务器：sudo reboot
```
部署步骤:
下载并解压发行版压缩包：
```
wget https://github.com/judge0/judge0/releases/download/v1.13.1/judge0-v1.13.1.zip
unzip judge0-v1.13.1.zip
```
访问[此网站](https://www.random.org/passwords/?num=1&len=32&format=plain&rnd=new)以生成随机密码。
使用生成的密码更新 judge0.conf 文件中的 REDIS_PASSWORD 变量。
再次访问[此网站](https://www.random.org/passwords/?num=1&len=32&format=plain&rnd=new)以生成另一个随机密码。
使用生成的密码更新 judge0.conf 文件中的 POSTGRES_PASSWORD 变量。
运行所有服务并等待几秒钟，直到所有内容都初始化完成：
```
cd judge0-v1.13.1
docker-compose up -d db redis
sleep 10s
docker-compose up -d
sleep 5s
```
您的 Judge0 CE v1.13.1 实例现已启动并运行；访问 http://<您的服务器 IP 地址>:2358/docs 获取文档。