FROM ghcr.io/astral-sh/uv:bookworm-slim

# 安装 tini 作为初始化系统
RUN apt-get update \
    && apt-get install -y tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*

LABEL maintainer="mgrsc <mail@occult.ac.cn>"

WORKDIR /app

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 设置 uv 环境变量
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"

# 复制项目文件
COPY . /app

# 赋予 entrypoint.sh 执行权限
RUN chmod +x /app/entrypoint.sh

# 使用 uv 安装依赖
RUN uv sync --frozen --no-dev

# 设置环境变量
ENV HOST=0.0.0.0
ENV PORT=8080
ENV COMMAND_START='["/"]'
ENV COMMAND_SEP='["."]'

# 暴露端口
EXPOSE 8080 5000

# 启动命令
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/entrypoint.sh"]