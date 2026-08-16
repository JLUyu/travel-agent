---
title: Travel Agent
emoji: 🧭
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# travel-agent

## 项目简介

Travel Agent 是一个带 Web 前端的多工具 Agent 项目。用户可在 Gradio 网页发起旅行、路线、天气、火车票、搜索、文件读取、Shell 执行、Skill 调用、SubAgent 调研等任务，后端 FastAPI 会按用户和会话维护独立 Agent 实例，并通过 LangGraph 驱动 LLM 推理和工具调用。

## ⚡️quick start

### 1.进入目录
```bash
cd travel-agent
```

### 2.安装依赖
```bash
sudo apt  install curl
curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
sudo apt install nodejs bubblewrap -y
npm install @anthropic-ai/sandbox-runtime

uv sync
```

### 3.启动主程序
```bash
uv run python main.py
```

## 公网部署

项目已提供 `Dockerfile`，可直接部署到 Hugging Face Docker Spaces、Render、Railway 等支持容器的平台。公网环境会默认关闭 Shell 工具、限制文件读取范围，并对单个访问来源做基础限流。

部署前必须在平台的 Secret/Environment Variables 中配置：

```env
DEEPSEEK_API_KEY=你的新密钥
EMBEDDING_API_KEY=你的新密钥
```

外部 MCP 工具按需配置，留空时对应工具不会加载：

```env
AMAP_MCP_URL=
RAILWAY_MCP_URL=
WEB_SEARCH_MCP_URL=
```

不要把真实密钥提交到仓库。若密钥曾出现在源码或 Git 历史中，应先在对应服务商后台吊销并重新生成。

本地验证容器：

```bash
docker build -t travel-agent .
docker run --rm -p 7860:7860 --env-file .env travel-agent
```

## 启动MySQL、Redis、RabbitMQ（可选）

如果需要启用 MySQL、Redis、RabbitMQ，请按下面顺序执行。以下命令适用于 WSL/Linux 环境。

### 1. 安装系统服务

```bash
sudo apt update
sudo apt install -y mysql-server redis-server rabbitmq-server
```

### 2. 启动 MySQL、Redis、RabbitMQ

如果系统支持 `systemctl`：

```bash
sudo systemctl enable mysql redis-server rabbitmq-server
sudo systemctl start mysql redis-server rabbitmq-server
```

如果 WSL 中 `systemctl` 不可用，使用：

```bash
sudo service mysql start
sudo service redis-server start
sudo service rabbitmq-server start
```

### 3. 初始化 MySQL 数据库和用户

```bash
sudo mysql
```

进入 MySQL 后执行：

```sql
CREATE DATABASE IF NOT EXISTS travel_agent CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'travel_agent'@'localhost' IDENTIFIED BY 'travel_agent';
GRANT ALL PRIVILEGES ON travel_agent.* TO 'travel_agent'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```
表示创建数据库travel_agent和用户travel_agent，密码为travel_agent，授予该用户对该数据库的所有权限，数据库只能从 localhost 连接

### 4. 检查服务状态

```bash
mysql -u travel_agent -ptravel_agent -e "SHOW DATABASES;"
redis-cli ping
sudo rabbitmqctl status
```
- `mysql -u travel_agent -ptravel_agent -e "SHOW DATABASES;"` 正常应该返回：

    +--------------------+

    | Database           |

    +--------------------+

    | information_schema |

    | performance_schema |

    | travel_agent       |

    +--------------------+

- `redis-cli ping` 正常应返回：
PONG

- `sudo rabbitmqctl status` 正常返回的内容应包含
Status of node rabbit@xxx
Runtime

### 5. 配置 `.env`

```env
MYSQL_ENABLED=true
REDIS_ENABLED=true
RABBITMQ_ENABLED=true
```

### 6. 安装 Python 依赖

```bash
uv sync
```

### 7. 启动主程序

```bash
uv run python main.py
```

### 8. 启动 RabbitMQ 后台 Worker

如果启用了 RabbitMQ 后台记忆归档，需要另开一个终端运行：

```bash
uv run python industrial_worker.py
```

### 9. WSL 重启后的服务恢复

如果重启 WSL 后服务没有自动启动，先执行：

```bash
sudo service mysql start
sudo service redis-server start
sudo service rabbitmq-server start
```

然后再启动主程序：

```bash
uv run python main.py
```

### 10.提问示例
```
天安门到颐和园怎么走
2026-04-15北京到上海火车票有余票吗
搜索关于codex的新闻,总结返回的第一篇文章
计算123+456
总结对话历史
使用skill获取上海天气
调用skill制定南京一人一日游攻略，预算500
```

### 11.退出
按ctrl+c

## 高并发运行

压测或评测时可以只启动后端，并按需开启多进程 worker：

```bash
BACKEND_WORKERS=4 AGENT_LLM_CONCURRENCY=10 AGENT_MCP_TOOL_CONCURRENCY=10 uv run python backend.py
```

建议高并发模式同时启用 MySQL 和 Redis：MySQL 保存 run、steps、checkpoint，Redis 保存取消标记和跨进程会话锁。常用并发参数：

```env
BACKEND_WORKERS=4
AGENT_LLM_CONCURRENCY=20
AGENT_MCP_TOOL_CONCURRENCY=20
AGENT_SYNC_CONTENT_DELAY=0
AGENT_STREAM_CONTENT_DELAY=0.02
AGENT_EVAL_SESSION_POOL_SIZE=10
AGENT_EVAL_PREWARM=true
```

- `BACKEND_WORKERS`：后端进程数；大于 1 时每个 worker 都有独立内存中的 Agent 缓存。
- `AGENT_LLM_CONCURRENCY`：单进程内主 Agent 和 SubAgent 共享的 LLM 并发上限，`0` 表示不限制。
- `AGENT_MCP_TOOL_CONCURRENCY`：单进程内 MCP 工具调用并发上限，`0` 表示不限制。
- `AGENT_SYNC_CONTENT_DELAY`：`/chat` 非流式接口的最终答案输出延迟，评测默认 `0`。
- `AGENT_STREAM_CONTENT_DELAY`：`/chat/stream` 的逐字输出延迟，前端默认保留 `0.02` 秒。
- `AGENT_EVAL_SESSION_POOL_SIZE`：promptfoo provider 使用的固定评测会话池大小，建议不小于 promptfoo 并发数。
- `AGENT_EVAL_PREWARM`：是否在第一批评测请求前调用 `/session/init` 预热会话池。

同一个 `user_id/session_id` 不建议并发跑多轮任务；高并发评测应让每条请求使用独立会话，或确保测试会话池大小不小于并发数。

## 沙箱

项目集成了 Anthropic `sandbox-runtime` 沙箱方案，用于文件权限和网络权限隔离。

- 沙箱配置文件：`.srt-settings.json`
- 主要配置需要隔离的网址和文件。
- Shell 命令执行将在沙箱运行，降低安全风险。

## 系统架构

### 前后端

- `main.py`：项目启动入口，负责启动 FastAPI 后端和 Gradio 前端。
- `backend.py`：FastAPI 后端服务，负责会话接口、聊天流式接口、任务停止、任务恢复等。
- `gradio_manager.py`：Gradio 前端界面，负责页面初始化、会话列表、聊天框、进度展示、归档情况刷新、停止按钮等交互。
- `session_manager.py`：管理不同用户、不同会话对应的 Agent 实例。

### Agent Loop

Agent 使用 LangGraph 构建 ReAct 风格工作流：

```text
START -> llm -> tool -> llm -> ... -> END
```

- `llm` 节点负责思考下一步，决定是否需要调用工具。
- `tool` 节点负责执行本地工具、MCP 工具或 SubAgent。
- `tool_guard.py` 会在工具真正执行前处理参数 schema 校验、重复调用拦截、失败计数和 MCP 重试分类。
- 工具执行结果会作为新的上下文交回 LLM。
- 当 LLM 判断信息足够时，生成最终中文回答。

举例来说，用户问“天安门到颐和园怎么走”，工作流大概是：

```text
用户问题 -> LLM 判断需要路线工具 -> 调用高德 MCP -> 得到路线结果 -> LLM 整理中文答案
```

### 工具体系

- 外部 MCP 工具：高德路线规划、12306 火车票查询、网络搜索等。
- 本地 MCP 工具：总结对话历史、简单计算等。
- 本地基础工具：文件读取、Shell 执行、澄清提问等。
- Skill：天气查询、旅行规划、Skill 创建等。
- SubAgent：复杂任务可以委托给子 Agent，例如旅行调研、景点资料收集等。
- 工具调用治理：
重复工具调用处理
参数缺失检测
失败工具重试

### 会话与记忆

- 短期会话历史保存在 `memory/{user_id}/sessions/{session_id}/session.jsonl`。
- 会话列表索引保存在 `memory/{user_id}/sessions.json`。
- 长期记忆保存在用户级 `MEMORY.md` 和 `HISTORY.md`。
- 当对话变长时，项目会通过记忆整合逻辑压缩旧对话，降低上下文压力。
- 新建会话不触发强制归档；旧会话历史仍保存在原会话 JSONL 中，长期记忆只按 token 超限规则归档。
- 归档触发后会按预算规则选择一段已完成对话写入 `HISTORY.md` / `MEMORY.md`，再更新 `last_consolidated` 指针。
- 如果 `RABBITMQ_ENABLED=true`，自动记忆整合会进入 RabbitMQ 后台队列；后台 Worker 整合完成后，Gradio 更新“本会话归档情况”。
- 如果启用了 MySQL，用户、会话、任务 run、任务步骤和 run 级 checkpoint 也会同步写入 MySQL。

### 工业化运行能力

- MySQL：保存用户、会话、任务 run、任务步骤、run 级 checkpoint。
- Redis：保存取消标记、会话锁和用户级记忆锁。
- RabbitMQ：用于后台记忆归档任务队列；后台归档完成后会通知 Gradio 刷新当前会话归档情况。

## 目录结构说明

```text
├── config.py                # 环境配置，包括模型、Embedding、MCP、MySQL、Redis、RabbitMQ 等配置
├── main.py                  # 主程序入口，启动 FastAPI 后端和 Gradio 前端
├── backend.py               # FastAPI 后端服务，处理前端请求并调用对应会话的 Agent
├── gradio_manager.py        # Gradio 界面管理，提供 Web 交互界面
├── travel_agent.py          # Agent 核心逻辑，定义 LLM 推理、工具调用、任务恢复、记忆管理等
├── session_manager.py       # 会话管理器，管理多个用户的独立 Agent 实例
├── session_store.py         # 会话数据存储，管理 JSONL 会话文件、索引和长期记忆文件
├── industrial_runtime.py    # MySQL、Redis、RabbitMQ 适配层，记录 run、steps、checkpoint 和后台任务
├── memory_manager.py        # 向量历史存储管理，处理对话历史的向量化和检索
├── memory_consolidator.py   # 记忆整合器，将对话历史压缩整合到长期记忆文件
├── skills_loader.py         # 加载和管理 Skill 能力
├── mcp_client.py            # MCP 客户端，建立 SSE 连接并获取工具描述
├── mcp_server.py            # 本地 MCP 服务端，定义本地 MCP 工具
├── subagents.py             # 子 Agent 系统，支持复杂任务委派
├── tools/                   # 本地工具模块目录
├── skills/                  # Skill 目录
├── tests/                   # pytest 测试，分为 unit、integration、full
├── observability.py         # Langfuse 初始化、请求级追踪等逻辑
├── .srt-settings.json       # 沙箱配置文件
├── pyproject.toml           # 项目依赖配置
└── uv.lock                  # uv 锁文件
```

## SubAgent

当任务复杂、步骤较多时，主 Agent 会调用 SubAgent 来完成任务。SubAgent 拥有独立上下文，可以减少主 Agent 上下文中的冗余信息。

举例来说，用户要求“调研上海适合亲子游的景点并按预算整理路线”，主 Agent 可以把景点资料收集委托给旅行研究类 SubAgent，再把结果汇总给用户。

## 测试
单元测试和集成测试，基于 pytest 框架。

```bash
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m full
```

加上 --cov-report=html 选项，可生成 HTML 格式的报告。

```bash
uv run pytest -m full --cov-report=html
```

- `unit`：单元测试，覆盖会话存储、Skill 加载、本地工具、Agent 解析/checkpoint、SubAgent、运行时纯函数等。
- `integration`：集成测试，覆盖 FastAPI、用户访问/会话创建/切换/刷新恢复、停止任务、多用户隔离、Agent 流式流程、本地 MCP/SSE、Skill、SubAgent、token 超限自动归档、Gradio 进度/归档刷新等；如果当前已连接 MySQL、Redis、RabbitMQ，会测到这些中间件的相关代码。
- `full`：完整测试集合，包含全部 `unit` 和 `integration` 测试。

## Promptfoo 评测

跑评测，设置并发数为 10，重复 2 次。默认 provider 会调用 `/chat/eval` 轻量接口
```bash
cd eval
promptfoo eval --max-concurrency 10 --no-table --repeat=2
```

`eval/promptfooconfig.yaml` 默认 `evaluateOptions.maxConcurrency` 为 10，`sessionPoolSize` 也为 10

查看报告
```bash
promptfoo view
```

## Langfuse 监控配置

项目已接入 Langfuse，用于监控模型输入、输出、token 消耗量、响应时间。在 `.env` 或 `config.py` 填写：

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

未配置 `LANGFUSE_PUBLIC_KEY` 或 `LANGFUSE_SECRET_KEY` 时，监控会自动关闭，不影响本地运行。配置完成后，主 Agent、子 Agent、记忆整合、本地 MCP 摘要和 embedding 调用会按现有 `session_id` 聚合到 Langfuse 的 session 视图中。

## ask_clarification 中断确认

用户问题缺失关键信息时，Agent 会先确认用户需求。本轮对话会暂停，输出确认信息，等待用户回复。

举例来说，如果用户只说“帮我订票”，但没有出发地、目的地或日期，Agent 会先询问这些关键信息，而不是直接调用火车票工具。
