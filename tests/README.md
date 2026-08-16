# Travel Agent 测试说明

这个文档主要说明测试的运行方式、测试的内容详述。

## 快速开始

单元测试：

```bash
uv run pytest -m unit
```

集成测试：

```bash
uv run pytest -m integration
```

全量测试：

```bash
uv run pytest -m full
```

加上 --cov-report=html 选项，可生成 HTML 格式的报告。

```bash
uv run pytest -m full --cov-report=html
```

## 三种测试命令

### 单元测试

单元测试主要测单个函数或单个类，速度较快。

当前覆盖内容包括：

- `SessionStore` 的 JSONL 保存、加载、删除、索引、空会话和 legacy 路径。
- `SkillsLoader` 的 Skill 扫描、metadata、requirements、always skill 和上下文拼接。
- 本地工具参数校验、文件读取、Shell 执行、澄清工具。
- `TravelAgent` 的工具调用解析、checkpoint 解析、节点选择和取消状态检查。
- `SubAgentRunner` 的描述生成、工具调用解析、递归 task 防护和未知子代理。
- `IndustrialRuntime` 的纯函数，例如 payload 脱敏和 JSON 截断。

### 集成测试

集成测试会把多个模块串起来测，确认核心业务流程能跑通。

当前覆盖内容包括：

- FastAPI 健康检查、会话创建/列表/消息/run/取消接口。
- 用户首次访问、老用户恢复、创建新会话、切换会话、刷新恢复进度。
- 多用户会话、消息、run 隔离。
- Agent 直接回答、`llm -> tool -> llm -> final`、工具失败恢复总结、澄清中断。
- 任务恢复：从 checkpoint 继续，以及工具执行中崩溃且无完成 checkpoint 时允许重试。
- 运行中任务取消：取消后 run 状态变为 `cancelled`，不继续发送最终完成事件。
- 本地 MCP/SSE 协议、MCPClient 调用和缺失工具错误。
- Skill、SubAgent、本地工具/MCP 协作。
- token 超限后自动归档；不超限不归档；新建会话不会触发强制归档剩余历史对话。
- Gradio 进度信号、归档信号、会话切换和刷新恢复辅助逻辑。
- 真实 MySQL、Redis、RabbitMQ：服务可连接时自动运行，不可连接时自动 skip。

基础业务集成测试不会调用真实 LLM，也不会连接真实外部 MCP 服务。LLM 使用 fake client，外部 MCP 使用测试替身，本地 MCP 会覆盖真实 SSE 协议链路。

### 全量测试

`full` 是完整测试集合，包含全部单元测试和集成测试。

如果当前已连接 MySQL、Redis、RabbitMQ，`integration` 和 `full` 都会测到这些中间件的相关代码：

- MySQL：建表、session、run、steps、checkpoint、running run 恢复读取和完成状态更新。
- Redis：取消标记、TTL、会话锁、记忆锁。
- RabbitMQ：记忆归档任务入队、消费、ack、后台 worker 写入 `memory_consolidate` step。

建议使用测试数据库、测试 Redis DB 和测试队列，避免写入开发或生产数据，配置详见 `.env.test.example`。

## 目录说明

```text
tests/
├── unit/          # 单元测试：单个函数或类，依赖 fake/mock
├── integration/   # 集成测试：业务集成 + 真实中间件自动探测
├── conftest.py    # 公共 fixture、fake LLM、fake MCP、fake runtime
└── README.md      # 本说明文档
```

## 覆盖率

项目已启用 `pytest-cov`，运行三条常用命令时会默认在终端显示覆盖率摘要。

如果需要更详细的 HTML 报告，可以临时追加：

```bash
uv run pytest -m full --cov-report=html
```

生成结果在 `htmlcov/` 目录下。
