// promptfoo 自定义 Provider：对接本地 Travel Agent /chat/eval 接口。
//
// 接口契约：
//   - 请求体仅携带 query 字段
//   - 返回 finalAnswer 作为 promptfoo 的 output（用于 LLM-as-judge 文本比对）
//   - 完整原始响应 JSON 放入 metadata.json，供上层 JavaScript 断言读取 tool_calls
//   - 记录本次调用的端到端延迟 latencyMs，用于全局 P50/P95/P99 派生指标
//   - trajectory 字段同步暴露工具链路，配合 yaml 顶层 tracing:true 支持 this.trajectory

const AGENT_ENDPOINT = process.env.AGENT_ENDPOINT || 'http://127.0.0.1:6008/chat/eval';
const DEFAULT_TIMEOUT_MS = 600000;
const DEFAULT_SESSION_POOL_SIZE = 10;

let sharedPrewarmPromise = null;

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(String(value ?? ''), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

class AgentProvider {
  constructor(options = {}) {
    this.providerId = options.id || 'travel-agent';
    this.config = options.config || {};
    this.endpoint = this.config.endpoint || AGENT_ENDPOINT;
    // Agent 单例可能会串行调用 LLM 和多个外部工具，评测侧超时要长于单次后端任务耗时。
    this.timeoutMs = parsePositiveInt(
      this.config.timeoutMs ?? process.env.AGENT_TIMEOUT_MS,
      DEFAULT_TIMEOUT_MS,
    );
    this.sessionPoolSize = parsePositiveInt(
      this.config.sessionPoolSize ?? process.env.AGENT_EVAL_SESSION_POOL_SIZE,
      DEFAULT_SESSION_POOL_SIZE,
    );
    this.prewarmSessions = String(
      this.config.prewarmSessions ?? process.env.AGENT_EVAL_PREWARM ?? 'true',
    ).toLowerCase() !== 'false';
    this.poolPrefix = String(
      this.config.sessionPoolPrefix ?? process.env.AGENT_EVAL_SESSION_POOL_PREFIX ?? 'promptfoo-eval',
    );
    this.nextPoolIndex = 0;
    this.sessionPool = Array.from({ length: this.sessionPoolSize }, (_, index) => ({
      user_id: `${this.poolPrefix}-user-${index + 1}`,
      session_id: `${this.poolPrefix}-session-${index + 1}`,
    }));
  }

  id() {
    return this.providerId;
  }

  getBaseUrl() {
    const url = new URL(this.endpoint);
    return `${url.protocol}//${url.host}`;
  }

  async ensureSessionPoolReady() {
    if (!this.prewarmSessions || this.sessionPool.length === 0) {
      return;
    }
    if (!sharedPrewarmPromise) {
      const baseUrl = this.getBaseUrl();
      // 预热固定会话池，避免 promptfoo 第一批并发请求全卡在 Agent 冷启动。
      sharedPrewarmPromise = Promise.all(
        this.sessionPool.map(({ user_id, session_id }) =>
          fetch(`${baseUrl}/session/init/${encodeURIComponent(user_id)}/${encodeURIComponent(session_id)}`, {
            method: 'POST',
          }).catch((err) => {
            console.warn(`预热会话失败 ${session_id}: ${err && err.message ? err.message : String(err)}`);
          }),
        ),
      );
    }
    await sharedPrewarmPromise;
  }

  pickSession(context) {
    const vars = (context && context.vars) || {};
    if (vars.user_id || vars.session_id) {
      const sessionId = String(vars.session_id || vars.user_id);
      return {
        user_id: String(vars.user_id || sessionId),
        session_id: sessionId,
      };
    }
    const index = this.nextPoolIndex % this.sessionPool.length;
    this.nextPoolIndex += 1;
    return this.sessionPool[index];
  }

  async callApi(prompt, context) {
    const vars = (context && context.vars) || {};
    const userQuery =
      vars.user_query !== undefined && vars.user_query !== null && String(vars.user_query).length > 0
        ? String(vars.user_query)
        : String(prompt || '');

    if (!userQuery) {
      return { error: 'user_query 为空，无法发送请求' };
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    const t0 = Date.now();

    try {
      await this.ensureSessionPoolReady();
      const session = this.pickSession(context);
      const response = await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userQuery, ...session }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const text = await response.text().catch(() => '');
        return {
          error: `Agent HTTP ${response.status}: ${text.slice(0, 500)}`,
        };
      }

      const data = await response.json();
      const latencyMs = Date.now() - t0;

      const finalAnswer =
        typeof data.finalAnswer === 'string'
          ? data.finalAnswer
          : typeof data.response === 'string'
            ? data.response
            : '';

      const toolCalls = Array.isArray(data.tool_calls) ? data.tool_calls : [];

      return {
        output: finalAnswer,
        metadata: {
          json: data,
          tool_calls: toolCalls,
          // 供开启 tracing 后 this.trajectory 直接读取
          trajectory: toolCalls,
          latencyMs,
        },
      };
    } catch (err) {
      if (err && err.name === 'AbortError') {
        return {
          error: `Agent 请求超时: ${this.timeoutMs}ms 后仍未收到 /chat 响应，已在 promptfoo provider 侧取消；后端可能仍会继续跑完本次任务`,
        };
      }
      return { error: `Agent 请求失败: ${err && err.message ? err.message : String(err)}` };
    } finally {
      clearTimeout(timer);
    }
  }
}

module.exports = AgentProvider;
