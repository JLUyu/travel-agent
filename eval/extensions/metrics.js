function percentile(values, p) {
  if (!Array.isArray(values) || values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.ceil(p * sorted.length) - 1;
  const idx = Math.max(0, Math.min(index, sorted.length - 1));
  return sorted[idx];
}

function mean(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return 0;
  }
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function variance(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return 0;
  }
  const avg = mean(values);
  return values.reduce((sum, v) => sum + Math.pow(v - avg, 2), 0) / values.length;
}

function afterAll({ results, suite }) {
  const latencies = [];
  const toolRecalls = [];
  // 按 testId 分组收集 llm_score，用于计算"同一用例重复 n 次的评分方差"
  // promptfoo eval --repeat n 时，同一 testId 会有 n 条结果
  // promptfoo eval（默认）时，每个 testId 只有 1 条结果，方差为 N/A
  const caseScoresMap = new Map();

  for (const result of results) {
    let latencyMs = 0;
    if (result.response && result.response.metadata && result.response.metadata.latencyMs !== undefined) {
      latencyMs = result.response.metadata.latencyMs;
    } else if (result.gradingResult && result.gradingResult.namedScores && result.gradingResult.namedScores.latency !== undefined) {
      latencyMs = result.gradingResult.namedScores.latency * 1000;
    }
    if (latencyMs > 0) {
      latencies.push(latencyMs);
    }

    if (result.gradingResult && result.gradingResult.namedScores) {
      const scores = result.gradingResult.namedScores;
      if (scores.llm_score !== undefined) {
        const key = result.testId || JSON.stringify(result.vars || {});
        if (!caseScoresMap.has(key)) {
          caseScoresMap.set(key, []);
        }
        caseScoresMap.get(key).push(scores.llm_score);
      }
      if (scores.tool_recall !== undefined) {
        toolRecalls.push(scores.tool_recall);
      }
    }
  }

  console.log('\n========================================');
  console.log('           Global Metrics Summary        ');
  console.log('========================================');

  if (toolRecalls.length > 0) {
    const avgToolRecall = mean(toolRecalls);
    console.log(`  avg_tool_recall: ${avgToolRecall.toFixed(4)}`);
  } else {
    console.log(`  avg_tool_recall: N/A (no data)`);
  }

  // 计算每个用例重复 n 次的 llm_score 方差，再对所有用例的方差取平均
  // 仅对重复次数 >= 2 的用例计算方差；若所有用例都只跑了 1 次，则输出 N/A
  const perCaseVariances = [];
  const allLlmScores = [];
  for (const [, scores] of caseScoresMap) {
    allLlmScores.push(...scores);
    if (scores.length > 1) {
      perCaseVariances.push(variance(scores));
    }
  }

  if (allLlmScores.length > 0) {
    const avgLlmScore = mean(allLlmScores);
    console.log(`  avg_llm_score:   ${avgLlmScore.toFixed(4)}`);
    if (perCaseVariances.length > 0) {
      const avgVar = mean(perCaseVariances);
      console.log(`  avg_llm_score_variance (per-case across repeats): ${avgVar.toFixed(6)}`);
    } else {
      console.log(`  avg_llm_score_variance: N/A (需要 promptfoo eval --repeat n, n>=2)`);
    }
  } else {
    console.log(`  avg_llm_score:   N/A (no data)`);
    console.log(`  avg_llm_score_variance: N/A (no data)`);
  }

  console.log('----------------------------------------');
  console.log('           Latency Percentiles           ');
  console.log('----------------------------------------');

  if (latencies.length > 0) {
    const p50 = percentile(latencies, 0.50) / 1000;
    const p95 = percentile(latencies, 0.95) / 1000;
    const p99 = percentile(latencies, 0.99) / 1000;
    const avg = mean(latencies) / 1000;

    console.log(`  Samples:      ${latencies.length}`);
    console.log(`  P50:          ${p50.toFixed(3)} s`);
    console.log(`  P95:          ${p95.toFixed(3)} s`);
    console.log(`  P99:          ${p99.toFixed(3)} s`);
    console.log(`  Avg:          ${avg.toFixed(3)} s`);
  } else {
    console.log(`  No latency data available`);
  }

  console.log('========================================\n');
}

module.exports = {
  afterAll,
};