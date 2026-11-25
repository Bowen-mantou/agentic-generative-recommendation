# 下一阶段路线（P0–P4）执行状态

- 日期：2026-11-25
- 基线：`e2e-baseline-v1`（5ea7b5a；E2E code `a1c987f`）
- 当前结论：**P0 完成；P1 完成 Top-10 版本；P2–P4 待执行**。
- 关联：[E2E 报告](E2E_INTEGRATION_REPORT.md)、[P1 报告](phase7/P1_RERANKER_REPORT.md)、[STATUS](STATUS.md)

## P0 — 冻结 E2E 基线 ✅

| 项 | 结果 |
|---|---|
| Commit/tag | `5ea7b5a` / `e2e-baseline-v1`；E2E code `a1c987f` |
| 配置哈希 | 见 [BASELINE_V1.md](BASELINE_V1.md) |
| Prompt | planner.initial.v1 / planner.replan.v1 / judgment.v1 / recommendation_generator_v1 |
| Trace | 3 条 STOP + 1 条 REPLAN + feedback，`docs/e2e/` |
| CI | run `45506379957` / Python 3.10+3.11 success |
| 文档口径 | 已修正 BRIEFING 的旧“E2E 未接”描述、测试数 308/15、BRIEFING 下一步 |

**冻结规则**：不得删除/跳过测试、不得把 fallback 当真实 Generator、不得把 E5-only 说成双路完成。
后续 P1–P4 均以该 tag 为对照；改动冻结评测集必须新增版本。

## P1 — Reranker 完整评测 🟡（Top-10 完成）

### 已完成

- 2,224 条 frozen v1.0 样本；
- E5-only Top-100 vs E5 + Qwen3-Reranker Top-10；
- NDCG@10 / MRR / Recall@10（+ E5 Recall@50/100）；
- 用户 cluster bootstrap 1,000 次；延迟。

结果：

| 方法 | Recall@10 | NDCG@10 | MRR | 延迟/样本 |
|---|---:|---:|---:|---:|
| E5-only | 1.98% | 0.0121 | 0.0104 | 8.3 ms |
| E5 + Reranker | 1.98% | 0.0127 | 0.0104 | 1,306 ms |

- NDCG delta +0.0006，95% CI [-0.0008, +0.0019]，bootstrap P(delta>0)=0.783；
- **无显著增益**；Reranker 慢约 157×；
- 原因是 Top-10 重排无法改变 Recall@10，且下一商品标签与语义相关性不完全一致。

### 未完成 / 下一步

- Top-50 / Top-100 候选预算（CPU 估算 4h/8h；需要 GPU 或并行）；
- 相关性人工标注（当前只有下一商品标签）；
- 位置偏差、多样性与 query-level 分析。

**P1 决策建议**：暂不把 Reranker 设为默认；保留可选阶段。若业务要求精排，先补 Top-100 + GPU 预算，再决定。

## P2 — Judgment + Controller 优化 ⏳

目标：判断 Judgment 是否可靠、STOP/REPLAN 是否值得、阈值如何校准。

### 建议实验

1. **Judgment 准确性（人工小样本）**
   - 从 v1.0 分层抽 100 条 Query（有/无 target 命中各半）；
   - 每条取 E5 Top-5，共 500 个 Judgment；
   - 人工标注：`preference_fit` 是否合理、attribute evidence 是否能从 catalog 字段复核、
     `match_assessment` 是否与人工判断一致；
   - 指标：schema 合法率、证据可追溯率、match accuracy、KR-20/Cohen's κ（如双人标注）；
   - 禁止用 target 商品当唯一正例，避免“下一商品=偏好”的循环论证。
2. **Controller 决策质量**
   - 同一批 Query 跑：E5-only + Judgment + Controller（无 Reranker）；
   - 让标注者判断 STOP 结果是否可接受、REPLAN 是否产生新有效候选；
   - 指标：STOP precision、REPLAN 收益（新 query 的 Recall/质量）、API 调用数、延迟；
   - 阈值网格：`required_count`、`min_judgments`、`min_valid_candidates`、`max_unknown_ratio`；
     选择“质量/成本”Pareto 点，不追求单点最优。
3. **成本**
   - 记录每请求 Planner/Judgment/Generator token 与延迟；
   - 给出 `max_replans=1` 与 `2` 的对比；无收益则保持 1。

**暂不执行原因**：需要人工标注预算（约 100 条 × 5 候选）和新的真实 API 运行；
先完成标注协议再跑，避免用未标注的 E2E trace 冒充校准。

## P3 — Memory 优化 ⏳

- 对比 `infer=False`（当前）与 `infer=True`：
  - 30 条显式反馈（正/负/冲突/多偏好）；检查抽取的 preference、冲突处理、是否误把推断当事实；
  - 指标：抽取 precision/recall、冲突检测、重复率、检索 P@5、延迟、成本；
- 用户隔离：A/B 两用户交叉读取；模拟数据 `is_synthetic=true` 隔离；
- 反馈更新：同一商品二次反馈是否覆盖/追加；错误反馈如何撤销；
- 迁移/回滚：Qdrant collection 版本化，禁止直接改 schema。

**未做原因**：Mem0 `infer=True` 需要配置 LLM（当前 Adapter 固定 infer=False），
且需要人工判断抽取质量；先定标注规则。

## P4 — 工程可靠性 ⏳

| 项 | 现状 | 下一步 |
|---|---|---|
| Bootstrap | 已补 `langgraph`/`mem0ai`/`qdrant-client`（见脚本） | 在空环境重跑验证 |
| Qdrant 安全 | 127.0.0.1、无鉴权/TLS | 加 API key/TLS 或反向代理；不暴露公网 |
| 备份 | D9 已有快照/恢复演练 | 定时备份 + 恢复演练记录 |
| 并发 | 未压测 | 4/8 并发 E2E，记录 QPS/P99/错误率 |
| API 预算 | 单请求 20 次上限 | 日预算/告警；Provider 侧限额 |
| 监控 | JSONL trace + 日志 | 指标导出（延迟、失败率、token、STOP 率） |
| 失败注入 | 已有单元测试 | 真实 Qdrant/DeepSeek 断连演练 |

## 实验组 A/B（用户指定）

| 组 | 配置 | 指标 | 当前状态 |
|---|---|---|---|
| A Baseline | E5 + FAISS 直接相似度排序 | NDCG@10、MRR、Recall@K、Latency | ✅ 2,224 条：Recall@10 1.98%，NDCG .0121，MRR .0104，8.3 ms |
| B Reranker | E5 Top-K → Qwen3-Reranker 重排 | 同上 | 🟡 Top-10：NDCG .0127（CI 跨 0），1,306 ms；Top-50/100 未跑 |

## 建议的下一步（按投入产出）

1. **P1 收尾**：若继续 Reranker，先跑 Top-100 的 500 条分层子集，确认是否有增益，再决定全量；
2. **P2 标注**：先做 100 条 Judgment 人工标注 + Controller 决策标注，成本可控；
3. **P3**：只做 `infer=True` 小样本，先看抽取质量再谈长期偏好；
4. **P4**：先补 Qdrant 鉴权/备份和 4 并发，不急于上生产监控。

> 本文件是执行路线，不是“已完成”声明；P2–P4 未跑之前不得写入结果指标。
