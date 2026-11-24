# 项目讨论简报（Current Status Briefing）

> **2026-11-25 01:15 更新：E5-only 真实 E2E G0–G6 已跑通；Phase 5/6 已接线。**
> 最新细节见 [STATUS.md](STATUS.md) 与 [E2E_INTEGRATION_REPORT.md](E2E_INTEGRATION_REPORT.md)。
> 以下正文保留 00:23 的历史快照，避免抹掉讨论过程；以最新状态为准。

- **生成时间：2026-11-25 00:23（Asia/Shanghai）**
- 仓库：https://github.com/Bowen-mantou/agentic-generative-recommendation
- 当前提交：`0488ec3`（GitHub CI：Python 3.10/3.11 均通过）
- 用途：给团队讨论用的现状说明。只陈述事实、已完成项、未完成项与待讨论问题，
  不把 Mock、图级测试或离线评测结果当作端到端验收。
- 数据与产物不在 Git：原始数据、索引、模型权重、评测 JSONL 位于 NFS（文档中用 `/mnt/agr-data` 表示）。

## 1. 一句话结论

- **评测数据 v1.0 已冻结**：2,685 条时间样本 → 2,551 条 Query → 528 条人工复核 →
  **2,224 条最终纳入**，0 条 Target 泄漏。
- **产品服务（E5-only）已端到端跑通**：`agr recommend` 已真实串联
  Memory → Planner → E5 → Fusion → Reranker → Evidence → Judgment → LangGraph Controller →
  Generator；但 **TIGER 不在主线**，不是双路召回完成。
- **Phase 4 TIGER 为 D11 No-Go**：真实 RQ-VAE + 自回归训练/生成完成，但 A1/B1/B2
  均未超过热度基线，Hybrid 未给 E5 带来增量；未全量训练，不作为默认召回。

## 2. 阶段状态

| 阶段 | 状态 | 事实 |
|---|---|---|
| Phase 0 工程骨架 | ✅ | 配置、日志、trace、Pydantic 契约 |
| Phase 1 E5 + FAISS | ✅ | 112,590 个 All_Beauty 商品；真实索引、检索、manifest |
| Phase 2 Fusion + 约束 | ✅ | 两级 RRF、三态约束、候选池真实数据演示 |
| Phase 3 Qwen3-Reranker | ✅ | Transformers 原生 CPU，真实候选池验证；分数非概率 |
| Phase 4 TIGER | ⛔ No-Go | A1/B1/B2 均未过 D11；Phase 4 未完成、未全量训练 |
| Phase 5 Planner/Judgment/Controller | 🟡 组件完成、E2E 未接 | DeepSeek 客户端、Planner/Replanner、逐商品 Judgment、LangGraph 1.2.24 + 图级测试 |
| Phase 6 Generator/Memory | 🟡 部分 | Qdrant + multilingual-e5-base 部署验证；Mem0 Adapter、Generator 未实现 |
| Phase 7 评测数据 | ✅ v1.0 冻结 | 2,224 条最终样本；528 条人工复核完成；manifest 冻结 |
| Phase 7 三路评测 | 🟡 Recall 阶段 | 2,224 条 TIGER/E5/Hybrid Recall 已跑；全量 Reranker、cluster bootstrap 未跑 |

## 3. v1.0 评测数据（冻结）

| 项 | 数值 |
|---|---|
| 原始/恢复 test 样本 | **2,685 / 2,685**（失败 0） |
| 重复 Target / 新商品 | 23 / 2,662 |
| Profile | 2,685 条（Python 规则；不使用评论正文） |
| Query | 2,551 generated + 134 insufficient_profile |
| 独立审核 | 2,224 PASS / 276 REVIEW / 51 FAIL / 134 ERROR |
| Python 校验 | 2,551 通过；Target ID/Title 精确扫描 **0 泄漏** |
| 人工复核 | 528 条：67 PASS / 461 FAIL |
| 最终纳入 / 排除 | **2,224 / 461** |
| 人审与 Auditor 一致 | 528 / 528（仅复核行；**不是 2,224 条全量人工确认**） |
| 冻结状态 | `human_review_completed=true`，`frozen=true` |
| tokens 成本 | prompt 4,854,920 / completion 524,067（`deepseek-flash`） |

**重要口径**：

- 67 条人工 PASS 全部来自 Auditor PASS 的抽样；461 条 REVIEW/FAIL/ERROR 全部被人工判 FAIL；
- 最终 2,224 条中，只有 67 条经过人工抽样确认，其余 2,157 条为 Auditor PASS 但未逐条人工复核；
- `insufficient_profile`（134 条）无 Query，不能靠人工 PASS 纳入。

文件：

- [冻结 manifest](phase7/v1.0/evaluation_manifest_final.json)
- [人工一致性报告](phase7/v1.0/human_agreement_report.json)
- [最终覆盖报告](phase7/v1.0/final_coverage_report.json)
- [已填写人工结果](phase7/v1.0/human_review_v1.0_filled.csv)
- [原始复核队列](phase7/v1.0/human_review_v1.0.md)

## 4. 最终 2,224 条的三路 Recall（未含 Reranker）

| 方法 | Recall@10 | Recall@50 | Recall@100 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| TIGER-only | 0.18% | 0.22% | 0.22% | 0.11% | 0.09% |
| E5-only | **1.98%** | **3.33%** | 3.60% | 1.21% | 1.04% |
| Hybrid RRF | 1.66% | 3.06% | 3.64% | 0.83% | 0.64% |

- TIGER 独有命中 1 条；E5 未命中 2,144 条；Rescue Rate **0.05%**；
- Hybrid 相对 E5：Recall@10 **−0.32pp**、Recall@50 **−0.27pp**；
- 这是 Recall 阶段结果，**没有统一 Reranker**，也没有用户 cluster bootstrap 置信区间；
  不能据此做最终模型结论。

## 5. 现在能真实跑通什么

**可以真实跑通/有真实证据的**

- E5 编码 + FAISS 检索（真实 112,590 商品）；
- 两级 RRF、硬约束三态、候选池构建；
- Qwen3-Reranker-0.6B 在真实候选池上的排序；
- TIGER RQ-VAE + 自回归训练/生成（但质量 No-Go）；
- DeepSeek Planner/Judgment 真实 API smoke（独立调用）；
- LangGraph 图级工作流测试（stub 依赖，不是真实 E2E）；
- 评测数据全链路 A–D（时间审计、Profile、Query、Auditor、校验、人工复核、冻结）。

**还不能跑通的**

- 没有 `agr recommend` 单命令串起真实 Planner → E5 → Fusion → Reranker → Judgment →
  Controller → Generator → Memory；
- Judgment 未在真实 E5 候选池上逐商品跑；
- Controller 未在真实检索结果上跑 REPLAN/STOP；
- Generator、Mem0 Adapter、`Memory.add` 未实现；
- 没有一条完整 E2E trace；
- 全量 Reranker 对照、用户 cluster bootstrap 未做。

## 6. 主要风险与未决事项

| 风险/事项 | 现状 |
|---|---|
| TIGER 路线 | D11 No-Go；是否关闭、延期或换数据范围未定 |
| Q12 验收口径 | E5 单路能否作为 Phase 0–6 验收，需要负责人确认偏差文本 |
| E2E 优先级 | Phase 5 E2E、Phase 6 Generator/Memory、Phase 7 Reranker 全量，谁先做未定 |
| 人审抽样 | 67/2,224 抽样全部通过；是否扩大抽检未定 |
| 评测完整性 | 2,224 条的 Reranker、cluster bootstrap 未完成 |
| Controller 阈值 | `calibrated=false`；min_judgments=3 等为工程初值 |
| 部署/安全 | Qdrant 仅回环、无鉴权/TLS；备份未定时；API 成本未设上限 |
| 数据范围 | All_Beauty 仅 8,201 个 ≥3 事件用户，限制 TIGER 可学习性 |

## 7. 建议讨论议题（供会议使用）

1. **TIGER 是否正式停止**：接受 D11 No-Go、保留 Pilot 证据，还是修改 Q5 数据范围后重做？
2. **Phase 0–6 验收定义**：是否批准“E5 单路 + Phase 4 未完成”的 Q12 偏差？是否允许以 E5 单路
   完成 Phase 5/6 验收，但明确不宣称双路完成？
3. **下一步优先级**：
   - A. 先接 Phase 5 真实 E2E（Planner→E5→Reranker→Judgment→LangGraph Controller）；
   - B. 先实现 Phase 6 Generator + Mem0 Adapter；
   - C. 先补完 2,224 条的 Reranker 对照与 cluster bootstrap。
4. **人工抽检策略**：67/2,224 的抽样是否足够？是否需要扩大到 200–300 条或做分层抽检？
5. **评测报告口径**：TIGER/E5/Hybrid 的 Recall-only 结果是否作为 Phase 7 的主结论，
   还是等 Reranker 后再定？
6. **部署与成本**：Qdrant 是否长期保留？是否加鉴权/TLS？DeepSeek API 预算上限是多少？

## 8. 建议的最小下一步（技术侧）

按优先级可组合执行，但需要会议确定顺序：

1. **E2E 接线**：新增 `agr recommend`，用真实 E5、Reranker、DeepSeek Planner/Judgment、
   LangGraph Controller 跑一条带 trace 的请求；再接 D6 Evidence Builder；
2. **Phase 6**：实现 Recommendation Generator、Mem0 Adapter，验证 `Memory.add` 与用户隔离；
3. **评测收尾**：2,224 条全量 Reranker、用户 cluster bootstrap、最终 Phase 7 报告；
4. **风险收尾**：LangGraph 纳入 bootstrap、Qdrant 鉴权/备份、API 预算与失败限流。

## 9. 证据索引

- 现状快照（更短版）：[docs/STATUS.md](STATUS.md)
- Phase 4 No-Go：[D1_B1_B2_GATE_REPORT.md](phase4/D1_B1_B2_GATE_REPORT.md)
- Phase 5：[PLANNER_JUDGMENT_CONTROLLER.md](phase5/PLANNER_JUDGMENT_CONTROLLER.md)
- Phase 6：[D9_MEMORY_DEPLOYMENT.md](phase6/D9_MEMORY_DEPLOYMENT.md)
- 评测协议：[EVALUATION_DATASET_CONSTRUCTION_PROTOCOL_V1.0.md](phase7/EVALUATION_DATASET_CONSTRUCTION_PROTOCOL_V1.0.md)
- v1.0：[final coverage](phase7/v1.0/final_coverage_report.json)、[human agreement](phase7/v1.0/human_agreement_report.json)、[comparison](phase7/v1.0/retrieval_comparison_v1.0.json)
- 测试：本地 269 passed / 14 skipped；CI 3.10/3.11 green
