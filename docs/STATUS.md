# 项目现状说明（Status Snapshot）

- **生成时间：2026-11-25 01:15（Asia/Shanghai）**
- 仓库：https://github.com/Bowen-mantou/agentic-generative-recommendation
- 当前代码基线：见仓库最新提交（`git log -1`）
- 服务器：`192.0.2.10`（公开文档占位地址），数据在 NFS `/mnt/agr-data`
- 讨论简报：[BRIEFING.md](BRIEFING.md)；E2E 报告：[E2E_INTEGRATION_REPORT.md](E2E_INTEGRATION_REPORT.md)

> 本文件是**快照**，如实区分“已真实跑通 / 仅有代码与测试 / 未实现”。
> 不把 Mock、图级测试或离线评测数据当作端到端服务验收。

## 1. 一句话结论

- **E5-only 真实 E2E 已跑通**：`agr recommend` 完成
  Memory Read → Planner → E5/FAISS → Fusion/约束 → Reranker → Evidence → Judgment →
  LangGraph Controller（STOP/REPLAN）→ Generator；3 次完整运行、1 次自然 REPLAN。
- **TIGER 仍为 D11 No-Go**：本轮以 E5-only 完成工程集成，**不宣称双路召回验收通过**；
  Q12 偏差仍需课程/老师确认。
- **评测数据 v1.0 已冻结**：2,224 条最终样本；528 条人工复核已应用。

## 2. 阶段状态

| 阶段 | 状态 | 事实 |
|---|---|---|
| Phase 0 工程骨架 | ✅ | 配置、日志、trace、Pydantic 契约 |
| Phase 1 E5 + FAISS | ✅ | 112,590 个 All_Beauty 商品；真实索引/检索 |
| Phase 2 Fusion + 约束 | ✅ | 两级 RRF、三态约束、真实候选池 |
| Phase 3 Qwen3-Reranker | ✅ | Transformers 原生 CPU；真实 E2E 中 50 候选评分 |
| Phase 4 TIGER | ⛔ No-Go | A1/B1/B2 未过 D11；保留产物与报告，不进入主线 |
| Phase 5 Planner/Judgment/Controller | ✅ E2E 通过 | DeepSeek Planner/Judgment、LangGraph 真实运行；STOP/REPLAN 有 trace |
| Phase 6 Generator/Memory | ✅ 本轮实现 | DeepSeek Generator、Mem0 Adapter；E2E 与显式反馈写回验证 |
| Phase 7 评测数据 | ✅ v1.0 冻结 | 2,224 条最终样本；人工复核完成；0 泄漏 |
| Phase 7 指标实验 | 🟡 Recall 阶段 | 全量 Reranker、Cluster Bootstrap 未跑（本轮不阻塞 E2E） |

## 3. E2E 真实证据

| 运行 | 结果 | 耗时 | 证据 |
|---|---|---:|---|
| stop run 1（新用户，空记忆） | quality_satisfied，5 推荐 | 61.6 s | `docs/e2e/run_stop_initial_*` |
| stop run 2（feedback 后） | quality_satisfied，5 推荐，1 memory hit | 57.9 s | `docs/e2e/run_stop_previous_*` |
| stop run 3（重复） | quality_satisfied，5 推荐，1 memory hit | 66.5 s | `docs/e2e/run_stop_latest_*` |
| REPLAN run | 第一轮 replan、第二轮 budget_exhausted，0 推荐 | 25.3 s | `docs/e2e/run_replan_*` |
| feedback write | add + read back 1 条；其他用户 0 条 | 14.1 s | `docs/e2e/feedback_writeback.json` |

- Trace/summary 记录 request/trace ID、候选、Reranker 分数、Judgment、Controller 决策、
  StopReason、Generator、耗时和 token（可获得处）；
- 真实 E2E 集成测试：服务器 `1 passed`（35.9 s）；
- 生成器真实模型 `deepseek-flash`，输出商品均来自 Controller 选择；语言修复后英文请求
  返回英文 reasons。

## 4. 评测数据 v1.0（冻结）

| 项 | 数值 |
|---|---|
| 时间样本恢复 | 2,685 / 2,685 |
| Query | 2,551 generated + 134 insufficient_profile |
| 独立审核 | 2,224 PASS / 276 REVIEW / 51 FAIL / 134 ERROR |
| 人工复核 | 528 条：67 PASS / 461 FAIL |
| 最终纳入 / 排除 | 2,224 / 461 |
| 泄漏 | 0 |
| 冻结 | `human_review_completed=true`，`frozen=true` |

> 2,224 条中仅 67 条经过人工抽样确认；其余为 Auditor PASS 但未逐条人工复核。
> 全量 Reranker 与 Cluster Bootstrap 仍待做。

## 5. 不能声称的事情

- ❌ “双路召回验收通过”；
- ❌ “TIGER 已完成/全量训练成功”；
- ❌ “2,224 条全部人工复核”；
- ❌ “Controller 阈值已校准”（`calibrated=false`）；
- ❌ “REPLAN 无候选时的 Generator fallback 等于真实生成成功”。

## 6. 剩余工作

1. 2,224 条全量 Reranker + 用户 Cluster Bootstrap（Phase 7）；
2. Q12 偏差确认与最终答辩口径；
3. Controller 阈值校准（Phase 7）；
4. Mem0 `infer=True`、BM25、并发压测；
5. Qdrant 鉴权/TLS 与备份调度；
6. 生产化（服务化 API、监控、成本上限）。

## 7. 测试与仓库

| 项 | 结果 |
|---|---|
| 本地 pytest | **308 passed / 15 skipped** |
| 真实 E2E 测试 | 服务器 1 passed（35.9 s） |
| ruff / mypy | 全绿（80 source files） |
| GitHub Actions | Python 3.10/3.11（提交后运行） |
| Qdrant | `agr-qdrant` v1.20.2，127.0.0.1:6333，collection `agr_user_memory` |
| 仓库策略 | 数据/权重/索引/密钥不入库；trace/summary 入 `docs/e2e/`，完整 trace 在 NFS |

## 8. 结论

- **工程主线（E5-only）已经端到端跑通并可复现**，具备演示与验收证据；
- **研究目标（TIGER + E5 双路）未完成**：TIGER D11 No-Go，必须如实报告；
- 评测数据已冻结，剩余的 Reranker/统计与控制器校准属于后续增强，不影响本轮 E2E 验收。

## 9. 证据索引

- E2E：[E2E_INTEGRATION_REPORT.md](E2E_INTEGRATION_REPORT.md)、[E2E_RUNBOOK.md](E2E_RUNBOOK.md)、[ACCEPTANCE_CHECKLIST.md](ACCEPTANCE_CHECKLIST.md)
- Phase 4 No-Go：[D1_B1_B2_GATE_REPORT.md](phase4/D1_B1_B2_GATE_REPORT.md)
- Phase 6：[GENERATOR_MEMORY.md](phase6/GENERATOR_MEMORY.md)、[D9_MEMORY_DEPLOYMENT.md](phase6/D9_MEMORY_DEPLOYMENT.md)
- 评测：[EVALUATION_DATASET_CONSTRUCTION_PROTOCOL_V1.0.md](phase7/EVALUATION_DATASET_CONSTRUCTION_PROTOCOL_V1.0.md)、[v1.0 manifest](phase7/v1.0/evaluation_manifest_final.json)
- 已知问题：[KNOWN_ISSUES.md](KNOWN_ISSUES.md)
