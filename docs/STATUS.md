# 项目现状说明（Status Snapshot）

- **生成时间：2026-11-24 23:55（Asia/Shanghai）**
- 仓库：https://github.com/Bowen-mantou/agentic-generative-recommendation
- 当前代码基线：见仓库最新提交（`git log -1`）
- 服务器：`192.0.2.10`（文档占位地址），数据在 NFS `/mnt/agr-data`

> 本文件是**快照**，如实区分“已真实跑通 / 仅有代码与测试 / 未实现”。
> 不把 Mock、图级测试或离线评测数据当作端到端服务验收。

## 1. 一句话结论

**产品链路还不能算“全线跑通”**：E5/FAISS、Fusion/约束、Reranker、TIGER Pilot、
Phase 5 的组件与 LangGraph 图级流程、DeepSeek 真实调用都已分别验证；但**还没有一条
真实的 `Planner → E5 → Fusion → Reranker → Judgment → Controller → Generator → Memory`
单命令端到端链路**。Phase 4 TIGER 为 D11 No-Go；Phase 6 Generator/Memory 未实现。

**评测数据链路已冻结 v1.0**：2,685 条时间样本全部恢复；全量 Query/审核完成；
人工复核 528 条已返回并应用；最终 **2,224 条**通过（human PASS + 生成成功 + 校验通过），
461 条被人工排除；manifest 已标记 `frozen=true`、`human_review_completed=true`。

## 2. 阶段状态

| 阶段 | 状态 | 事实 |
|---|---|---|
| Phase 0 工程骨架 | ✅ | 配置、日志、trace、Pydantic 契约 |
| Phase 1 E5 + FAISS | ✅ | 112,590 个 All_Beauty 商品，真实索引与检索 |
| Phase 2 Fusion + 约束 | ✅ | 两级 RRF、三态约束、候选池真实演示 |
| Phase 3 Qwen3-Reranker | ✅ | Transformers 原生 CPU，真实候选池验证 |
| Phase 4 TIGER | ⛔ No-Go | A1/B1/B2 均未过 D11；TIGER 不作为默认召回 |
| Phase 5 Planner/Judgment/Controller | 🟡 组件完成、E2E 未接 | LangGraph 1.2.24 + 图级测试；无真实 E5→Judgment→Controller 闭环 |
| Phase 6 Generator/Memory | 🟡 部分 | Qdrant + multilingual-e5-base 部署验证；Mem0 Adapter、Generator 未实现 |
| Phase 7 评测数据 | ✅ v1.0 已冻结 | 2,224 条最终样本；人工复核完成；manifest 已冻结 |
| Phase 7 三路评测 | 🟡 Recall 阶段 | 2,224 条 TIGER/E5/Hybrid Recall 已跑；全量 Reranker 未跑 |

## 3. v1.0 冻结结果

| 项 | 数值 |
|---|---|
| 原始/恢复 test 样本 | **2,685 / 2,685**（失败 0） |
| 重复 Target / 新商品 | 23 / 2,662 |
| Profile | 2,685 条（Python 规则，无评论正文） |
| Query | 2,551 generated + 134 insufficient_profile |
| 独立审核 | 2,224 PASS / 276 REVIEW / 51 FAIL / 134 ERROR |
| Python 校验 | 2,551 通过；Target ID/Title 精确扫描 **0 泄漏** |
| 人工复核 | 528 条：**67 PASS / 461 FAIL**（抽样 PASS 67 条全部通过；全部 461 条 flagged 被拒） |
| 人工结论应用 | **最终纳入 2,224 条；排除 461 条** |
| 人工与 Auditor 一致率 | 528/528（仅限被复核行；**不代表全部 2,224 条逐条人工确认**） |
| 冻结状态 | ✅ `human_review_completed=true`，`frozen=true` |
| API 成本 | prompt 4,854,920 / completion 524,067 tokens（`deepseek-flash`） |

关键文件：

- 冻结 manifest：[evaluation_manifest_final.json](phase7/v1.0/evaluation_manifest_final.json)
- 人工审核汇总：[human_agreement_report.json](phase7/v1.0/human_agreement_report.json)
- 最终覆盖：[final_coverage_report.json](phase7/v1.0/final_coverage_report.json)
- 已填写人工结果：[human_review_v1.0_filled.csv](phase7/v1.0/human_review_v1.0_filled.csv)
- 原始复核文档（空白决策列）：[human_review_v1.0.md](phase7/v1.0/human_review_v1.0.md)

> 人工结论说明：复核样本中所有 Auditor 标为 REVIEW/FAIL/ERROR 的 461 条均被判 FAIL；
> Auditor PASS 的 67 条抽样全部被判 PASS。一致率为 100%，但这只覆盖 528 条复核行；
> 其余 2,157 条 Auditor PASS 样本未逐条人工确认，这是抽样 QC 而非全量标注。

## 4. 三路 Recall 对照（2,224 条最终样本，未跑 Reranker）

| 方法 | Recall@10 | Recall@50 | Recall@100 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| TIGER-only | 0.18% | 0.22% | 0.22% | 0.11% | 0.09% |
| E5-only | **1.98%** | **3.33%** | 3.60% | 1.21% | 1.04% |
| Hybrid RRF | 1.66% | 3.06% | 3.64% | 0.83% | 0.64% |

- TIGER 独有命中 1 条；E5 未命中 2,144 条；Rescue Rate **0.05%**；
- Hybrid − E5：Recall@10 **−0.32pp**、Recall@50 **−0.27pp**（无增量）；
- 与 Phase 4 No-Go 一致；这是 Recall 阶段结果，**未包含统一 Reranker**。

## 5. 端到端缺口（为什么不能算全线跑通）

| 缺口 | 现状 |
|---|---|
| 统一 E2E CLI | 无 `agr recommend`；Planner/E5/Reranker/Judgment/Controller 未注入真实依赖 |
| 真实 Judgment | DeepSeek 真实调用只做过示例商品；未对真实 E5 候选池逐商品判断 |
| Controller 闭环 | LangGraph 图级测试通过；未在真实检索结果上跑 REPLAN/STOP |
| Generator | 未实现（DeepSeek 推荐表达 + 证据约束） |
| Memory Update | Mem0 Adapter 未实现；`Memory.add` 未验证 |
| Reranker 全量对照 | 2,224 条最终样本未跑统一 Reranker |
| 统计区间 | 尚未按用户 cluster bootstrap 报告最终置信区间 |

## 6. 测试与仓库状态

| 项 | 结果 |
|---|---|
| 本地 pytest | **269 passed / 14 skipped** |
| ruff / mypy | 全绿（75 source files） |
| GitHub Actions | Python 3.10/3.11 CI 绿色 |
| Qdrant | `agr-qdrant` v1.20.2，127.0.0.1:6333，空库运行中 |
| 服务器数据 | `/mnt/agr-data/evaluation/v1.0`（不提交 Git） |
| 仓库策略 | 数据/权重/索引/密钥不入库；报告与 manifest 入库 |

## 7. 剩余工作

1. **评测**：在 2,224 条最终样本上跑统一 Reranker 对照，并按用户 cluster bootstrap 报告置信区间。
2. **产品 E2E**：实现 `agr recommend`，真实串接 Planner/E5/Reranker/Judgment/LangGraph Controller，
   再实现 Generator 与 Mem0 Adapter，最后跑完整 trace。
3. **Phase 5 验收**：真实 E5 候选池上的 Judgment、CLARIFY 恢复、预算/失败路径。
4. **Phase 6 验收**：Generator、Memory.add、反馈闭环。

## 8. 结论

- **评测数据 v1.0：已完成并冻结**（2,224 条，人工复核已应用）。
- **产品服务：仍是“分段真实、整体未接”**，**不能声明 Phase 0–6 完成或双路验收完成**。

## 9. 证据索引

- Phase 4 No-Go：[D1_B1_B2_GATE_REPORT.md](phase4/D1_B1_B2_GATE_REPORT.md)
- Phase 5：[PLANNER_JUDGMENT_CONTROLLER.md](phase5/PLANNER_JUDGMENT_CONTROLLER.md)
- Phase 6：[D9_MEMORY_DEPLOYMENT.md](phase6/D9_MEMORY_DEPLOYMENT.md)
- 评测协议：[EVALUATION_DATASET_CONSTRUCTION_PROTOCOL_V1.0.md](phase7/EVALUATION_DATASET_CONSTRUCTION_PROTOCOL_V1.0.md)
- v1.0 冻结：[manifest](phase7/v1.0/evaluation_manifest_final.json)、[coverage](phase7/v1.0/final_coverage_report.json)、[agreement](phase7/v1.0/human_agreement_report.json)、[comparison](phase7/v1.0/retrieval_comparison_v1.0.json)
