# 项目现状说明（Status Snapshot）

- **生成时间：2026-11-24 23:46（Asia/Shanghai）**
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

**评测数据链路已跑到“全量数据生成 + 人工复核待返回”阶段**：2,685 条时间样本全部恢复，
全量 Query/审核已生成；**528 条人工复核队列已单独成文档**，等待人工结果后冻结 v1.0。

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
| Phase 7 评测数据 | 🟡 全量生成完成、待人工复核 | 2,685 样本；528 条人工队列；未冻结 |
| Phase 7 三路评测 | 🟡 Recall 阶段 | 全量 2,224 PASS 的 TIGER/E5/Hybrid Recall 已跑；Reranker 全量未跑 |

## 3. 当前全量评测数据结果（v1.0）

| 项 | 数值 |
|---|---|
| 原始/恢复 test 样本 | **2,685 / 2,685**（失败 0） |
| 重复 Target / 新商品 | 23 / 2,662 |
| Profile | 2,685 条（Python 规则，无评论正文） |
| Query | **2,551 generated + 134 insufficient_profile** |
| 独立审核 | **2,224 PASS / 276 REVIEW / 51 FAIL / 134 ERROR** |
| Python 校验 | 2,551 通过；Target ID/Title 精确扫描 **0 泄漏** |
| 人工复核 | **528 条**（276 REVIEW + 51 FAIL + 134 ERROR + 67 条 PASS 抽样，约 3%） |
| 未冻结 | `final_frozen_eligible=0`；`human_review_completed=false` |
| API 成本 | prompt **4,854,920** / completion **524,067** tokens（`deepseek-flash`） |

人工复核文档：

- Markdown（便于阅读）：[human_review_v1.0.md](phase7/v1.0/human_review_v1.0.md)
- CSV（请填写后返回）：[human_review_v1.0.csv](phase7/v1.0/human_review_v1.0.csv)
- Manifest：[evaluation_manifest.json](phase7/v1.0/evaluation_manifest.json)

## 4. 全量 Recall 对照（2,224 PASS，未跑 Reranker）

| 方法 | Recall@10 | Recall@50 | Recall@100 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| TIGER-only | 0.18% | 0.22% | 0.22% | 0.11% | 0.09% |
| E5-only | **1.98%** | **3.33%** | 3.60% | 1.21% | 1.04% |
| Hybrid RRF | 1.66% | 3.06% | 3.64% | 0.83% | 0.64% |

- TIGER 独有命中 1 条；E5 未命中 2,144 条；Rescue Rate **0.05%**；
- Hybrid − E5：Recall@10 **−0.32pp**、Recall@50 **−0.27pp**（无增量）；
- 该结果与 Phase 4 No-Go 一致；这是 Recall 阶段结果，**未包含统一 Reranker**。

## 5. 端到端缺口（为什么不能算全线跑通）

| 缺口 | 现状 |
|---|---|
| 统一 E2E CLI | 无 `agr recommend`；Planner/E5/Reranker/Judgment/Controller 未注入真实依赖 |
| 真实 Judgment | DeepSeek 真实调用只做过示例商品；未对真实 E5 候选池逐商品判断 |
| Controller 闭环 | LangGraph 图级测试通过；未在真实检索结果上跑 REPLAN/STOP |
| Generator | 未实现（DeepSeek 推荐表达 + 证据约束） |
| Memory Update | Mem0 Adapter 未实现；`Memory.add` 未验证 |
| Reranker 全量对照 | 只有 Phase 3 真实演示与 100 Pilot 计划；全量 2,224 PASS 未跑 |
| 人工复核 | 528 条待返回；返回前不冻结 v1.0 |

## 6. 测试与仓库状态

| 项 | 结果 |
|---|---|
| 本地 pytest | **269 passed / 14 skipped** |
| ruff / mypy | 全绿（75 source files） |
| GitHub Actions | Python 3.10/3.11 CI 绿色（历史提交） |
| Qdrant | `agr-qdrant` v1.20.2，127.0.0.1:6333，空库运行中 |
| 服务器数据 | `/mnt/agr-data/evaluation/v1.0`（不提交 Git） |
| 仓库策略 | 数据/权重/索引/密钥不入库；报告与 manifest 入库 |

## 7. 需要人工执行的事项

1. **填写并返回** `human_review_v1.0.csv` 的 528 行：`human_decision` 填 `PASS` / `FAIL`，
   `human_notes` 写理由；不要查看 Target。
2. 返回后：应用人工结论 → 计算人工与 Auditor 一致率 → 冻结 `v1.0` → 重跑全量 Reranker 对照。
3. 如需端到端服务验收：实现并跑通 `agr recommend`（真实 Planner/E5/Reranker/Judgment/
   LangGraph Controller/Generator/Memory + 完整 trace）。

## 8. 结论

- **评测数据**：机器侧已接近收尾，只差 528 条人工复核与冻结（以及人工后 Reranker 全量对照）。
- **产品服务**：仍是“分段真实、整体未接”，**不能声明 Phase 0–6 完成或双路验收完成**。

## 9. 证据索引

- Phase 4 No-Go：[D1_B1_B2_GATE_REPORT.md](phase4/D1_B1_B2_GATE_REPORT.md)
- Phase 5：[PLANNER_JUDGMENT_CONTROLLER.md](phase5/PLANNER_JUDGMENT_CONTROLLER.md)
- Phase 6：[D9_MEMORY_DEPLOYMENT.md](phase6/D9_MEMORY_DEPLOYMENT.md)
- 评测协议：[EVALUATION_DATASET_CONSTRUCTION_PROTOCOL_V1.0.md](phase7/EVALUATION_DATASET_CONSTRUCTION_PROTOCOL_V1.0.md)
- 全量报告/覆盖/泄漏：[coverage](phase7/v1.0/coverage_report.json)、[quality](phase7/v1.0/quality_report.json)、[leakage](phase7/v1.0/leakage_check.json)、[comparison](phase7/v1.0/retrieval_comparison_v1.0.json)
