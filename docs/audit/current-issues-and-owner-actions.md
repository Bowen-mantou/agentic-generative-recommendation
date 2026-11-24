# 当前问题与负责人行动清单

- 日期：2026-11-24（19:40 更新）
- 仓库：`C:\new\intern\plan\projects\Agentic-Generative-Recommendation`，HEAD `b0ef4f5`
- 服务器：`192.0.2.10`（占位地址），项目 `/home/agr/Agentic-Generative-Recommendation`
- 用途：记录**当前仍存在的问题**，区分「需要负责人决策/提供资源」与「工程侧待办」。
  本文只陈述事实、影响和待决选项，不含倾向性建议。

## 0. 管理摘要

| 项 | 当前状态 |
|---|---|
| Phase 0–3 | 已完成并有报告/证据（审计、E5+FAISS、Fusion/约束池、Reranker） |
| Phase 4（TIGER） | **D11 Gate = No-Go**；A1/B1/B2 均未超过热度基线，Hybrid 均低于 E5-only；**未全量训练**，Phase 4 不标记完成 |
| Phase 5（Planner/Judgment/Controller） | 代码、D3–D8 硬化、LangGraph 图级运行、260 单测完成；**真实 E2E（DeepSeek+E5+Reranker+Judgment）未接线** |
| Phase 6（Generator/Memory） | 只有 D9 基础设施（Qdrant 容器 + multilingual-e5-base 验证）与占位目录；**Generator、Mem0 Adapter 未实现**，`Memory.add` 未验证 |
| Phase 7（评测） | 协议已实现；2,685 条时间审计完成；100 条 Pilot（82 PASS/14 REVIEW/1 FAIL/3 ERROR，0 泄漏）；**20 条人工复核待做**，未冻结 |
| 测试 | 本地 260 passed / 14 skipped；服务器 271 passed / 3 skipped（LangGraph 真实运行） |
| 服务器服务 | `agr-qdrant` 容器运行中（v1.20.2，127.0.0.1:6333，空库，Up 4h） |

**P0（阻塞验收）问题 4 项，P1（质量/可复现风险）10 项，负责人需执行 O1–O6。**

---

## 1. 问题清单

### P0：阻塞 Phase 0–6 验收

#### P0-1 Phase 4 No-Go 后的 Q12 偏差尚未获批准

- **事实**：D11 Gate 中 A1/B1/B2 的 TIGER Recall@10 分别为 0.149% / 0.253% / 0.074%，
  同集热度基线 0.335% / 0.815% / 0.335%，E5 用户画像 2.905% / 4.352% / 2.905%；
  Hybrid RRF 均低于 E5-only。详细数字见 [D1_B1_B2_GATE_REPORT.md](../phase4/D1_B1_B2_GATE_REPORT.md)。
- **影响**：按 Q12，不得把 E5 单路标记为“双路召回验收完成”；Phase 4 未完成。
  需要负责人批准偏差文本后，才能把 Phase 5/6 的验收口径正式写成“E5 单路 + Phase 4 未完成”。
- **当前处置**：未全量训练；TIGER 产物保留为 Pilot 证据。
- **需要负责人**：O1。

#### P0-2 Phase 5 真实 E2E 未接线

- **事实**：
  - `src/recommender/cli.py` 只有 `index / history / pool / search / rerank / tiger`，**没有 `recommend`/`plan`/`judge` 命令**；
  - `Controller.workflow.WorkflowDependencies` 需要注入 Planner/Judge/retrieve/rerank，但尚未用真实 E5、Reranker、DeepSeek 构造；
  - D6 的 `JudgmentEvidenceBuilder` 已实现，但 `workflow.py` 仍使用默认 `build_judgment_candidates(pool, ranked, plan)`，**未接入 Catalog 补证**；
  - 真实 DeepSeek smoke 只覆盖 Planner 与 2 个示例商品的 Judgment，不是 E5 候选池上的端到端运行。
- **影响**：Phase 5 不能按“真实 Planner→检索→池→约束→Reranker→Judgment→LangGraph Controller”验收；
  Phase 6 的 Generator/Memory 也无法接入完整链路。
- **需要负责人**：O4（真实 API 使用与预算上限）。

#### P0-3 D10 人工 Query 集尚未标注（Pilot 已构建，待人工复核）

- **事实**：D10 协议已工程实现；现有 2,685 条 TIGER 测试样本已 100% 恢复，100 条 Pilot 已生成
  97 条 synthetic Query（82 PASS / 14 REVIEW / 1 FAIL / 3 ERROR），0 泄漏；但
  **20 条人工复核队列尚未处理**（`human_audit.csv` 决策列全空），全量 2,685 条未生成，数据集未冻结。
  规范见 [EVALUATION_DATASET_CONSTRUCTION_PROTOCOL_V1.0.md](../phase7/EVALUATION_DATASET_CONSTRUCTION_PROTOCOL_V1.0.md)，
  结果见 [PILOT_REPORT_V0.1.md](../phase7/PILOT_REPORT_V0.1.md)。
- **影响**：Phase 7 的 E5 文本召回、Fusion、Reranker、Judgment、Planner 质量评估全部缺少主评测集；
  不得用 AI 自动生成的标签冒充人工相关性标注。
- **需要负责人**：O3（标注人员、数量、语言、截止时间）。

#### P0-4 Phase 6 主体未实现

- **事实**：`src/recommender/generation/__init__.py` 与 `src/recommender/memory/__init__.py`
  仅占位；`configs/memory.yaml` 仍为 `memory.enabled: false`、embedder `intfloat/e5-base-v2`；
  `Memory.add` 端到端（自定义 embedder 注入）未验证；Mem0 Adapter、Recommendation Generator 均未开始。
- **影响**：Phase 0–6 “完整系统”验收缺失 Generator 与 Memory 两个模块；
  D9 只是基础设施验证，不是 Phase 6 完成。
- **需要负责人**：O5（Qdrant 运维策略与 Memory embedding 模型确认）。

---

### P1：质量、可复现性与运维风险

| ID | 问题 | 事实/证据 | 影响 |
|---|---|---|---|
| P1-1 | LangGraph 未进入服务器持久环境 | `/tmp/agr/venv` 有 langgraph 1.2.24；`scripts/bootstrap_s4000.sh` **未安装** workflow extra；`/tmp` 可被清理 | 环境重建后 LangGraph 集成测试/正式入口失效 |
| P1-2 | CLARIFY 无恢复能力 | `step_clarification` 直接以 CLARIFY 终止；controller 中无 checkpointer；文档写明“CLARIFY 仍挂起返回” | 多轮澄清挂起/恢复不能验收 |
| P1-3 | Controller 阈值未校准 | `configs/controller.yaml` `calibrated: false`；`min_judgments=3` 为工程初值 | REPLAN/STOP 质量不可引用为实验结论 |
| P1-4 | R17 配置化未落地 | 客户端有“空 content + finish_reason=length → reasoning_effort=none 重试”的回退；`configs/models.yaml`/`config.py` 中**没有** `reasoning_effort` 字段 | 只能依赖自动回退；成本和截断行为不可配置/不可审计 |
| P1-5 | 真实 API 故障与配额未验证 | 429/5xx/超时仅 MockTransport；无 token/费用上限；服务器 `.env` 有 key | 真实 E2E 可能被限流/超额中断 |
| P1-6 | Qdrant 运维策略未定 | 容器仅回环、无鉴权/TLS；本地盘 named volume；NFS 快照脚本存在但**无定时任务**；`restart=unless-stopped` | 共享服务器上的长期服务、备份与安全责任未明确 |
| P1-7 | Memory embedding 配置与 D9 不一致 | D9 推荐/验证 `intfloat/multilingual-e5-base`；`configs/memory.yaml` 仍是 `intfloat/e5-base-v2`；fastembed 未装（无 BM25） | 中英混合 Memory 的召回质量与稀疏检索能力未定 |
| P1-8 | D6 未接入真实链路 | `Judge.judge_ranked` 与 builder 已实现，但 workflow 未使用；真实 Catalog 下的 details 分布未测 | Judgment 仍可能缺 description/details 证据 |
| P1-9 | 数据规模限制 | All_Beauty：8,201 用户 ≥3 事件、43,044 用户 ≥2；TIGER 已 No-Go | 任何 TIGER 重启都需先修改 Q5/数据范围并重新 Gate |
| P1-10 | 评测工程仍需产品化 | 已实现 `src/recommender/evaluation/`（时间审计/画像/生成/审核/校验）与 `agr eval` CLI；三路对照仍是 `scripts/retrieval_comparison_pilot.py` | 全量冻结、Reranker 对照与统计流程未产品化 |

---

### P2：清理与一致性问题

| ID | 问题 | 事实 | 影响 |
|---|---|---|---|
| P2-1 | 服务器工作树残留 | 服务器缺少 `data/samples/eval/manual_queries_template.csv`（sync 排除 data/）；有一个未跟踪的旧 smoke JSON | 仅 `git status` 噪音，不影响运行 |
| P2-2 | D9 文档状态与决策记录不一致 | `docs/phase6/D9_MEMORY_DEPLOYMENT.md` 写“D9 仍为待确认”；`FINAL_DECISIONS §7.1` 已写“执行状态” | 审计口径不一致，需要负责人追认 |
| P2-3 | TIGER pilot 产物保留策略未定 | `artifacts/semantic_ids/{rqvae_pilot_a1,a2,samples_a1,b1,b2,generator_*}` 仍在 NFS | 占用小（约 40 MB），但归档/保留期限未定 |
| P2-4 | CRLF/本地覆盖配置 | 本机写文件会产生 CRLF；服务器有 `configs/local.yaml` 覆盖 | 已在测试 fixture 处理；换行噪音仍可能出现 |

---

## 2. 需要负责人执行的操作（O1–O6）

> 回复格式示例：`O1=批准; O2=追认; O3=标注员A/200条/中英/12月5日; O4=上限50元/天; O5=保留+备份每日/保留7天; O6=先Phase5`。
> 未回复的项保持现状，不启动需要授权的工作。

### O1. 批准 Q12 偏差文本（Phase 4 No-Go 后）

请确认以下偏差文本（或给出修改）：

> “TIGER 方向经 D11 Gate 判定 No-Go：A1/B1/B2 均未超过热度基线，Hybrid 未超过 E5-only；
> Phase 4 标记为未完成，不进行全量训练。Phase 5/6 以 E5 单路实现与验收；Phase 7 报告中如实
> 报告 TIGER/popularity/E5/hybrid 对照；不得宣称双路召回验收完成。”

需要你回复：**批准 / 修改（给出措辞）**。

### O2. 追认 D1–D11 决策与执行记录

D1–D11 是按你转交的建议稿执行的（原文标注“待确认”），目前 `FINAL_DECISIONS §7.1` 已写入执行状态。
请确认：**追认为正式决策 / 仅追认部分（列出）/ 需要回退某项**。
特别涉及 D9（Qdrant 容器已实际运行）与 D10（人工标注尚未做）。

### O3. D10 人工标注资源

请提供：

| 项 | 需要你填写 |
|---|---|
| 标注负责人/标注员 | 姓名或角色 |
| 数量 | 建议范围内（100–300）的具体数字 |
| 语言 | 中文 / 英文 / 中英混合 |
| 截止日期 | 日期 |
| 标注依据 | 是否允许标注员查看商品 description/details（当前协议允许） |

没有该输入，Phase 7 只能做 next-item / persona 合成评测，不能完成人工主评测集。

### O4. 真实 DeepSeek API 使用授权与预算上限

Phase 5 E2E 需要使用真实 API（Planner + Judgment + 后续 Generator）。请提供：

| 项 | 需要你填写 |
|---|---|
| 是否允许使用服务器 `.env` 中的 key | 是 / 否 |
| 预算上限 | 按天或按阶段：____ tokens 或 ____ 元 |
| 失败重试上限 | 默认 2 次；是否允许调整 |
| 是否允许在 Phase 5 验收前产生真实调用费用 | 是 / 否 |

### O5. Qdrant 运维策略与 Memory embedding 模型

请确认：

| 项 | 选项 |
|---|---|
| `agr-qdrant` 是否长期保留运行 | 保留 / 停止（Phase 6 再启动） |
| 是否要求鉴权/TLS | 保持回环无鉴权 / 需要 API key / 需要 TLS |
| 备份频率与保留 | 每日 / 每周 / 手动；保留 __ 份 |
| Memory embedding | `intfloat/multilingual-e5-base`（D9 验证）/ `intfloat/e5-base-v2`（当前配置）/ 其他 |

### O6. Phase 5 与 Phase 6 的优先级

真实 E2E 与 Phase 6 实现都未开始，资源有限时请指定顺序：
**先 Phase 5 E2E / 先 Phase 6 / 并行**（可给出时间上限）。

---

## 3. 不需要负责人操作、工程侧将执行的事项

| 项 | 内容 | 依赖 |
|---|---|---|
| E1 | Phase 5 CLI E2E：`agr recommend`，接入真实 E5、Reranker、DeepSeek Planner/Judgment、LangGraph Controller、Trace | O4 |
| E2 | 将 D6 `Judge.judge_ranked` + Catalog 批量查询接入 workflow | 无 |
| E3 | 补 `configs/models.yaml`/`config.py` 的 `reasoning_effort` 并传入客户端构造 | O2（D 决策追认） |
| E4 | 更新 `scripts/bootstrap_s4000.sh` 安装 workflow extra；把 LangGraph 纳入服务器标准环境 | 无 |
| E5 | Phase 6：Recommendation Generator（DeepSeek，带证据约束）、Mem0 Adapter、`Memory.add` 端到端测试；更新 `configs/memory.yaml` | O5 |
| E6 | Phase 7：把 D11/D10 评测入口迁入 `src/recommender/evaluation/`，固定协议、seed、快照 | O3 |
| E7 | 清理服务器工作树残留、统一 D9 文档状态；提交并同步 | O2 |
| E8 | 维护风险登记与阶段报告（R2/R10/R11/R18 已更新到 `docs/audit/risk-status.md`） | 无 |

---

## 4. 证据与验证快照

| 项 | 结果 |
|---|---|
| 本地 pytest | **260 passed, 14 skipped**（2026-11-24 19:36） |
| 服务器 pytest | **271 passed, 3 skipped**（LangGraph 真实运行；2026-11-24 19:36） |
| ruff / mypy | 全绿（66 source files；提交时验证） |
| Git | `cee079b` Phase 5 → `c07e22f` D6 → `9082d3b` D3–D8 → `234f080` D1/D11 → `b0ef4f5` AGENTS；发布时已重写历史 |
| Phase 4 报告 | [TIGER_PILOT_REPORT.md](../phase4/TIGER_PILOT_REPORT.md)、[D1_B1_B2_GATE_REPORT.md](../phase4/D1_B1_B2_GATE_REPORT.md) |
| Phase 5 | [PLANNER_JUDGMENT_CONTROLLER.md](../phase5/PLANNER_JUDGMENT_CONTROLLER.md)、2 份真实 smoke JSON |
| Phase 6 | [D9_MEMORY_DEPLOYMENT.md](../phase6/D9_MEMORY_DEPLOYMENT.md)；`scripts/memory/*` |
| Phase 7 | [EVALUATION_PROTOCOL_D10.md](../phase7/EVALUATION_PROTOCOL_D10.md)；模板 0 条标注 |
| Qdrant | `agr-qdrant` v1.20.2，127.0.0.1:6333，`Up 4 hours`，collections=[]（空库） |
| TIGER 产物 | `/mnt/agr-data/artifacts/semantic_ids/`（A1/B1/B2 + checkpoints） |

## 5. 状态定义

- **P0**：不解决就无法完成 Phase 0–6 验收或 Phase 7 主评测。
- **P1**：不阻塞当前开发，但会造成质量、可复现性、成本或运维风险。
- **P2**：一致性与清理事项，不影响功能正确性。
