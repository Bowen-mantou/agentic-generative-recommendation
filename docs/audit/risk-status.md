# 风险状态记录（Gate 1 进展）

- 更新日期：2026-11-24
- 状态定义沿用 [FINAL_DECISIONS_AND_RISK_REGISTER.md](../decisions/FINAL_DECISIONS_AND_RISK_REGISTER.md) §3：
  `Open` / `In Progress` / `Mitigated` / `Accepted` / `Closed`
- 每个阶段结束更新本文件；`Mitigated` 不代表风险消失。

## 1. Gate 1 状态

| 风险 | 主题 | 状态 | 证据 / 更新 | 下一步 |
|---|---|---|---|---|
| R1 | 外部 TIGER 与 All_Beauty 不兼容 | **Mitigated**（Q1 B→A 已确认，按 A1 Pilot 执行） | 无外部权重；Cornac v3.2.0 审计通过（[TIGER_IMPLEMENTATION_PLAN.md](../phase4/TIGER_IMPLEMENTATION_PLAN.md)）；自研 RQ-VAE 已训练 | 无外部权重依赖；保留 Cornac 作校验 |
| R2 | 自建 TIGER 训练成本超预算 | **No-Go（D11）** | A1/B1/B2 均未通过 D11：TIGER Recall@10 0.15%/0.25%/0.07%，同集 popularity 0.34–0.82%、E5-profile 2.9–4.4%；hybrid 均低于 E5-only（[D1_B1_B2_GATE_REPORT.md](../phase4/D1_B1_B2_GATE_REPORT.md)） | 不启动全量训练；Phase 4 未完成；Phase 5/6 用 E5；Q12 偏差说明 |
| R3 | Reviews 下载 / 用户序列构建 | **Mitigated** | reviews 326,611,506 B（SHA-256 `36ca7c48…`）；694,252 事件 / 631,986 用户 / 112,565 商品；0 跳过；SQLite 已建并校验 | 保持时间切分与合成隔离 |
| R4 | Qwen3-Reranker 原生打分实现错误 | **Closed**（有边界） | [reranker-smoke.md](reranker-smoke.md)：`Qwen3ForCausalLM` 跑通；yes=9693 / no=2152 单 token；官方模板三方一致；10 对手写文本 100% 区分；复现模型卡样例（7.597 vs 7.625）。分数饱和，**不可当概率/阈值** | 仅用于同 query 相对排序；阈值待标注数据 |
| R5 | CPU Reranker 延迟/内存不满足 | **Mitigated**（实测数字已给出） | 512 token：单条 P50 0.41 s、batch4 摊薄 0.29 s；2048 token：单条 3.65 s、batch4 2.03 s；20 候选（512 token，batch4）≈ 6 s/query；fp32 权重 2.38 GB，batch1 峰值 3.9–5.2 GB | 批量/截断/候选数需按延迟预算设定；MUSA 未用（Q11） |
| R6 | Constraint Schema 聚合语义不一致 | **Mitigated** | Q3 实现 + Phase 2 evaluator/pool；单测覆盖全通过/任一失败/部分未知/无约束；真实 E5 候选池 99→15/75/9（[phase2-report.md](phase2-report.md)） | 保持回归 |
| R7 | e5-base-v2 与 Query 语言不匹配 | **Open** | 当前 Query 均为英文；未建人工相关性小集 | Phase 7 建小集评测 |
| R8 | 商品价格/描述缺失影响约束 | **Mitigated**（数据事实已量化） | price 15.7%、description 17.0%、features 15.4%、details 96%；真实候选池 99 个中 75 个 `unknown`、15 个可验证 | Phase 7 报告可验证/未知比例 |
| R9 | Controller 阈值缺少标注数据 | **Open** | 尚无审核集 | Phase 5/7（Q6） |
| R10 | Mem0/Qdrant/Embedding 版本或维度不兼容 | **Mitigated（部分）** | Qdrant v1.20.2 运行、15/15 冒烟、重启/备份恢复；mem0ai 2.2.1 + qdrant-client 1.20.0 round-trip；multilingual-e5-base 768 维验证（[D9 报告](../phase6/D9_MEMORY_DEPLOYMENT.md)） | 自定义 embedder 注入 + Memory.add 未验证 |
| R11 | Qdrant 用户过滤隔离 | **Mitigated** | D9：u001/u002 payload 过滤检索与 filter 删除精确隔离 | 保持 Phase 6 回归 |
| R12 | InteractionHistory 与 Mem0 混用 | **In Progress** | SQLite 事件表已实现且与 Mem0 分离；Mem0 未接入；`is_synthetic` 必填 | Phase 6 接入 Mem0 后复核 |
| R13 | NFS 备份/持久性/并发 | **In Progress** | 产物与数据已迁至 NFS；备份策略未确认（env-recon 未验证） | 负责人确认（Q10）或建副本 |
| R14 | Git 初始化误伤父仓库 | **Closed** | 独立仓库 `.git` 于项目目录；初始提交 `fab7e6b`；凭据未入 Git；父仓库未改动 | 保持 |
| R15 | 合并后 Startup Prompt 仍有冲突 | **Mitigated** | 根目录 V1.2 为唯一权威；旧版归档到 `docs/archive/`；Phase 0–7 与 Q1–Q12 已对齐 | 新增决策时同步 |
| R16 | 把“不接 MUSA”误解为训练资源已解决 | **Open** | Q11 仅限定第一阶段 CPU；TIGER 训练资源见可行性报告 §3 | 随 Q1 决策处理 |
| R17 | DeepSeek 模型 ID/JSON/配额变化 | **In Progress** | `/v1/models` 200（`deepseek-flash`、`deepseek-v4-pro`）；结构化输出/超时/重试待 Phase 5 | Phase 5 集成测试 |
| R18 | LangGraph 循环/重复检索 | **In Progress** | D3：真实 CompiledStateGraph 运行；REPLAN/预算/重复 query/失败组件 9 个集成测试；与纯 Python 状态一致 | 真实 E2E（DeepSeek+E5+Reranker+Judgment）未接 |
| R19 | 时间切分或目标泄漏 | **Open** | 评测协议未实现；InteractionStore 支持 cutoff 过滤 | Phase 7 |
| R20 | 模拟反馈误报为真实行为 | **Open** | 契约与表结构要求 `is_synthetic`；尚无模拟数据 | Phase 6/7 |

## 2. 本阶段新增证据

| 项 | 路径 |
|---|---|
| TIGER 外部实现/权重核查 | [tiger-compatibility-report.md](tiger-compatibility-report.md) |
| TIGER 自建可行性（待决策） | [tiger-feasibility-assessment.md](tiger-feasibility-assessment.md) |
| Phase 0/1 执行记录 | [phase1-report.md](phase1-report.md) |
| 语义 ID 冲突代理实验 | 服务器 `/tmp/rqvae_proxy.json`（3×256）与 `/tmp/rqvae_proxy_4x1024.json`（4×1024） |
| Interaction History | `/mnt/agr-data/data/processed/interaction_history.db` |
| Reranker 冒烟 | [reranker-smoke.md](reranker-smoke.md) |
| Phase 2 候选池 | [phase2-report.md](phase2-report.md)、[evidence/pool_demo_v2.json](evidence/pool_demo_v2.json) |
| Phase 3 精排 | [phase3-report.md](phase3-report.md)、[evidence/rerank_demo_v2.json](evidence/rerank_demo_v2.json) |
| Phase 4 Pilot | [TIGER_PILOT_REPORT.md](../phase4/TIGER_PILOT_REPORT.md)、[TIGER_IMPLEMENTATION_PLAN.md](../phase4/TIGER_IMPLEMENTATION_PLAN.md)、[D1_B1_B2_GATE_REPORT.md](../phase4/D1_B1_B2_GATE_REPORT.md) |
| Phase 5 硬化 / D6 | [PLANNER_JUDGMENT_CONTROLLER.md](../phase5/PLANNER_JUDGMENT_CONTROLLER.md) |
| D9 Memory 部署 | [D9_MEMORY_DEPLOYMENT.md](../phase6/D9_MEMORY_DEPLOYMENT.md) |

## 3. 待更新

- R4/R5 已按 `reranker-smoke.md` 回填（R4 Closed 有边界；R5 Mitigated）。
- R2：等待负责人对 Q1 的解释与路径选择。
