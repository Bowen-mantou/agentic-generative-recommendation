# 待负责人决策事项（Phase 4–7，2026-11-24 更新）

> 用途：集中列出**需要项目负责人决定**的事项，便于逐条答复。本文只陈述事实、可选方案与影响，
> 不含推荐或倾向性意见。
> 前序记录：[FINAL_DECISIONS_AND_RISK_REGISTER.md](FINAL_DECISIONS_AND_RISK_REGISTER.md)（Q1–Q12 已确认）、
> [open-questions.md](open-questions.md)（历史提问）、
> [TIGER_PILOT_REPORT.md](../phase4/TIGER_PILOT_REPORT.md)、
> [PLANNER_JUDGMENT_CONTROLLER.md](../phase5/PLANNER_JUDGMENT_CONTROLLER.md)。

**回复格式示例**：`D1=B; D2=2 小时/最多 3 次; D3=A; D4=A; ...`（也可只回复要改的项，其余保持现状）。

## 总览

| ID | 待决问题 | 影响 | 状态 |
|---|---|---|---|
| D1 | A1 Pilot 未达门槛后，TIGER 走哪条路线 | Phase 4 是否继续、Q12 双路验收 | 待定 |
| D2 | 若继续 TIGER，Pilot/训练的资源上限 | 时间、算力、排期 | 待定 |
| D3 | Phase 5 的 LangGraph 要求 | Phase 5 验收、依赖 | 待定 |
| D4 | API 失败的 StopReason 契约 | 契约变更、轨迹语义 | 待定 |
| D5 | 重复 Query 检测语义 | State 字段、阈值是否可用 | 待定 |
| D6 | Judgment 输入的商品元数据范围 | 判断证据、Prompt 成本 | 待定 |
| D7 | Planner 失败时的降级语义 | 可用性 vs 不伪造计划 | 待定 |
| D8 | 质量 STOP 是否必须有 Judgment | Controller 行为、评测口径 | 待定 |
| D9 | Phase 6 Qdrant 部署与 Embedding 选择 | 服务部署、运维、隔离 | 待定 |
| D10 | Phase 7 的 E5 评测 Query 构造方式 | 评测可比性、防泄漏 | 待定 |
| D11 | TIGER Pilot→全量训练的量化门槛 | 是否进入全量训练 | 待定 |

---

## D1. A1 Pilot 未达门槛后，TIGER 路线选择

**已确认事实**

- A1（3 级 × 256、序列 ≥3）已完成：RQ-VAE 编码健康（100,702 唯一 ID、17.5% 碰撞、
  val MSE 1.87e-4、CPU 77 s）；生成器 470k 参数、CPU 111 s。
- 测试集（2,685 样本）：TIGER Recall@10 **0.15%** / Recall@50 0.19%；
  热度基线 0.34% / 0.78%；E5 用户画像（500 样本）1.6% / 2.6%。
- 可用用户序列：≥3 事件 8,201 人（19,222 样本）；≥2 事件 43,044 人。
- 训练资源不是瓶颈（RQ-VAE 77 s、生成器 111 s CPU）；瓶颈是样本规模与泛化。

**可选方案**

| 方案 | 内容 |
|---|---|
| A | 直接运行 A2（4 级 × 1024、序列 ≥2），作为一次性对照（同时改变码本与样本范围） |
| B | 拆成两组受控实验：B1 = 3×256 + 序列 ≥2（只变样本范围）；B2 = 4×1024 + 序列 ≥3（只变码本） |
| C | 修改 Q5 数据范围（增加品类/数据量）后重做 Pilot |
| D | 重新定义 Phase 4 验收：保留“真实生成链路”证据，但 TIGER 不作为主力召回；Phase 7 如实报告其效果 |
| E | 偏差方案：Phase 4 标记未完成，以 E5 单路推进 Phase 5/6，按 Q12 提交偏差说明 |

**影响**：Phase 4 能否完成、Phase 7 的 TIGER-only/双路对照是否可做、剩余人时分配。
**需要你提供**：选择哪一项；若选 A/B，是否接受相应地调整 Q5/Q12 的验收表述。

---

## D2. 若继续 TIGER，Pilot/训练的资源上限

**已确认事实**：Q11 固定 CPU 基线；单次 RQ-VAE 77 s、生成器 111 s；A2 预计更慢
（4×1024 码本、更多样本），但未实测。当前没有 wall-clock 上限，无法判断“超预算”的边界。

**可选方案**（可组合）

| 维度 | 选项 |
|---|---|
| 单实验时长 | ≤30 分钟 / ≤2 小时 / ≤1 个工作日 / 不设上限 |
| 并排实验数 | 最多 1 组 / 最多 3 组 / 更多 |
| 失败重试 | 不允许自动重试 / 允许 1 次配置修正 / 允许调参搜索 |

**影响**：排除“无限试错”；影响 Phase 4 排期与 Phase 5/6 并行度。
**需要你提供**：具体上限数值。

---

## D3. Phase 5 的 LangGraph 要求

**已确认事实**

- 决策固定“Controller 使用 LangGraph + Python”；当前服务器 venv **未安装 langgraph**。
- 已实现等价纯 Python 状态机（`src/recommender/controller/workflow.py`），节点映射写入
  `docs/phase5/PLANNER_JUDGMENT_CONTROLLER.md`；70 个 Phase 5 单测通过，但未在 LangGraph 运行时验证。
- `pyproject.toml` 已声明 `workflow` 可选依赖组（langgraph / langchain-core）。

**可选方案**

| 方案 | 内容 |
|---|---|
| A | 安装 langgraph，并增加一层 LangGraph 适配（保留纯 Python 状态机作为回退） |
| B | 接受纯 Python 状态机为 Phase 5 等价实现（需在验收记录中写明偏离） |
| C | 安装 langgraph，但仅用于集成测试，运行路径仍用纯 Python |

**影响**：Phase 5 验收表述、依赖体积、后续可复现性。
**需要你提供**：选择哪一项。

---

## D4. API 失败的 `StopReason` 契约

**已确认事实**：`StopReason` 枚举当前只有 `quality_satisfied / budget_exhausted /
max_rounds_reached / no_valid_candidates / retrieval_degraded`；API 失败目前映射到
`retrieval_degraded` 或 `no_valid_candidates`，并在 reasons 中注明，没有独立枚举值。

**可选方案**

| 方案 | 内容 |
|---|---|
| A | 新增 `external_api_failed`（Schema 变更，需同步测试与文档） |
| B | 继续复用 `retrieval_degraded`（reasons 区分） |
| C | 新增 `api_failed` 且区分 Planner/Judgment/Generator 子类 |

**影响**：运行轨迹语义、Controller 评测口径、下游兼容性。
**需要你提供**：选择哪一项。

---

## D5. 重复 Query 检测语义

**已确认事实**：`RecommendationState` 只有 `seen_query_hashes: list[str]`；
`configs/controller.yaml` 中有 `duplicate_query_similarity: 0.95`，但没有 Query 原文，
无法计算相似度。当前实现为**严格 hash 相等**检测，阈值配置未生效。

**可选方案**

| 方案 | 内容 |
|---|---|
| A | State 增加 `seen_queries: list[str]`（保留原文），实现相似度阈值 |
| B | 明确只做严格 hash 检测，删除/标注无效的 `duplicate_query_similarity` 配置 |
| C | 仅存归一化文本摘要（去掉停用词/排序）后再 hash |

**影响**：State Schema、防重复检索效果、配置一致性。
**需要你提供**：选择哪一项。

---

## D6. Judgment 输入的商品元数据范围

**已确认事实**：候选池当前 metadata 只有 `title / price / main_category / brand`；
模块设计中的 Judgment 输入还包含 `description` 与属性（details）。当前 Prompt 因此缺少
描述/属性证据，Judgment 可能无法给出完整证据判断。

**可选方案**

| 方案 | 内容 |
|---|---|
| A | 扩展候选池 metadata：`description`（截断）+ 关键 `details` 属性（受 token 预算限制） |
| B | Judgment 层按 `product_id` 从 Catalog 查询完整记录（候选池不扩展） |
| C | 维持现状，Judgment 只基于 title/price/category/brand 判断，并在输出标注证据受限 |

**影响**：Judgment 质量、Prompt token 成本、候选池契约。
**需要你提供**：选择哪一项。

---

## D7. Planner 失败时的降级语义

**已确认事实**：模块二 §8 写有“多次失败降级到默认均衡召回计划”；Phase 5 当前实现为
**结构化失败**（不伪造计划，抛出 `PlanGenerationError`），与“不得把 API 失败伪装成成功”一致，
但与“默认均衡计划”的表述不一致。

**可选方案**

| 方案 | 内容 |
|---|---|
| A | 维持结构化失败（不生成默认计划） |
| B | 生成显式标记 `degraded=true` 的默认均衡计划（来源 system_default），并写入 assumptions |
| C | 二者可配置（默认结构化失败，允许显式开启降级计划） |

**影响**：可用性 vs 伪造风险；需要同步修改模块二文档或 ADR。
**需要你提供**：选择哪一项。

---

## D8. 质量 STOP 是否必须有 Judgment

**已确认事实**：Phase 5 当前要求至少 1 条 Judgment 结果才允许 `quality_satisfied` 停止；
没有 Judgment 时只能按预算/无候选/降级等理由 STOP。模块文档未明确该门槛。

**可选方案**

| 方案 | 内容 |
|---|---|
| A | 维持“至少 1 条 Judgment” |
| B | 允许仅凭 Reranker + 约束结果 STOP（Judgment 可选） |
| C | 阈值可配置（例如 `min_judgments`） |

**影响**：Controller 行为、评测口径、API 成本。
**需要你提供**：选择哪一项。

---

## D9. Phase 6：Qdrant 部署与 Mem0 Embedding

**已确认事实**

- Q7 已确认 Mem0 + 独立 Qdrant + 本地 Embedding；当前服务器没有 Qdrant 服务，
  `dp` 容器也无 `mem0ai/qdrant-client`。
- 宿主是共享服务器；新增长期运行容器需要端口（默认 6333）与持久化卷（NFS 可用）。
- Mem0 embedder checkpoint 尚未指定；本地已有 `intfloat/e5-base-v2`（768 维）。

**可选方案**

| 维度 | 选项 |
|---|---|
| Qdrant | 独立容器（持久卷）/ 嵌入式 Qdrant / 暂缓到 Phase 7 |
| Embedder | 复用 `intfloat/e5-base-v2` / 其他本地模型 / Mem0 默认 embedder |
| 数据卷 | NFS `/mnt/agr-data/qdrant` / 其他路径 |

**影响**：Phase 6 依赖、运维、用户隔离测试；是否允许在共享服务器上常驻服务。
**需要你提供**：三个维度的选择。

---

## D10. Phase 7：E5 评测 Query 构造方式

**已确认事实**：TIGER 的 next-item 评测已按时间切分完成；E5 需要“用户 Query”，但
Amazon Reviews 是评论/评分数据，没有真实购物 Query。模块一提出用 Persona + 模拟对话；
模拟数据必须标记 `is_synthetic`，不能混入真实指标。

**可选方案**

| 方案 | 内容 | 风险 |
|---|---|---|
| A | 基于截点前历史构建 Persona + 模拟购物 Query（标记 synthetic） | 模拟 Query 偏差；需单独评测口径 |
| B | 模板化 Query（品类/属性/价格组合） | 与真实需求距离较远 |
| C | 人工标注小规模 Query 集（例如 100–300 条） | 人时成本 |
| D | 只用 TIGER next-item 协议 + E5 用户画像（无自然语言 Query） | 不能测 Planner/E5 文本链路 |

**影响**：Phase 7 的 E5/双路/端到端指标是否可信、是否可复现。
**需要你提供**：选择哪一项（可组合，如 C + A）。

---

## D11. TIGER Pilot→全量训练的量化门槛

**已确认事实**：计划 §2.5 要求“Pilot 达标才进入全量训练”，但未给具体数值；
A1 当前 Recall@10 0.15%。没有门槛就无法客观判断 A2 是否“达标”。

**可选方案**

| 方案 | 门槛定义 |
|---|---|
| A | 必须超过热度基线（当前 Recall@10 > 0.34%） |
| B | 必须超过 E5 用户画像基线（当前 1.6%，但仅 500 样本参考） |
| C | 负责人指定绝对阈值（例如 Recall@10 ≥ 1% / 2% / 5%） |
| D | 不设数值门槛，Pilot 报告由负责人主观评审 |

**影响**：Phase 4 是否继续、全量训练的启动条件、后续实验预算。
**需要你提供**：选择哪一项；若选 C，请给出具体数值。

---

## 附：已处理、无需你决定的事项（供核对）

| 项 | 处理 |
|---|---|
| R17 `deepseek-flash` 推理截断 | 已加 `deepseek.reasoning_effort: none` 配置（`configs/models.yaml`、`config.py`）；Phase 5 客户端支持注入与回退 |
| Q3 无约束状态 | 已实现为 `null` + `not_configured_count`，单测覆盖 |
| Q8 Git | 独立仓库，服务器已同步 git 历史；凭据未入库 |
| Q9 启动提示词 | 根目录 V1.2 唯一权威，旧版已归档 |
| Q10 存储 | 数据/缓存/产物/大日志在 NFS；代码与文档在 Git |
| Q11 MUSA | 未接入；RQ-VAE/生成器/E5/Reranker 全部 CPU |
| Cornac 外部实现 | 已审计（Apache-2.0，tag `bb3b42d`），仅作校验/回退，不新增运行时依赖 |
