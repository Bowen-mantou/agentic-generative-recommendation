# 设计契约审计：六份 V1.1 设计文档的字段级提取

> 审计对象：`AGENTS.md` + `docs/` 下六份 V1.1 设计文档
> 审计目的：为后继「Pydantic Schema 落地」与「分模块实现」提供**可逐字段引用**的契约基线
> 产出文件：本文件（`docs/audit/design-contracts.md`），不修改任何其他文件
> 审计日期：2026-11-24

---

## 0. 审计范围、引用约定与结论摘要

### 0.1 审计对象与版本

| 编号 | 文件 | 文档内版本声明 | 总行数 |
|---|---|---|---|
| A | `AGENTS.md`（项目根） | V1.1（与六份技术设计文档统一） `A:L3` | 282 |
| 总 | `docs/项目总体方案.md` | V1.1（统一技术路线修订版） `总:L8` | 264 |
| 一 | `docs/模块一：User Memory 用户记忆系统.md` | V1.1（统一数据边界修订版） `一:L8` | 763 |
| 二 | `docs/模块二：Recommendation Planner 推荐规划器.md` | V1.1（统一 Plan Schema 修订版） `二:L9` | 890 |
| 三 | `docs/模块三：Candidate Retrieval 商品候选召回系统.md` | V1.1（统一数据与融合协议修订版） `三:L9` | 1056 |
| 四 | `docs/Reranker 模型选型与设计.md` | V1.1（跨模块职责修订版） `四:L5` | 304 |
| 五 | `docs/最终阶段：Judgment、Adaptive Controller、Recommendation Generator 与 Memory Update.md` | V1.1（统一接口与记忆边界修订版） `五:L7` | 907 |

辅助参考（非六份技术基线，仅用于交叉校验）：`docs/STARTUP_PROMPT.md`（V1.1，242 行）、根目录 `STARTUP_PROMPT.md`（旧版，295 行）、`README_替换说明.md`（28 行）。

### 0.2 引用与标注约定

- 行号引用格式：`二:L604` = 模块二文档第 604 行；`A:L123` = AGENTS.md 第 123 行；`总:L149`；`五:L380`。
- 字段来源标注（写在「出处」列的方括号内）：
  - `[D]` 文档**正式定义**（正文或"字段说明"表明确写了该字段/枚举/约束）；
  - `[E]` 仅出现在**示例 JSON/示例代码**中，文档未以正式定义形式声明；
  - `[I]` 由两处以上原文**组合推导**得出（本审计的推断，落地前需确认）；
  - `[?]` **文档未定义**（本审计明确标注缺口，禁止实现时凭印象补齐）。
- 「必填/默认」列：`必填` 表示文档要求该字段必须存在；`选填(null)` 表示文档示例中显式给了 `null`；`?` 表示文档未说明。
- 类型列使用 Python typing 记法（`str`、`int`、`float`、`bool`、`list[str]`、`Literal[...]`、`dict[str, X]`、`datetime`、`T | None`），**这是给 Pydantic 建模用的目标类型**，不是文档原文字面。

### 0.3 结论摘要（先看这 8 条）

1. **唯一计划协议已锁定**：Initial/Replanner 共用 `RecommendationPlan`，顶层 13 个字段在 `A:L104-112`、`二:L604-652`、`二:L664-680` 三处一致，可直接落 Pydantic；但模块二 4.3 节的 `current_preferences/historical_preferences` 与 6.2 节的 `current/historical` 冲突（`二:L294-309` vs `二:L629-634`）。
2. **候选池 5.6 节只有 9 个字段是正式定义**（`三:L764-784`），`fusion_score/constraint_status/source_ranks/raw_scores/product_metadata` 的子结构全部靠示例推断；`unverified_candidates` 元素结构文档未定义（`三:L783`）。
3. **硬约束状态出现第 4 种值 `not_required`**（`五:L211`），与 `A:L118-123`、`三:L748-752` 的三态 `pass/fail/unknown` 不一致，且 `constraint_status` 在候选池里是 `str`（`三:L772`）、在 Judgment 输入里是 `dict`（`五:L170-173`）——同一字段名两种类型，是**当前最容易写错的一处契约**。
4. **Controller 的示意代码引用了 State 里不存在的字段**：代码用 `state["hard_constraints_satisfied"]`、`state["quality_sufficient"]`（`五:L400-404`），但 `RecommendationState` 字段清单只有 `constraint_summary/valid_product_count/required_count/quality_metrics`（`五:L469-472`），无上述两键；`action` 也声明为裸 `str`（`五:L475`）而未给枚举。
5. **RRF 的 k 值全六份文档无任何数值**：公式见 `三:L691-694`、E5 内部 RRF 见 `三:L502`、跨路 RRF 见 `三:L717`，但 k 只被描述为"平滑常数"（`三:L701`），`A:L234` 又要求"RRF 常数配置化"——属于必须由负责人拍板的缺口。
6. **RQ-VAE 层数 / codebook 数 / 码本大小、E5 checkpoint / 前缀 / 维度、FAISS 索引类型、DeepSeek 模型 ID 与超时并发** 全部为"文档未定义"（`三:L164`、`三:L194`、`三:L505`、`三:L520`、`A:L272-276`），实现前必须补配置与 ADR，不能按经验硬编码。
7. **Reranker 加载方式自相矛盾**：`四:L59` 说"通过 yes/no 对应 Token 的分数计算相关性"（生成式判分），`四:L198-228` 却给出 `sentence_transformers.CrossEncoder(...)` 代码（且带 `prompts`/`default_prompt_name` 参数）；`A:L273` 把"推理接口、输入长度和硬件开销"列为未验证事项。
8. **Judgment/Generator 的输出 Schema 都不含 trace 必需字段**：`五:L231-264`、`五:L560-581` 无 `schema_version/request_id/trace_id`，而 `A:L240` 要求所有外部调用带 `request_id/trace_id`；`五:L380` 要求预算耗尽时"明确说明未满足或未验证的需求"，但 Generator 输出结构里没有承载该说明的系统级字段。

---

## 1. 每模块 I/O 与边界表

> 说明：本节回答"这个模块吃什么、吐什么、不许干什么"。字段级细节见第 3 节。

### 1.1 User Memory（模块一）

| 维度 | 内容 | 出处 |
|---|---|---|
| 输入 | ① Amazon Reviews 2023（Beauty and Personal Care）：user_id、ASIN、rating、review_title、review_text、时间、商品标题/类别/属性/描述 ② DeepSeek 模拟购物对话（Persona 驱动，五类场景） ③ 用户当前购物需求（用于 Memory Retrieval） | `一:L104-117`、`一:L253-261`、`一:L602-621` |
| 输出 | ① `MemoryRecord`（Mem0 语义记忆，含 memory_type/content/confidence/source/status） ② `InteractionEvent`（SQLite 结构化事件，供 TIGER） ③ `UserContext`（long_term_preferences / short_term_interests / relevant_episodes / current_request） | `一:L415-444`、`一:L469-480`、`一:L648-672` |
| 存储分工 | Mem0+Qdrant 管语义记忆；SQLite 管原始评论证据、来源映射、按时间排序的事件、审计信息；LangGraph Checkpointer 只管工作流状态 | `一:L369-373` |
| 职责边界（可做） | 偏好提取、Persona 构建、记忆写入/更新/冲突处理、记忆检索、User Context 组装 | `一:L344-350`、`一:L556-598`、`一:L637-675` |
| 明确禁止 | ① 不得独立训练 SASRec / 兴趣融合模型 ② 不得把 Mem0 语义记忆当 TIGER 行为序列 ③ 不得把评论/评分伪装成 click/browse/purchase ④ 模拟数据不得混入真实统计，必须 `is_synthetic=true` ⑤ 不得直接把 LLM 自报置信度当记忆置信度 ⑥ 不得把系统生成的推荐理由写入长期记忆 | `一:L18`、`一:L371`、`一:L149`、`一:L330`、`一:L461`、`五:L629` |

### 1.2 Interaction History（跨模块一/二/三，无独立文档章节）

| 维度 | 内容 | 出处 |
|---|---|---|
| 输入 | Amazon Reviews 的 review/rating 代理事件 + 后续真实 click/favorite/purchase + 用户明确反馈事件 | `一:L486` |
| 输出 | 按 `timestamp` 升序排列的、以内部 `product_id` 标识的事件序列 | `一:L487`、`三:L214-217` |
| 消费方 | TIGER（唯一行为序列来源）；Planner 只可读"统计摘要" | `三:L29`、`一:L465`、`二:L129` |
| 明确禁止 | 不得从 Mem0 自由文本拼接序列；模拟事件默认不进入真实评测序列；不得缺失 `is_synthetic` 标记 | `A:L143`、`A:L98`、`一:L489` |

### 1.3 Planner / Replanner（模块二）

| 维度 | 内容 | 出处 |
|---|---|---|
| 输入（4 类） | Current Request（User Input）、User Memory（Mem0）、Session History（LangGraph State）、System Capabilities（System Config） | `二:L131-136` |
| 输入（Replanner 追加） | original_request、previous_plan、execution_feedback（retrieved_candidates/after_deduplication/constraint_satisfied/judgment_passed/missing_attributes/diversity_issue）、controller_feedback（reason/suggested_direction）、budget_state | `二:L464-504` |
| 输出 | 唯一 `RecommendationPlan`（13 个顶层字段，见 3.1） | `二:L604-652`、`A:L104-112` |
| 模式差异 | 仅 `mode=initial/replan` 与 Prompt/输入模板不同；模型、API Client、输出 Schema 完全相同；Replanner 输出**完整新计划**而非 patch | `二:L42-44`、`二:L566-568`、`二:L415-417` |
| 职责边界（可做） | 理解需求、融合记忆、抽硬约束/软偏好、选召回策略与预算档位、生成 E5 Query、生成精排 Query、判断是否澄清、根据反馈调整计划 | `二:L430-440`、`二:L448-456` |
| 明确禁止 | ① 不得调用 FAISS/TIGER ② 不得生成或虚构商品 ID / Semantic ID ③ 不得改写原始硬约束（除非用户明确改需求） ④ 不得自行增加预算 ⑤ 不得输出顶层 `retrieval_strategy`/`tiger`/`e5_faiss` 平行字段 ⑥ 不决定 STOP/REPLAN（Controller 决定） | `A:L139`、`二:L362`、`二:L570`、`二:L662`、`A:L114`、`二:L723-729` |

### 1.4 TIGER 生成式召回（模块三 §3）

| 维度 | 内容 | 出处 |
|---|---|---|
| 输入 | ① 商品库（离线构建 Semantic ID） ② SQLite `InteractionHistory` 按时间排序的 `product_id` 序列 | `三:L343-351`、`三:L381` |
| 中间表示 | RQ-VAE 多级离散码 → Semantic ID（`[12, 05, 31]` 仅示意）→ 映射表 Semantic ID ↔ product_id | `三:L166-204` |
| 输出 | `{retrieval_source:"TIGER", candidates:[{product_id, semantic_id, generation_score, rank}]}` | `三:L355-373` |
| 阶段 B（第一阶段必做） | 商品预处理→预训练 Embedding→（现有实现或轻量）RQ-VAE→用户 Semantic ID 序列→轻量自回归 Next-Item→Semantic ID 解码→Top-K 商品 | `三:L307-315` |
| 阶段 A（后置） | RQ-VAE 训练优化、码本实验、Beam Search、Semantic ID 长度/码本消融、长尾与冷启动 | `三:L327-339` |
| 明确禁止 | ① 不得接 Planner 的自然语言 Query 作为输入 ② 不得从 Mem0 文本构造序列 ③ Planner 不得替 TIGER 生成 Semantic ID ④ Mock 只能联调，不能进正式指标 ⑤ 无历史序列时不得伪造序列 | `三:L381-385`、`A:L143`、`三:L325`、`三:L837` |

### 1.5 E5 + FAISS（模块三 §4）

| 维度 | 内容 | 出处 |
|---|---|---|
| 输入 | Planner 生成的 1..N 条自然语言 Query；离线：商品目录文本（Title/Category/Brand/Description/Features） | `三:L416-440`、`三:L466-478` |
| 处理 | 商品向量离线一次编码入库；Query 在线编码；余弦配置需双向 L2 归一化 + 内积；多 Query 各自 Top-K → 按 product_id 去重 → E5 内部 RRF → 唯一 E5 排名 | `三:L444-460`、`三:L498-505`、`三:L513-518` |
| 输出 | `{retrieval_source:"E5_FAISS", candidates:[{product_id, similarity_score, rank}]}` | `三:L535-551` |
| 职责边界 | 主要承担自然语言需求 + Mem0 偏好语义化；历史偏好只作检索/排序信号 | `三:L553-593` |
| 明确禁止 | ① 不微调 Encoder（第一阶段） ② 不得把多 Query 原始相似度直接相加 ③ 不得因向量相似度高就推断缺失属性已满足 ④ 不得用 E5 分数代替硬约束 | `A:L73`、`三:L498`、`三:L593`、`三:L591` |

### 1.6 Candidate Fusion + Constraint Evaluator（模块三 §5）

| 维度 | 内容 | 出处 |
|---|---|---|
| 输入 | TIGER 候选（含 Semantic ID）+ E5 候选 + Planner Plan 的 `constraints.hard` + Product Catalog | `三:L601-608`、`三:L730` |
| 处理顺序 | ID 统一映射 → 去重（保留来源与原始排名）→ 两级 RRF → 硬约束过滤（唯一 Python Evaluator）→ 统一候选池 | `三:L610-624`、`三:L626-675`、`三:L677-726`、`三:L728-758` |
| 输出 | `CandidatePool`：`request_id` + `candidates[]` + `unverified_candidates[]`（见 3.6） | `三:L764-784` |
| 硬约束判定 | `pass`→计入有效候选；`fail`→剔除并留诊断；`unknown`→不得当 pass，进 `unverified_candidates`；只有全部必需硬约束 pass 才计入 `valid_product_count` | `三:L748-754`、`A:L118-123` |
| 明确禁止 | ① 不得生成推荐理由 ② 不得做最终精排 ③ 不得把软偏好当硬过滤 ④ 不得在别处重复实现价格/品类比较 ⑤ 不得混加原始分数 | `三:L789`、`三:L756-758`、`三:L754`、`A:L129` |

### 1.7 Reranker（文档四）

| 维度 | 内容 | 出处 |
|---|---|---|
| 输入 | Instruction（固定任务指令）+ Query（Planner 结构化需求拼装）+ Document（统一商品文本）；原则上只接收硬约束 `pass` 的商品 | `四:L109-176`、`四:L147` |
| 输出 | 每个 Query-Product Pair 一个**原始相关性分数**；按分降序；`product_id`、原始分数、排名、召回来源写入 `RankedProduct` | `四:L182-194` |
| 模型 | 主：Qwen3-Reranker-0.6B；对照：BAAI/bge-reranker-v2-m3；轻量备选：BAAI/bge-reranker-base(278M) | `四:L88-92` |
| 职责边界 | 只在已过硬约束的候选内做相关性排序；批量构建 pair 降低开销 | `四:L22`、`四:L230` |
| 明确禁止 | ① 不得生成计划/召回/硬约束过滤/最终推荐决策 ② 不得替代硬约束校验 ③ 不得替代 Judgment ④ 不得把分数解释为购买概率 ⑤ 不得为缺失属性补写文本 ⑥ 第一阶段不得微调 | `四:L22`、`四:L147`、`四:L290`、`四:L194`、`四:L180`、`四:L234` |

### 1.8 Judgment（文档五 §3）

| 维度 | 内容 | 出处 |
|---|---|---|
| 输入（4 类） | Current Request、User Preferences、Recommendation Plan（需求/硬约束/优先级）、Candidate Product（含 constraint_status） | `五:L131-138`、`五:L142-176` |
| 粒度 | 单商品独立判断；可同批请求，但每商品必须有独立结果 | `五:L178-188` |
| 输出 | `product_id` + `preference_fit{level,matched,unmatched,unknown}` + `attribute_evidence{attr:{status,source}}` + `match_assessment` + `uncertainty` + `warnings` | `五:L230-264` |
| 职责边界 | 只输出商品级结构化判断；消费上游硬约束状态，不重复校验数值 | `五:L269-280`、`五:L267` |
| 明确禁止 | ① 不生成检索 Query ② 不重新召回 ③ 不修改 Reranker 排序 ④ 不决定 REPLAN/STOP ⑤ 不得为商品补写不存在属性 | `五:L273-278`、`五:L267` |

### 1.9 Adaptive Controller（文档五 §4）

| 维度 | 内容 | 出处 |
|---|---|---|
| 输入 | 拉链式读取 State：约束状态、Judgment 结果、有效候选数、多样性、轮数、BudgetState | `A:L163`、`五:L303-310` |
| 输出 | 动作 `REPLAN` / `STOP`（细化：`STOP_BUDGET_EXHAUSTED` / `STOP_QUALITY_SATISFIED`）+ `stop_reason` + `quality_satisfied` + `budget_exhausted` | `五:L390-409` |
| 实现 | LangGraph 编排 + Python 判断；不引入 Controller LLM | `五:L292-299` |
| 预算 | 多维预算：remaining_replan_rounds / remaining_retrieval_calls / tiger_candidate_budget / e5_candidate_budget / remaining_api_calls | `五:L423-437` |
| 明确禁止 | ① 不得引入第二个 LLM ② 不得重复实现硬约束 ③ 预算耗尽不得报成质量达标 ④ 不得用单一 `remaining_budget:int` | `五:L299`、`五:L297`、`五:L380`、`五:L425` |

### 1.10 Recommendation Generator（文档五 §5）

| 维度 | 内容 | 出处 |
|---|---|---|
| 输入 | user_request、selected_products[{product_id,rank,title,price,matched_preferences,warnings}] + User Memory + Plan + Reranker 结果 + Judgment 结果 + 商品原始元数据 | `五:L519-554` |
| 输出 | `{recommendations:[{rank, product_id, title, price, reasons[], limitations[]}]}` | `五:L556-581` |
| 展示 | Top-K 商品、名称价格、核心属性、理由、偏好匹配情况、必要限制 | `五:L583-590` |
| 明确禁止 | ① 只能用 Controller 确认的候选 ② 理由须可追溯到商品原始信息或 Judgment ③ 不得编造属性/价格/功能 ④ 不得把未知写成已知 ⑤ 不得擅自改 Reranker 顺序 ⑥ 有效候选不足时允许少于 K | `五:L594-601`、`A:L167` |

### 1.11 Memory Update（文档五 §6）

| 维度 | 内容 | 出处 |
|---|---|---|
| 触发 | 只有推荐后的**有效用户反馈/行为**才触发，不在内部循环每轮触发 | `五:L758`、`A:L171` |
| 输出结构 | `memory_updates[]`：{type, action, content, source, target_store} 与 {type=interaction, action, event{event_type,product_id,content,source,is_synthetic}, target_store} | `五:L647-670` |
| 写入去向 | 语义偏好 → Mem0；结构化商品事件 → SQLite `InteractionEvent`；通过 user_id / product_id / source_id 关联但分离存储 | `五:L673` |
| 明确禁止 | ① 无反馈不得推断用户喜欢推荐商品、不得写长期偏好 ② 没点击≠不喜欢 ③ 模型推断未经确认不得变用户事实 ④ 模拟反馈必须 `is_synthetic=true` 并与真实隔离 | `五:L675-684`、`五:L758`、`A:L171` |

### 1.12 Trace / Observability（无独立章节，分散定义）

| 维度 | 内容 | 出处 |
|---|---|---|
| Planner 轨迹（12 字段） | request_id、plan_id、parent_plan_id、mode、input_context、output_plan、validation_result、replan_reason、model_name、latency、token_usage、timestamp | `二:L761-776` |
| 每轮执行结果 | TIGER 召回数量、E5 召回数量、候选合并数量、Reranker 排序结果、Judgment 结果、Controller 动作、最终推荐结果 | `二:L778-786` |
| 存储 | JSONL + SQLite | `A:L80`、`总:L139` |
| 全局要求 | request_id / trace_id、Token/延迟日志、敏感信息脱敏、API 失败不得伪装 | `A:L240` |
| 缺口 | 检索/精排/Judgment/Controller/Generator 的 trace 字段级定义均未给出（见 4.3） | `[?]` |

---

## 2. 跨模块数据契约

### 2.1 `product_id` 映射契约

| 项目 | 内容 | 出处 |
|---|---|---|
| 唯一主键 | `product_id`，跨 Planner/E5/Fusion/Reranker/Judgment/Generator 一律使用 | `A:L90`、`总:L149`、`一:L147`、`三:L614-616` |
| 原始字段 | Amazon `parent_asin` 保留在**数据接入层**，并映射为 `product_id` | `A:L90`、`三:L614` |
| Semantic ID | 仅 TIGER 内部生成与解码使用，**不是**跨模块主键 | `A:L90`、`三:L616` |
| 映射表 | 双向：Semantic ID → product_id（解码）；product_id → 商品元数据（Catalog） | `三:L196-204`、`三:L612`、`三:L787` |
| 未定义/风险 | ① `parent_asin`→`product_id` 的基数（1:1 还是 1:N）未定义 ② 用户行为表只有 `parent_asin` 与 `product_id`，无子 ASIN 字段，但 §3.1 数据来源写"商品 ID（ASIN）" ③ 映射函数签名/冲突处理未定义 | `一:L109` vs `一:L126-127`；`A:L271` |

### 2.2 候选协议字段（三路形态对照）

| 阶段 | 字段集合 | 出处 | 备注 |
|---|---|---|---|
| TIGER 原始 | `retrieval_source`, `candidates[].{product_id, semantic_id, generation_score, rank}` | `三:L355-373` | 含 Semantic ID（内部） |
| E5 原始 | `retrieval_source`, `candidates[].{product_id, similarity_score, rank}` | `三:L535-551` | 无 semantic_id |
| 去重中间态 | `product_id`, `retrieval_sources[]`, `source_ranks{tiger,e5}` | `三:L661-673` | 注意 `source_ranks` 键是小写 `tiger`/`e5`，而 `retrieval_sources` 值是 `"TIGER"`/`"E5_FAISS"`——大小写不一致，需在 Schema 中固定 |
| 融合候选池 | `product_id`, `fusion_score`, `constraint_status`, `retrieval_sources[]`, `source_ranks{tiger,e5}`, `raw_scores{tiger,e5}`, `product_metadata{title,price,category}` | `三:L764-784` | `unverified_candidates` 元素结构未定义 |
| 精排结果 | `product_id`, 原始分数, 排名, 召回来源（`RankedProduct`，字段未逐一定义） | `四:L192` | 分数键名 `[?]`（示例仅文本 `Product P001: 5.82`） |
| Judgment 输出 | `product_id`, `preference_fit`, `attribute_evidence`, `match_assessment`, `uncertainty`, `warnings` | `五:L231-264` | 无 `rank`/`score`，与 `RankedProduct` 靠 product_id 连接 |

### 2.3 分数语义（必须写进字段 docstring，禁止混用）

| 分数 | 产生者 | 语义 | 取值范围 | 用途 | 禁止解释 | 出处 |
|---|---|---|---|---|---|---|
| `generation_score` | TIGER 自回归解码 | 生成路径的模型分数 | 文档未定义 | 仅作诊断/TIGER 内部排名 | 不得与 E5 相似度相加 | `三:L362`、`三:L375`、`A:L129` |
| `similarity_score` | E5 + FAISS | 向量相似度（余弦配置下为归一化内积） | 文档未给上界声明 | E5 路径内部排名 | 不等同"属性满足" | `三:L541`、`三:L513-518`、`三:L593` |
| `fusion_score` | 两级 RRF | 排名融合分 `Σ 1/(k+rank)` | 理论上 >0 且远小于 1（示例 0.0325） | 候选池排序 | 不是概率 | `三:L694`、`三:L772` |
| `rerank score`（键名未定） | Qwen3-Reranker | 原始相关性分数（yes/no logits 机制，未校准） | 未校准，可负（示例 -1.24） | 同 Query 内排序 | 不是购买概率 | `四:L187-194` |
| Judgment 等级 | DeepSeek | 语义判断等级（high/medium…） | 枚举 | 供 Controller 汇总 | 不是分数，不得用于排序 | `五:L234-259` |

### 2.4 硬约束状态契约

| 项目 | 内容 | 出处 |
|---|---|---|
| 状态全集（基线） | `pass` / `fail` / `unknown` | `A:L118-123`、`三:L748-752` |
| `pass` 语义 | 字段存在且满足约束 → 进入有效候选 | `三:L750` |
| `fail` 语义 | 明确违反 → 剔除，保留诊断记录 | `三:L751` |
| `unknown` 语义 | 缺字段/无法验证 → 不得当 pass；可留作诊断与 Replan | `三:L752`、`A:L123` |
| 计数规则 | 仅"全部必需硬约束 pass"的商品计入 `valid_product_count` | `A:L123`、`三:L754` |
| 冲突 | `not_required`（`五:L211`）不在三态内；`constraint_status` 类型在 `三:L772` 为 str、`五:L170` 为 dict | 见 4.2-C1/C2 |

### 2.5 预算契约

| 项目 | 内容 | 出处 |
|---|---|---|
| 档位（Plan 内） | `standard` / `expanded`，由 Python 配置映射为实际候选数量 | `二:L658`、`三:L827`、`A:L133` |
| 全局预算（Controller） | remaining_replan_rounds / remaining_retrieval_calls / tiger_candidate_budget / e5_candidate_budget / remaining_api_calls | `五:L427-434`、`A:L133` |
| 归属 | 预算不属于 Plan；Replanner 的 `budget_state` 仅供规划参考，不得自行加预算 | `二:L662`、`二:L566` |
| 强制停止 | 预算耗尽 = 强制 STOP，但 ≠ 质量达标；必须分别记录 `quality_satisfied` / `budget_exhausted` | `五:L380`、`A:L133` |
| 配置文件 | `configs/controller.yaml`（示例值 2/4/50/100/12 仅为示例，不是默认值） | `五:L437` |
| 缺口 | `tiger_candidate_budget`/`e5_candidate_budget` 是"每轮额度"还是"剩余额度"语义未定义；`standard`/`expanded` 到数量的映射表文档未给 | `五:L431-432`、`三:L827` |

### 2.6 Trace / ID 契约

| 项目 | 内容 | 出处 |
|---|---|---|
| `request_id` | 一次推荐任务 ID；候选池顶层携带 | `二:L765`、`三:L766` |
| `trace_id` | 仅 AGENTS 与最终阶段 State 要求，Plan/Trace 表未列 | `A:L240`、`五:L484` |
| `plan_id` / `parent_plan_id` / `plan_version` | 计划链：initial 时 parent=null、version=1；replan 指向上一版 | `二:L661`、`二:L609-613` |
| `schema_version` | Plan 输出协议版本（示例 "1.0"）；仅 Plan 有，其他 Schema 未定义 | `二:L610` |
| 缺口 | Judgment/Generator/Reranker/Retrieval 输出均无 request_id/schema_version；无法按 trace_id 串联 | 见 4.3 |

### 2.7 时间与 `is_synthetic` 契约

| 项目 | 内容 | 出处 |
|---|---|---|
| `is_synthetic` 存在处 | InteractionEvent（bool）、MemoryRecord（bool）、memory_updates[].event（bool） | `一:L479`、`一:L442`、`五:L665` |
| 隔离要求 | 模拟数据不得混入真实评测序列；TIGER 第一版默认只用真实 review/rating | `A:L98`、`一:L489`、`三:L245` |
| 缺口 | 字段类型不统一：InteractionEvent.timestamp 为 unix 整数（`一:L476`），MemoryRecord.created_at/last_observed_at 为 `"2026-11-23"` 字符串（`一:L439-440`）；无时区声明 | 见 4.2-C9 |
| 缺口 | Session/对话级 `is_synthetic` 未定义（模拟对话只有 memory 记录与 event 上有标记） | `一:L321-328`、`五:L665` |

---

## 3. 字段级 Schema 清单（Pydantic 落地的直接依据）

### 3.0 建模约定与本节的可靠性分级

| 约定 | 内容 |
|---|---|
| 权威性 | `A:L114`「Pydantic 是接口校验的权威定义」；`二:L682`「该 Schema 是 Planner、Retrieval、Reranker、Replanner 和 Controller 之间唯一计划协议；任何模块不得自行发明平行字段名」 |
| 未知字段策略 | 建议 `model_config = ConfigDict(extra="forbid")`，但**文档未规定**；采用前需负责人确认（见 4.1-B1） |
| 枚举策略 | 文档只给"仅允许 X/Y"的规则（`二:L656-658`），未给完整枚举表；所有 `Literal` 值域均以本文档摘录值 + 4.2 缺口清单为准 |
| 结构来源 | 顶层字段有正文"字段说明"表（`二:L664-680`）；**嵌套子结构只有示例 JSON，没有字段说明表**——这是本项目最主要的结构性缺口，Pydantic 落地时必须由负责人确认字段名与必填性（见 4.1-B2） |

### 3.1 `RecommendationPlan`（唯一计划协议，Initial 与 Replanner 共用）

#### 3.1.1 顶层字段（13）

| # | 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|---|
| 1 | `schema_version` | `str`（示例 `"1.0"`） | 必填，默认未定义 | 输出协议版本 | `[D]` `二:L610`、`二:L668`、`A:L104` |
| 2 | `plan_id` | `str`（示例 `plan_001`） | 必填 | 当前计划 ID | `[D]` `二:L611`、`二:L669`、`A:L105` |
| 3 | `parent_plan_id` | `str \| None` | 必填可空；initial 必须 `null`，replan 指向上一版 | 上一版计划 ID | `[D]` `二:L612`、`二:L661`、`二:L670` |
| 4 | `plan_version` | `int`（≥1） | 必填 | 计划版本 | `[D]` `二:L613`、`二:L671` |
| 5 | `mode` | `Literal["initial","replan"]` | 必填 | 工作模式，仅两值 | `[D]` `二:L614`、`二:L656`、`二:L672` |
| 6 | `intent` | `Intent`（3.1.2） | 必填 | 用户需求理解 | `[D]`（顶层）/`[E]`（子字段） `二:L615-620`、`二:L673` |
| 7 | `constraints` | `Constraints`（3.1.3） | 必填 | 硬约束 + 软偏好 | `[D]` `二:L621-628`、`二:L674` |
| 8 | `preferences` | `Preferences`（3.1.4） | 必填 | 四类偏好 + 优先级 | `[D]` `二:L629-634`、`二:L660`、`二:L675` |
| 9 | `retrieval` | `RetrievalPlan`（3.1.5） | 必填 | 召回策略 | `[D]` `二:L635-643`、`二:L676` |
| 10 | `ranking` | `RankingGuidance`（3.1.6） | 必填 | 精排目标 | `[D]` `二:L644-647`、`二:L677` |
| 11 | `clarification` | `Clarification`（3.1.7） | 必填 | 是否追问 | `[D]` `二:L648`、`二:L678` |
| 12 | `assumptions` | `list[str]` | 必填，默认 `[]` | 规划时使用的假设 | `[D]` `二:L649`、`二:L679` |
| 13 | `change_summary` | `list[str]` | 必填，默认 `[]`（initial 为空） | 本轮计划调整说明 | `[D]` `二:L650`、`二:L680` |

禁止的平行字段（会破坏契约）：顶层 `retrieval_strategy`、`tiger`、`e5_faiss`、`hard_constraints`、`soft_preferences`、`rerank_query` —— `[D]` `A:L114`、`二:L358`、`三:L825`。

#### 3.1.2 `Intent`

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `intent` | `str`（示例 `product_recommendation`） | 必填 | 意图类型；值域未定义 | `[E]` `二:L616`、`[?]` 枚举 |
| `category` | `str`（示例 `Beauty`） | 必填 | 商品大类 | `[E]` `二:L617` |
| `product_type` | `str`（示例 `moisturizer`） | 必填 | 商品细类 | `[E]` `二:L618` |
| `usage_scenario` | `str`（示例 `daily_use`） | 必填 | 使用场景 | `[E]` `二:L619` |

#### 3.1.3 `Constraints` / `HardConstraint` / `SoftPreference`

| 类 | 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|---|
| `Constraints` | `hard` | `list[HardConstraint]` | 必填 | 硬性约束，结构化对象 | `[D]` `二:L622-624`、`二:L659` |
| `Constraints` | `soft` | `list[SoftPreference]` | 必填 | 软性偏好 | `[D]` `二:L625-627`、`二:L659` |
| `HardConstraint` | `field` | `str`（示例 `price`/`category`） | 必填 | 商品字段名 | `[E]` `二:L623`、`二:L529` |
| `HardConstraint` | `operator` | `str`（示例 `lte`/`eq`） | 必填 | 比较算子；**枚举未定义** | `[E]` `二:L623`、`二:L529`、`[?]` 值域 |
| `HardConstraint` | `value` | `float \| int \| str` | 必填 | 约束值 | `[E]` `二:L623`、`二:L529` |
| `HardConstraint` | `unit` | `str`（示例 `USD`） | 选填（category 示例无 unit） | 单位 | `[E]` `二:L623`、`二:L528` |
| `HardConstraint` | `source` | `str`（示例 `current_request`） | 必填 | 约束来源 | `[E]` `二:L623`、`[?]` 值域（还应有 user_memory） |
| `SoftPreference` | `attribute` | `str`（示例 `lightweight`） | 必填 | 偏好属性 | `[E]` `二:L626`、`二:L532` |
| `SoftPreference` | `source` | `str`（示例 `current_request`/`user_memory`） | 必填 | 来源 | `[E]` `二:L626-627` |
| `SoftPreference` | `priority` | `str`（示例 `high`/`medium`） | 必填 | 优先级；**枚举未定义** | `[E]` `二:L626`、`[?]` 值域 |

> 注意：`二:L248-254` 用自然语言说明"Hard = 用户明确提出且必须满足"，`二:L284` 又说"硬性约束最终由 Python 规则执行，Planner 负责提取和表达约束"。所以 `HardConstraint` 只是**声明**，执行权在 Constraint Evaluator（`A:L118`）。

#### 3.1.4 `Preferences`

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `current` | `list[str]` | 必填 | 当前需求偏好 | `[D]` `二:L630`、`二:L660` |
| `historical` | `list[str]` | 必填 | 历史偏好（软信号） | `[D]` `二:L631`、`二:L660` |
| `negative` | `list[str]` | 必填 | 负向偏好 | `[D]` `二:L632`、`二:L660` |
| `priority` | `list[str]` | 必填 | 属性排序优先级（字符串数组） | `[D]` `二:L633`、`二:L660` |

> **冲突警示**：`二:L294-309`（4.3 节示例）使用 `current_preferences` / `historical_preferences` / `negative_preferences` / `priority`，与 6.2 节 `current` / `historical` / `negative` / `priority` 不一致。以 6.2 节与 `A:L108` 为准（见 4.2-C3）。

#### 3.1.5 `RetrievalPlan` / `TigerPlan` / `E5Plan`

| 类 | 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|---|
| `RetrievalPlan` | `strategy` | `Literal["hybrid_balanced","behavior_priority","semantic_priority"]` | 必填 | 召回策略，仅三值 | `[D]` `二:L636`、`二:L657`、`三:L795-801` |
| `RetrievalPlan` | `tiger` | `TigerPlan` | 必填 | TIGER 召回配置 | `[D]` `二:L637` |
| `RetrievalPlan` | `e5_faiss` | `E5Plan` | 必填 | E5 召回配置 | `[D]` `二:L638-642` |
| `TigerPlan` | `enabled` | `bool` | 必填 | 是否启用 TIGER | `[E]` `二:L637`、`三:L810` |
| `TigerPlan` | `candidate_budget` | `Literal["standard","expanded"]` | 必填 | 预算档位 | `[D]` `二:L658`、`[E]` `二:L637` |
| `E5Plan` | `enabled` | `bool` | 必填 | 是否启用 E5 | `[E]` `二:L639` |
| `E5Plan` | `queries` | `list[str]` | 必填 | 检索 Query；**数量与长度上限未定义** | `[E]` `二:L640`；`[?]` 阈值（`二:L742` 要求校验但没给数） |
| `E5Plan` | `candidate_budget` | `Literal["standard","expanded"]` | 必填 | 预算档位 | `[D]` `二:L658`、`[E]` `二:L641` |

策略 → 预算档位映射（`二:L328-332`、`三:L797-801`）：

| strategy | TIGER | E5+FAISS |
|---|---|---|
| `hybrid_balanced` | standard | standard |
| `behavior_priority` | expanded | standard |
| `semantic_priority` | standard | expanded |

> `candidate_budget` 到**实际候选数量**的映射表文档未给（`二:L334`、`二:L658`、`三:L827` 只说"由 Python 配置映射"）→ 见 5.2。

#### 3.1.6 `RankingGuidance`

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `rerank_query` | `str` | 必填 | 供 Reranker 的查询表达 | `[E]` `二:L646`、`二:L372` |
| `priority_attributes` | `list[str]` | 必填 | 重点属性，顺序即优先级 | `[E]` `二:L646`、`二:L374-379` |

#### 3.1.7 `Clarification`

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `required` | `bool` | 必填 | 是否需要追问 | `[E]` `二:L648`、`二:L395` 与 `二:L405` |
| `question` | `str \| None` | 必填可空（required=false 时 null） | 追问内容 | `[E]` `二:L396`、`二:L406` |
| `reason` | `str \| None` | 必填可空 | 追问原因 | `[E]` `二:L397`、`二:L407` |

> 追问节制规则：非必要信息缺失时不应频繁追问，应用合理默认值并记录到 `assumptions`（`[D]` `二:L411`）。

#### 3.1.8 Initial vs Replanner 差异（同 Schema、不同输入/约束）

| 维度 | Initial | Replanner | 出处 |
|---|---|---|---|
| 触发 | 首次提出需求 | Controller 判定需调整 | `二:L421`、`二:L444` |
| 模型/Client/Schema | 相同 | 相同 | `二:L415-417`、`二:L44` |
| `mode` | `"initial"` | `"replan"` | `二:L656` |
| `parent_plan_id` | `null` | 上一版 plan_id | `二:L661` |
| `plan_version` | 1（示例） | 递增（示例 2） | `二:L613`、`二:L518` |
| `change_summary` | `[]` | 列出改动（4 条示例） | `二:L650`、`二:L557-562` |
| 输入差异 | 4 类输入 | 4 类输入 + 执行反馈 + controller_feedback + budget_state | `二:L423-428` vs `二:L458-504` |
| 硬约束 | 提取 | **不得擅自修改**（除用户明确改需求） | `二:L570`、`二:L455` |
| 输出形态 | 完整计划 | **完整新计划**，不是 patch | `二:L568` |
| 预算 | 只选档位 | 只读参考，不得加预算 | `二:L662`、`二:L566` |

### 3.2 `ReplannerInput`（模块二 5.3，全部为示例字段）

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `mode` | `Literal["replan"]` | 必填 | 模式 | `[E]` `二:L466` |
| `original_request` | `OriginalRequest`（见下） | 必填 | 原始需求快照 | `[E]` `二:L468-472` |
| `previous_plan` | `dict` / `RecommendationPlan` | 必填 | 上一轮计划；示例只含 retrieval 片段 | `[E]` `二:L474-481`；`[?]` 完整度未定义 |
| `execution_feedback` | `ExecutionFeedback`（见下） | 必填 | 上一轮执行统计 | `[E]` `二:L483-492` |
| `controller_feedback` | `ControllerFeedback`（见下） | 必填 | 重规划原因与方向 | `[E]` `二:L494-497` |
| `budget_state` | `BudgetState`（见 3.9.1） | 必填 | 剩余预算（仅参考） | `[E]` `二:L499-503`、`二:L662` |

`OriginalRequest`（`[E]` `二:L469-472`）：`category:str`、`product_type:str`、`max_price:int|float`。**与 `Intent`+`HardConstraint` 结构不同，需确认是否复用 Plan 结构**（见 4.3-G1）。

`ExecutionFeedback`（`[E]` `二:L483-492`）：

| 字段 | 目标类型 | 语义 |
|---|---|---|
| `retrieved_candidates` | `int` | 召回总数（示例 100） |
| `after_deduplication` | `int` | 去重后数（示例 72） |
| `constraint_satisfied` | `int` | 满足硬约束数（示例 65） |
| `judgment_passed` | `int` | Judgment 通过数（示例 28）；"通过"判定标准未定义 |
| `missing_attributes` | `list[str]` | 缺证据属性（示例 `fragrance_free`） |
| `diversity_issue` | `bool` | 多样性是否不足 |

`ControllerFeedback`（`[E]` `二:L494-497`）：`reason:str`（示例 `INSUFFICIENT_ATTRIBUTE_EVIDENCE`）、`suggested_direction:str`（示例 `SEARCH_FOR_FRAGRANCE_FREE_PRODUCTS`）。**两个枚举值域均未定义**（`[?]`，见 4.4-T1）。

### 3.3 Plan 校验与降级（模块二 §8）

| 检查项 | 内容 | 出处 |
|---|---|---|
| 1 | JSON 格式合法 | `[D]` `二:L737` |
| 2 | 必填字段存在 | `[D]` `二:L738` |
| 3 | 字段类型正确 | `[D]` `二:L739` |
| 4 | 召回策略属于预定义枚举 | `[D]` `二:L740` |
| 5 | 召回预算在允许范围 | `[D]` `二:L741` |
| 6 | Query 数量与长度符合限制 | `[D]` `二:L742`（**限制数值未给**） |
| 7 | Replanner 保留原始硬约束 | `[D]` `二:L743` |
| 8 | 计划中模块在 System Capabilities 中可用 | `[D]` `二:L744` |
| 降级 | 多次失败 → 默认均衡召回计划 | `[D]` `二:L751` |
| 降级 | TIGER 不可用→保留 E5；E5 不可用→保留 TIGER | `[D]` `二:L752-753` |
| `PlanValidationResult` 结构 | 字段（is_valid/errors/...）**文档未定义** | `[?]`（Trace 只有 `validation_result` 一列，`二:L771`） |

`SystemCapabilities`（Planner 输入，`[E]` `二:L206-222`）：`retrieval_modules:list[str]`、`available_product_fields:list[str]`、`reranker_available:bool`、`judgment_available:bool`。**未进入 `RecommendationState` 字段清单**（见 4.3-G4）。

### 3.4 TIGER 输出 Schema（模块三 3.6）

#### 3.4.1 `TigerRetrievalResult`

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `retrieval_source` | `Literal["TIGER"]` | 必填 | 召回来源标识 | `[E]` `三:L357` |
| `candidates` | `list[TigerCandidate]` | 必填 | 候选列表 | `[E]` `三:L358` |

#### 3.4.2 `TigerCandidate`

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `product_id` | `str` | 必填 | 统一内部主键（解码后） | `[E]` `三:L360`、`A:L90` |
| `semantic_id` | `list[int]`（示例三层） | 必填（示例中每条都有） | TIGER 内部多级码；**层数与码本未定义** | `[E]` `三:L361`；`[?]` 层数/码本 |
| `generation_score` | `float` | 必填 | 解码产生的生成分数（非 LLM 造分） | `[E]` `三:L362`、`三:L375` |
| `rank` | `int`（1 起） | 必填 | TIGER 内排名 | `[E]` `三:L363` |

#### 3.4.3 TIGER 序列与映射（输入侧）

| 结构 | 形态 | 说明 | 出处 |
|---|---|---|---|
| 用户历史序列 | `list[SemanticId]`，`SemanticId = list[int]`（示例 3 层） | 从 SQLite 事件按 timestamp 排序后映射 | `[E]` `三:L236-243`、`三:L212-217` |
| 序列最大长度 | `[?]` **文档未定义** | 影响训练样本与生成条件 | `三:L251-271`（只有 2-3 步示例） |
| 商品映射表 | `product_id ↔ semantic_id` | 双向、供解码 | `[D]`（语义）`三:L196-204`、`三:L612` |

### 3.5 E5 召回 Schema（模块三 4.5）

| 类 | 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|---|
| `E5RetrievalResult` | `retrieval_source` | `Literal["E5_FAISS"]` | 必填 | 来源标识 | `[E]` `三:L537` |
| `E5RetrievalResult` | `candidates` | `list[E5Candidate]` | 必填 | 候选列表 | `[E]` `三:L538` |
| `E5Candidate` | `product_id` | `str` | 必填 | 内部主键 | `[E]` `三:L540` |
| `E5Candidate` | `similarity_score` | `float` | 必填 | 向量相似度（余弦配置下为归一化内积） | `[E]` `三:L541`、`三:L513-518` |
| `E5Candidate` | `rank` | `int` | 必填 | E5 内部（融合后）排名 | `[E]` `三:L542` |
| 中间态（未命名） | 每条 Query 的 `{product_id → rank}` 列表 | `[?]` 无正式类名 | E5 内部去重与 RRF 的输入 | `[I]` `三:L500-503` |

E5 内部 RRF 公式（文档唯一给出数值型公式之一，但**无 k 值**）：

```
RRF_E5(d) = Σ_q 1 / (k + rank_q(d))
```

出处 `[D]` `三:L502`；k 仅定义"平滑常数"`三:L701` —— **数值文档未定义**。

### 3.6 Candidate Fusion 候选池 Schema（模块三 5.6，最关键的跨模块契约）

#### 3.6.1 `CandidatePool`

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `request_id` | `str` | 必填 | 一次推荐任务 ID | `[E]` `三:L766` |
| `candidates` | `list[Candidate]` | 必填 | 通过硬约束的候选（pass 集） | `[E]` `三:L768`、`三:L750` |
| `unverified_candidates` | `list[?]` | 必填，默认 `[]` | `unknown` 集；**元素结构文档未定义** | `[E]` `三:L783`、`三:L752`；`[?]` 结构 |

#### 3.6.2 `Candidate`（5.6 节逐字段）

| # | 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|---|
| 1 | `product_id` | `str` | 必填 | 内部唯一主键 | `[E]` `三:L770` |
| 2 | `fusion_score` | `float` | 必填 | 两级 RRF 融合分（示例 0.0325，非概率） | `[E]` `三:L771`、`三:L694` |
| 3 | `constraint_status` | `Literal["pass","fail","unknown"]` | 必填（池内应恒为 `pass`） | 硬约束聚合状态 | `[E]` `三:L772`、`[D]` `三:L748-752`、`A:L118-123` |
| 4 | `retrieval_sources` | `list[Literal["TIGER","E5_FAISS"]]` | 必填 | 召回来源 | `[E]` `三:L773`、`三:L664-667` |
| 5 | `source_ranks` | `dict[Literal["tiger","e5"], int]` | 必填但键可缺（缺失路径不出现或记 0，未定义） | 各路原始排名 | `[E]` `三:L774`、`三:L668-671`、`三:L717` |
| 6 | `raw_scores` | `dict[Literal["tiger","e5"], float]` | 必填但键可缺 | 各路原始分数（`generation_score` / `similarity_score`） | `[E]` `三:L775` |
| 7 | `product_metadata` | `ProductMetadata`（3.6.3） | 必填 | 商品快照 | `[E]` `三:L776-780` |

#### 3.6.3 `ProductMetadata`（候选池内嵌，非完整 Catalog）

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `title` | `str` | 必填 | 商品名 | `[E]` `三:L777` |
| `price` | `float` | 必填 | 价格（货币统一性未定义） | `[E]` `三:L778`；`[?]` 币种 |
| `category` | `str` | 必填 | 类别 | `[E]` `三:L779` |

> 5.6 节只给了 3 个元数据字段；而 Reranker Document 还需要 `brand`/`attributes`/`description`（`四:L153-176`），说明候选池并非精排输入的完整来源——精排应从 Product Catalog 拉取（`三:L787`）。

#### 3.6.4 硬约束结果结构（**文档未定义字段名**）

文档只给了状态语义与判定规则（`三:L748-754`、`A:L118-123`），**没有**定义承载 per-field 结果的类。落地时必须新建，建议（需确认）：

| 建议字段 | 类型 | 依据 |
|---|---|---|
| `field` | `str` | 对应 `HardConstraint.field`（`二:L623`） |
| `status` | `Literal["pass","fail","unknown"]` | `A:L118-123` |
| `observed_value` | `float \| int \| str \| None` | "字段存在且满足/缺失"语义（`三:L750-752`） |
| `reason` | `str \| None` | "保留诊断记录"（`三:L751`） |

### 3.7 Reranker Schema（文档四 §5）

#### 3.7.1 输入模板（三段，均为固定文本模板）

| 段 | 内容 | 出处 |
|---|---|---|
| Instruction（4 句） | "Rank products according to the user's shopping requirements and preferences. / Prioritize explicit requirements and relevant product attributes. / Consider the user's stated preferences when comparing candidate products. / Do not assume missing product attributes are satisfied." | `[D]` `四:L113-125` |
| Query（3 段） | `User Intent:`、`Must-have Constraints:`（Category/Price）、`Preferences:`（3 条） | `[D]` `四:L133-145` |
| Document（7 段） | `Product ID:`、`Title:`、`Category:`、`Brand:`、`Price:`、`Attributes:`（列表）、`Description:` | `[D]` `四:L153-176` |

缺失属性必须显式保留未知，不得补写（`[D]` `四:L180`）。Query 中可保留约束作语义上下文，但不得用 Reranker 分数代替硬约束校验（`[D]` `四:L147`）。

拼接规则（`ranking.rerank_query` + `priority_attributes` + Plan 的 intent/constraints/preferences → 上述三段模板）**文档未给算法**（`[?]`，见 4.3-G3）。

#### 3.7.2 `RankedProduct`（文档仅点名，未定义字段）

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `product_id` | `str` | 必填 | 内部主键 | `[D]`（点名）`四:L192` |
| 分数（**键名未定**） | `float` | 必填 | 原始相关性分数，未校准，可负 | `[D]`（语义）`四:L184-194`；`[?]` 键名 |
| `rank` | `int` | 必填 | 精排排名（1 起） | `[D]` `四:L192` |
| `retrieval_sources` | `list[str]` | 必填 | 召回来源 | `[D]` `四:L192` |
| `raw_scores` | `dict[str,float]` | `[?]` 未说明是否继承候选池 | 上游原始分 | `[?]` |

#### 3.7.3 输出协议与其他限制

| 项目 | 内容 | 出处 |
|---|---|---|
| 分数机制（声明） | "通过 yes/no 对应 Token 的分数计算相关性，而不是传统单层分类头" | `[D]` `四:L59` |
| 工程接口（代码） | `sentence_transformers.CrossEncoder("Qwen/Qwen3-Reranker-0.6B", prompts={...}, default_prompt_name="shopping")` + `model.predict(pairs)` | `[E]` `四:L200-228` |
| 批量 | 批量构建 Query-Product Pair，减少逐条调用 | `[D]` `四:L230`；**batch size 未给**（`[?]`） |
| 长度限制 | **文档未定义**（max_length/truncate 策略缺失） | `[?]`；`A:L273` 列为未验证 |
| 超时/并发 | **文档未定义**；`A:L240` 要求并发上限 | `[?]` |
| 冻结策略 | 第一阶段冻结，不微调 | `[D]` `四:L234`、`A:L76` |

### 3.8 Judgment Schema（文档五 §3）

#### 3.8.1 `JudgmentInput`（仅示例，无正式字段表）

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `current_request` | `str` | 必填 | 用户当前需求原文 | `[E]` `五:L144` |
| `user_preferences` | `{positive:list[str], negative:list[str]}` | 必填 | 用户偏好 | `[E]` `五:L146-154` |
| `constraints` | `{max_price:float, category:str}` | 必填 | 约束摘要（**与 Plan 结构化约束不同构**） | `[E]` `五:L156-159`；`[?]` 结构冲突见 4.2-C1 |
| `candidate` | `JudgmentCandidate`（见下） | 必填 | 单商品 | `[E]` `五:L161-174` |

`JudgmentCandidate`：`product_id:str`、`title:str`、`price:float`、`description:str`、`attributes:list[str]`、`constraint_status:dict[str,str]`（示例 `{"price":"pass","category":"pass"}`）。出处 `[E]` `五:L162-173`。

#### 3.8.2 `JudgmentResult`（逐商品）

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `product_id` | `str` | 必填 | 内部主键 | `[E]` `五:L232` |
| `preference_fit.level` | `str`（示例 `high`） | 必填 | 偏好匹配等级；**枚举未定义** | `[E]` `五:L235`；`[?]` 值域 |
| `preference_fit.matched_preferences` | `list[str]` | 必填 | 已匹配偏好 | `[E]` `五:L236-239` |
| `preference_fit.unmatched_preferences` | `list[str]` | 必填，可空 | 未匹配偏好 | `[E]` `五:L240` |
| `preference_fit.unknown_preferences` | `list[str]` | 必填，可空 | 无法确认的偏好 | `[E]` `五:L241-243` |
| `attribute_evidence` | `dict[str, {status:str, source:str\|None}]` | 必填 | 属性证据（status 示例 `supported`/`unknown`；source 示例 `description`/null） | `[E]` `五:L246-255`；`[?]` 枚举 |
| `match_assessment` | `str`（示例 `partial_match`） | 必填 | 直接/部分/不匹配；**枚举未定义** | `[E]` `五:L257`、`五:L198`；`[?]` 值域 |
| `uncertainty` | `str`（示例 `medium`） | 必填 | 不确定性等级；**枚举未定义** | `[E]` `五:L259`；`[?]` 值域 |
| `warnings` | `list[str]` | 必填，可空 | 面向下游的告警 | `[E]` `五:L261-263` |
| 失败态字段 | `status` / `error`（承载 `judgment_unavailable`） | **文档未定义** | `五:L838` 要求"标记为 judgment_unavailable / unknown"，但输出 Schema 无对应字段 | `[?]` 见 4.3-G2 |

判断维度定义（`[D]` `五:L194-199`）：Preference Fit / Attribute Evidence / Match Assessment / Uncertainty 四维；硬约束不在此四维内（`[D]` `五:L201`、`五:L267`）。

### 3.9 Controller Schema（文档五 §4）

#### 3.9.1 `BudgetState`

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `remaining_replan_rounds` | `int` | 必填 | 剩余重规划轮数（示例 2） | `[E]` `五:L429`；`[D]` `A:L133` |
| `remaining_retrieval_calls` | `int` | 必填 | 剩余检索调用次数（示例 4） | `[E]` `五:L430` |
| `tiger_candidate_budget` | `int` | 必填 | TIGER 候选额度（示例 50）；"每轮/剩余"语义未定义 | `[E]` `五:L431`；`[?]` 语义 |
| `e5_candidate_budget` | `int` | 必填 | E5 候选额度（示例 100）；同上 | `[E]` `五:L432`；`[?]` 语义 |
| `remaining_api_calls` | `int` | 必填 | 剩余 API 调用次数（示例 12） | `[E]` `五:L433` |
| 禁用 | 单个 `remaining_budget:int` 被明确否定 | — | `[D]` `五:L425` | |

配置来源：`configs/controller.yaml`（`[D]` `五:L437`）；示例数字"仅为配置示例"（`[D]` `五:L437`）。

#### 3.9.2 `RecommendationState`（LangGraph State，26 字段，逐字提取）

| # | 字段 | 目标类型（文档声明） | 语义分组 | 出处 |
|---|---|---|---|---|
| 1 | `user_request` | `str` | 用户信息 | `[D]` `五:L449` |
| 2 | `user_memory` | `dict` | 用户信息 | `[D]` `五:L450` |
| 3 | `interaction_history` | `list` | 用户信息 | `[D]` `五:L451` |
| 4 | `session_history` | `list` | 用户信息 | `[D]` `五:L452` |
| 5 | `current_plan` | `dict` | Planner（应为 `RecommendationPlan`） | `[D]` `五:L455` |
| 6 | `tiger_candidates` | `list` | 召回 | `[D]` `五:L458` |
| 7 | `e5_candidates` | `list` | 召回 | `[D]` `五:L459` |
| 8 | `fused_candidates` | `list` | 召回 | `[D]` `五:L460` |
| 9 | `ranked_products` | `list` | 精排（`RankedProduct`） | `[D]` `五:L463` |
| 10 | `judgment_results` | `list` | Judgment | `[D]` `五:L466` |
| 11 | `constraint_summary` | `dict` | 约束汇总 | `[D]` `五:L469` |
| 12 | `valid_product_count` | `int` | 有效候选数（全硬约束 pass） | `[D]` `五:L470`；`[D]` `A:L123` |
| 13 | `required_count` | `int` | 所需候选数；**来源与计算未定义** | `[D]` `五:L471`；`[?]` |
| 14 | `quality_metrics` | `dict` | 质量指标；**内部键未定义** | `[D]` `五:L472`；`[?]` |
| 15 | `action` | `str` | 控制器动作；**枚举未定义** | `[D]` `五:L475`；`[?]` 值域 |
| 16 | `stop_reason` | `str` | 停止原因；**枚举未定义** | `[D]` `五:L476`；`[?]` |
| 17 | `quality_satisfied` | `bool` | 质量是否达标（与预算分离） | `[D]` `五:L477`、`五:L380` |
| 18 | `budget_exhausted` | `bool` | 预算是否耗尽（与质量分离） | `[D]` `五:L478`、`五:L380` |
| 19 | `budget_state` | `dict` | 多维预算（应为 `BudgetState`） | `[D]` `五:L479`、`五:L474` |
| 20 | `current_round` | `int` | 当前轮次 | `[D]` `五:L480` |
| 21 | `max_replan_rounds` | `int` | 最大重规划次数；**数值未定义** | `[D]` `五:L481`；`[?]` |
| 22 | `seen_query_hashes` | `list` | 重复 Query 检测；**哈希算法未定义** | `[D]` `五:L482`；`[?]` |
| 23 | `request_id` | `str` | 任务 ID | `[D]` `五:L483` |
| 24 | `trace_id` | `str` | 追踪 ID | `[D]` `五:L484` |
| 25 | `selected_products` | `list` | Controller 选定给 Generator 的商品 | `[D]` `五:L487` |
| 26 | `recommendation` | `dict` | Generator 输出 | `[D]` `五:L488` |

#### 3.9.3 Controller 决策（动作与停止原因）

| 项目 | 内容 | 出处 |
|---|---|---|
| 动作出口 | `REPLAN` / `STOP`（LangGraph 两条主路径） | `[D]` `五:L314`、`五:L493` |
| 细化返回值（示例代码） | `STOP_BUDGET_EXHAUSTED` / `REPLAN` / `STOP_QUALITY_SATISFIED` | `[E]` `五:L395-406` |
| 映射规则 | 两个 STOP 原因统一映射到 LangGraph `STOP`，但必须保留 `stop_reason` 与 `quality_satisfied` | `[D]` `五:L409` |
| STOP 质量条件 | 有效候选数足够、硬约束满足、核心偏好覆盖较好、无需继续消耗预算 | `[D]` `五:L373-378` |
| REPLAN 典型触发 | 核心偏好商品不足、重要属性缺证、召回与关键需求偏差、多样性不足、可能遗漏 | `[D]` `五:L320-326` |
| 判定代码引用字段 | `budget_state.remaining_replan_rounds`、`budget_state.remaining_retrieval_calls`、`valid_product_count`、`required_count`、`hard_constraints_satisfied`、`quality_sufficient` | `[E]` `五:L393-404`；**后两键不在 State 字段表内**（4.2-C5） |
| 防无限循环 | 最大 Replan 次数、最大检索预算、重复 Query 检测、每轮候选数量限制、API 失败处理 | `[D]` `五:L497-503` |
| 阈值来源 | "所有质量阈值通过验证集确定，不直接使用未经验证的固定阈值" | `[D]` `五:L421` |

### 3.10 Generator Schema（文档五 §5）

#### 3.10.1 `GeneratorInput`

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `user_request` | `str` | 必填 | 用户当前需求 | `[E]` `五:L534` |
| `selected_products` | `list[SelectedProduct]` | 必填 | 仅 Controller 确认的商品 | `[E]` `五:L536`；`[D]` `五:L596` |
| `SelectedProduct` | `{product_id:str, rank:int, title:str, price:float, matched_preferences:list[str], warnings:list[str]}` | 必填 | 单商品装箱 | `[E]` `五:L537-551` |
| 其他输入 | User Memory、Plan、Reranker 结果、Judgment 结果、商品原始元数据 | `[D]` `五:L521-528`（**未进入示例 JSON**） | 契约不完整 | `[?]` |

#### 3.10.2 `RecommendationOutput`

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `recommendations` | `list[RecommendationItem]` | 必填 | 允许长度 < K | `[E]` `五:L562`；`[D]` `五:L601` |
| `RecommendationItem.rank` | `int` | 必填 | 与 Reranker 顺序一致 | `[E]` `五:L564`、`[D]` `五:L600` |
| `RecommendationItem.product_id` | `str` | 必填 | 必须在候选池内 | `[E]` `五:L566`；`[D]` `五:L596` |
| `RecommendationItem.title` | `str` | 必填 | 商品名 | `[E]` `五:L566` |
| `RecommendationItem.price` | `float` | 必填 | 价格 | `[E]` `五:L567` |
| `RecommendationItem.reasons` | `list[str]` | 必填 | 理由，须有证据 | `[E]` `五:L569-573`；`[D]` `五:L597` |
| `RecommendationItem.limitations` | `list[str]` | 必填，可空 | 商品信息限制 | `[E]` `五:L575-577` |
| 系统级 caveat（预算耗尽/未验证需求） | **文档未定义字段** | — | `五:L380` 要求"明确说明未满足或未验证的需求"，但输出结构无系统级字段 | `[?]` 见 4.3-G5 |
| `request_id` / `schema_version` | **文档未定义** | — | `A:L240` 要求 trace 字段 | `[?]` |

Top-K 的 K 值：**文档未定义**（`五:L585` 只说"Top-K"；`总:L111`、`模块三:L315` 也只说 Top-K）→ 见 5.2。

### 3.11 Memory 与 InteractionEvent Schema（模块一 + 文档五 §6）

#### 3.11.1 `MemoryRecord`（模块一 §7，字段说明表为正式定义）

| # | 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|---|
| 1 | `memory_id` | `str` | 必填 | 记忆 ID | `[E]` `一:L417` |
| 2 | `user_id` | `str` | 必填 | 用户 ID | `[E]` `一:L418` |
| 3 | `memory_type` | `str`（示例 `long_term_preference`） | 必填 | 记忆类型；**枚举未定义** | `[D]`（有该字段）`一:L451`；`[E]` `一:L420`；`[?]` 值域 |
| 4 | `content` | `str` | 必填 | 记忆内容 | `[D]` `一:L452` |
| 5 | `category` | `str`（示例 `skincare`） | 必填（示例有） | 商品/兴趣类别 | `[E]` `一:L424` |
| 6 | `polarity` | `str`（示例 `positive`） | 必填（示例有） | 正/负向；**枚举未定义** | `[E]` `一:L425`；`[?]` |
| 7 | `confidence` | `float`（示例 0.88） | 必填 | 记忆可靠度 | `[D]` `一:L453`；计算规则见 `一:L461` |
| 8 | `source` | `{source_type:str, source_ids:list[str]}` | 必填 | 证据来源 | `[D]`（source/source_ids）`一:L454-455`；`[E]` `一:L429-435` |
| 9 | `evidence_count` | `int` | 必填 | 支撑证据数 | `[D]` `一:L456` |
| 10 | `created_at` | 日期字符串（示例 `"2026-11-23"`） | 必填 | 创建时间 | `[E]` `一:L439` |
| 11 | `last_observed_at` | 日期字符串（示例 `"2026-11-20"`） | 必填 | 最近观察时间 | `[D]` `一:L457`；`[E]` `一:L440` |
| 12 | `is_synthetic` | `bool` | 必填 | 是否模拟数据 | `[D]` `一:L458`；`[E]` `一:L442` |
| 13 | `status` | `str`（示例 `active`） | 必填 | 记忆状态；**枚举未定义** | `[D]` `一:L459`；`[?]` |

> `一:L461` 明确："置信度由证据质量、重复程度、明确表达程度和时间等因素综合计算，不能直接使用 LLM 自报的置信度" → 置信度算法文档未给（`[?]`，见 5.2）。

#### 3.11.2 `InteractionEvent`（模块一 §7.1，字段约束为正式定义）

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `event_id` | `str` | 必填（示例） | 事件 ID | `[E]` `一:L471` |
| `user_id` | `str` | 必填 | 用户 ID | `[E]` `一:L472` |
| `product_id` | `str` | 必填 | 内部统一主键 | `[D]` `一:L485`；`[E]` `一:L473` |
| `event_type` | `str` | 必填 | 事件类型；候选值：`review`/`rating`（Amazon）、`click`/`favorite`/`purchase`（后续真实）、`explicit_positive_feedback`/`explicit_negative_feedback`（用户反馈） | `[D]` `一:L486`；`[E]` `一:L474` |
| `rating` | `int` | 视事件类型（review/rating 必有） | 评分 | `[E]` `一:L475` |
| `timestamp` | `int`（unix，示例 1795300000） | 必填 | 行为时间（用于排序） | `[D]` `一:L487`；`[E]` `一:L476` |
| `source` | `str`（示例 `amazon_reviews_2023`） | 必填 | 数据来源 | `[E]` `一:L477` |
| `source_id` | `str`（示例 `review_001`） | 必填 | 原始记录 ID | `[E]` `一:L478` |
| `is_synthetic` | `bool` | 必填 | 模拟标记 | `[D]` `一:L488`；`[E]` `一:L479` |

#### 3.11.3 `UserContext`（模块一 9.3，Planner 的 Memory 视图）

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `user_id` | `str` | 必填 | 用户 ID | `[E]` `一:L650` |
| `long_term_preferences` | `list[str]` | 必填 | 长期偏好 | `[E]` `一:L652-655` |
| `short_term_interests` | `list[str]` | 必填 | 近期兴趣 | `[E]` `一:L657-659` |
| `relevant_episodes` | `list` | 必填，可空 | 相关事件摘要 | `[E]` `一:L661`（元素结构未定义） |
| `current_request` | `{category:str, product_type:str, max_price:float, preferences:list[str]}` | 必填 | 当前需求 | `[E]` `一:L663-671` |

#### 3.11.4 `Persona`（模块一 §4）

`user_id:str`；`persona:{preferred_categories:list[str], positive_preferences:list[str], negative_preferences:list[str], known_interests:list[str]}`。出处 `[E]` `一:L217-234`。Persona 必须可追溯到历史证据，且不得整体视为真实用户事实（`[D]` `一:L237-239`）。

#### 3.11.5 `MemoryUpdateBatch`（文档五 6.3，写入 DTO）

| 字段 | 目标类型 | 必填/默认 | 语义 | 出处 |
|---|---|---|---|---|
| `memory_updates` | `list[MemoryUpdateItem]` | 必填 | 一次反馈产生的更新集合 | `[E]` `五:L649` |
| item（语义记忆） | `{type:str, action:str, content:str, source:str, target_store:str}` | 必填 | 示例 type=`long_term_preference`、action=`add`、source=`explicit_user_feedback`、target_store=`mem0` | `[E]` `五:L650-656` |
| item（交互事件） | `{type:"interaction", action:str, event:{event_type,product_id,content,source,is_synthetic}, target_store:"sqlite_interaction_history"}` | 必填 | 结构化事件写入 SQLite | `[E]` `五:L657-668` |
| 关联键 | `user_id`、`product_id`、`source_id` 关联两库 | `[D]` `五:L673` | | |

> `action` 枚举（add/update/delete…）与 `type` 枚举文档未定义（`[?]`）。

### 3.12 SQLite 数据表 Schema（模块一 §3.1，正式表格定义）

#### 3.12.1 用户行为表

| 字段 | 类型 | 必填 | 语义 | 出处 |
|---|---|---|---|---|
| `user_id` | `str` | 是 | 用户 ID | `[D]` `一:L125` |
| `parent_asin` | `str` | 是 | Amazon 原始商品分组标识 | `[D]` `一:L126` |
| `product_id` | `str` | 是 | 内部统一主键（接入层映射） | `[D]` `一:L127` |
| `rating` | `float\|int` | 是 | 用户评分 | `[D]` `一:L128` |
| `review_title` | `str` | 否 | 评论标题 | `[D]` `一:L129` |
| `review_text` | `str` | 否 | 评论内容 | `[D]` `一:L130` |
| `timestamp` | `int`（unix） | 是 | 行为时间 | `[D]` `一:L131` |
| `verified_purchase` | `bool` | 否 | 是否验证购买 | `[D]` `一:L132` |

#### 3.12.2 商品信息表（Product Catalog）

| 字段 | 类型 | 必填 | 语义 | 出处 |
|---|---|---|---|---|
| `parent_asin` | `str` | 是 | 原始分组标识 | `[D]` `一:L138` |
| `product_id` | `str` | 是 | 内部主键 | `[D]` `一:L139` |
| `title` | `str` | 是 | 商品名 | `[D]` `一:L140` |
| `category` | `str` | 是 | 类别 | `[D]` `一:L141` |
| `features` | `list[str]\|str` | 否 | 商品特征 | `[D]` `一:L142` |
| `description` | `str` | 否 | 描述 | `[D]` `一:L143` |
| `price` | `float` | 是（硬约束依赖） | 价格 | `[D]` `一:L144` |
| `details` | `dict` | 否 | 其他属性 | `[D]` `一:L145` |

> **缺口**：Reranker Document 模板需要 `Brand`（`四:L162`），模块二 `available_product_fields` 也列了 `brand`（`二:L216`），但商品信息表**没有 `brand` 字段**（只有 `details` 兜底）→ 见 4.2-C8。
> **缺口**：`price` 的货币、单位、是否可为 `"$25.99"` 字符串、缺失值表示（null vs -1）均未定义（`[?]`）。

### 3.13 Trace Schema

| 类 | 字段 | 出处 |
|---|---|---|
| `PlannerTrace`（12 字段，正式表格） | `request_id`, `plan_id`, `parent_plan_id`, `mode`, `input_context`, `output_plan`, `validation_result`, `replan_reason`, `model_name`, `latency`, `token_usage`, `timestamp` | `[D]` `二:L761-776` |
| `RoundExecutionTrace`（7 项，散列式列举） | TIGER 召回数量、E5 召回数量、候选合并数量、Reranker 排序结果、Judgment 判断结果、Controller 动作、最终推荐结果 | `[D]`（语义）`二:L778-786`；字段名未定义 `[?]` |
| `RetrievalTrace` | **文档未定义** | `[?]` |
| `RerankerTrace`（分数/排名/耗时） | 六份文档未给字段；`docs/STARTUP_PROMPT.md:L205` 只要求"记录原始分数和耗时" | `[?]`（见 `docs/STARTUP_PROMPT.md:L205`） |
| `JudgmentTrace` | **文档未定义** | `[?]` |
| `ControllerTrace` | **文档未定义**（State 有 `action`/`stop_reason`，但无 trace 结构） | `[?]` |
| `GeneratorTrace` | **文档未定义** | `[?]` |
| `MemoryTrace` | **文档未定义**（模块一交付物只写"Memory 运行轨迹与来源记录"） | `一:L749` |

### 3.14 Pydantic 建模清单总览（按模块分组、按落地优先级排序）

| 优先级 | 类名 | 模块 | 依据可靠性 | 说明 |
|---|---|---|---|---|
| P0 | `RecommendationPlan` | 二 | 顶层 `[D]`，子结构 `[E]` | 最高优先级，先落顶层 + 枚举 |
| P0 | `Intent`, `Constraints`, `HardConstraint`, `SoftPreference`, `Preferences`, `RetrievalPlan`, `TigerPlan`, `E5Plan`, `RankingGuidance`, `Clarification` | 二 | `[E]` + 规则 `[D]` | 与 Plan 同批 |
| P0 | `Candidate`, `CandidatePool`, `ProductMetadata` | 三 | `[E]`（5.6 节） | 下游 Reranker/Judgment/Controller 全部依赖 |
| P0 | `InteractionEvent`, `UserContext`, `MemoryRecord` | 一 | `[D]`（字段约束）/`[E]`（结构） | TIGER 与 Planner 的共同上游 |
| P0 | `RecommendationState`, `BudgetState` | 五 | `[D]`（TypedDict）/`[E]`（预算 JSON） | LangGraph 骨架 |
| P1 | `TigerCandidate`, `TigerRetrievalResult` | 三 | `[E]` | 阶段 B 验收需要 |
| P1 | `E5Candidate`, `E5RetrievalResult` | 三 | `[E]` | 阶段 1 最先可跑 |
| P1 | `ConstraintResult`, `ConstraintSummary` | 三/五 | `[?]`（结构未定义） | **必须先由负责人定义** |
| P1 | `RankedProduct` | 四 | `[?]`（只点名） | **必须先由负责人定义字段名** |
| P1 | `JudgmentResult`, `PreferenceFit`, `AttributeEvidence` | 五 | `[E]` | 需补枚举 |
| P1 | `RecommendationOutput`, `RecommendationItem` | 五 | `[E]` | 需补系统级 caveat |
| P1 | `ReplannerInput`, `ExecutionFeedback`, `ControllerFeedback` | 二 | `[E]` | 需与 Plan 结构对齐 |
| P2 | `PlannerTrace`, `RoundExecutionTrace` | 二 | `[D]`/`[E]` | 观测性 |
| P2 | `Persona`, `MemoryUpdateBatch`, `MemoryUpdateItem` | 一/五 | `[E]` | 反馈闭环 |
| P2 | 其他 Trace 类（Retrieval/Reranker/Judgment/Controller/Generator/Memory） | — | `[?]` | 需先定义 |

---

## 4. 冲突与缺口清单

> 用法：**B 类必须由项目负责人先决策再写代码**；C/G/T 类可在实现时按本文档标注补齐，但不得静默选择。
> 每条都给了行号证据，便于在评审会上直接翻到原文。

### 4.1 阻断级（P0，先决策再动手）

| ID | 主题 | 冲突/缺口 | 证据 | 影响 | 建议处置 |
|---|---|---|---|---|---|
| B1 | 硬约束状态与 `constraint_status` 契约 | 三态 `pass/fail/unknown`（`A:L118-123`、`三:L748-752`）与第 4 态 `not_required`（`五:L211`）并存；同名 `constraint_status` 在候选池里是 `str`（`三:L772`），在 Judgment 输入里是 `dict`（`五:L170-173`） | `A:L118-123`、`三:L748-752`、`三:L772`、`五:L170-173`、`五:L211` | Constraint Evaluator、候选池、Judgment 三个模块的接口同时不确定；`unknown` 是否算 `pass` 的判定会分叉 | 拆为 `constraint_status: Literal["pass","fail","unknown"]`（聚合）+ `constraint_results: dict[str, ConstraintResult]`（逐字段）；`not_required` 改为"约束不存在"而非状态值；建 ADR |
| B2 | 嵌套子结构只有示例、无正式字段表 + 模块二内部命名冲突 | `intent/constraints/preferences/retrieval/ranking/clarification` 的子字段仅在示例中（`二:L615-648`）；`二:L294-309` 用 `current_preferences`/`historical_preferences`，`二:L629-634` 用 `current`/`historical` | `二:L294-309`、`二:L615-648`、`二:L629-634`、`二:L664-680`（只覆盖顶层） | Pydantic 字段名、必填性无法唯一确定；两套示例会导致前后端/上下模块对不上 | 以 6.2 节为准落 Schema，把 4.3 节示例标注为过时并在文档里修正；`extra="forbid"` 需负责人确认 |
| B3 | RRF 的 k 无数值、两级是否同 k 未定义 | 公式 `三:L691-694`、E5 内部 `三:L502`、跨路 `三:L717`；k 仅"平滑常数" `三:L701`；`A:L234` 要求"RRF 常数配置化" | `三:L502`、`三:L691-694`、`三:L701`、`三:L717`、`A:L234` | Fusion 无法产出可复现基线；不同实现数值不可比 | 配置化 `rrf_k_e5` 与 `rrf_k_cross`（是否同值待定），默认值需负责人确认后写入 `configs/` |
| B4 | Controller State 与示意代码不一致 | 代码读 `state["hard_constraints_satisfied"]`、`state["quality_sufficient"]`，但 State 字段表里没有这两个键；`action:str`、`stop_reason:str` 无枚举；`budget_state:dict` 弱类型 | `五:L397-404` vs `五:L446-489`（尤其 `五:L469-482`） | LangGraph 条件边与节点签名无法定稿，运行期 KeyError | 二选一：补 State 字段并定义 `Action`/`StopReason` 枚举，或改写代码改用 `constraint_summary`/`quality_metrics`；由负责人定 |
| B5 | Reranker 加载与打分接口自相矛盾 | `四:L59` 声明"yes/no 对应 Token 的分数"（生成式判分），`四:L198-228` 给出 `sentence_transformers.CrossEncoder(..., prompts={...}, default_prompt_name=...)` + `predict`；无 `max_length`/batch/超时；`A:L273` 列为未验证 | `四:L59`、`四:L198-228`、`四:L230`、`A:L273` | 模型适配层无法开工；`prompts`/`default_prompt_name` 是否为该库合法参数需实测 | 先做 30 分钟冒烟实验（`transformers` `AutoModelForCausalLM` + yes/no logits 与 CrossEncoder 两种路径各跑 4 条 pair），再定接口；结果写 ADR |
| B6 | 商品主键映射基数与 Catalog 字段缺失 | 用户行为表/商品表只有 `parent_asin`+`product_id`，无子 ASIN；但数据来源写"商品 ID（ASIN）"；`parent_asin`→`product_id` 是 1:1 还是 1:N 未定义；Catalog 无 `brand` | `一:L109`、`一:L126-127`、`一:L138-139`、`一:L147`、`A:L271`；`四:L162`、`二:L216` | 去重、评估、TIGER 解码都可能在同一父 ASIN 下重复；Reranker Document 的 Brand 段无数据源 | 明确 `product_id` 生成规则（建议保留 `parent_asin` 为稳定键 + 子 `asin` 列）；Catalog 增 `brand`；建 ADR |

### 4.2 字段名 / 类型 / 结构不一致（C 类）

| ID | 问题 | 冲突双方 | 证据 | 影响 |
|---|---|---|---|---|
| C1 | `constraint_status` 两种类型 | 候选池 `"pass"` vs Judgment 输入 `{"price":"pass"}` | `三:L772` vs `五:L170-173` | 同一字段名两种 schema，序列化必然失败（详见 B1） |
| C2 | 硬约束状态多出 `not_required`，另有一个跨层混淆的 `judgment_unavailable` | `not_required`（硬约束第 4 态）vs 三态；`judgment_unavailable` 是 Judgment 失败态而非约束态 | `五:L211` vs `A:L118-123`；`五:L838` | 状态机值域不闭合；统计 `pass` 比例时口径不明 |
| C3 | 偏好四键命名冲突（模块二内部） | 4.3 节 `current_preferences/historical_preferences` vs 6.2 节 `current/historical` | `二:L294-309` vs `二:L629-634`、`二:L660` | Plan 校验器写哪一套都可能判错 |
| C4 | 用户/记忆偏好命名至少 5 套 | ①`long_term_preferences/short_term_interests/negative_preferences` ②`positive_preferences/negative_preferences` ③`interested_categories` ④`relevant_episodes` ⑤`user_preferences.positive/negative` | `二:L160-170`；`三:L567-575`；`一:L191-201`、`一:L217-233`；`一:L648-672`；`五:L146-154` | 需要一层 Adapter 归一；否则每接一个模块就重写一次映射 |
| C5 | State 字段缺失 + 弱类型 | `hard_constraints_satisfied`、`quality_sufficient` 只在代码里；`budget_state` 声明 `dict` | `五:L397-404` vs `五:L446-489`、`五:L479` | 见 B4 |
| C6 | BudgetState 字段数量不一致 | 最终阶段 5 字段 vs Replanner 输入 3 字段 | `五:L427-434` vs `二:L499-503` | Replanner 输入的 `budget_state` 与 Controller 的 `BudgetState` 不能是同一模型 |
| C7 | `eligible_candidates` 只出现在文字里、不在输出格式里 | 文字："pass 进入 `eligible_candidates`"；5.6 输出：只有 `candidates` + `unverified_candidates` | `三:L750` vs `三:L764-784` | 命名不一致；下游不知道有 `fail` 集（诊断记录放哪未定义） |
| C8 | 商品目录缺 `brand`、候选池元数据只有 3 字段 | Catalog 8 字段无 `brand`；Reranker Document 需 `Brand`；`product_metadata` 只有 `title/price/category` | `一:L136-145`、`四:L162`、`三:L776-780` | 精排输入拼接会拿到空 brand；"来自 Product Catalog"（`四:L180`）无法满足 |
| C9 | 时间格式不统一 | `InteractionEvent.timestamp` 为 unix 整数；`MemoryRecord.created_at/last_observed_at` 为 `"YYYY-MM-DD"` 字符串；无时区 | `一:L476`、`一:L439-440` | 时间排序、时间切分、`<` 比较可能静默出错 |
| C10 | 来源键大小写不一致 | `retrieval_sources` 值 `"TIGER"`/`"E5_FAISS"`，但 `source_ranks`/`raw_scores` 键 `"tiger"`/`"e5"` | `三:L664-667`、`三:L668-671`、`三:L774-775` | 建议枚举化：`retrieval_sources: list[SourceEnum]`、`source_ranks: dict[SourceEnum,int]`，值统一小写 |
| C11 | 分数键名 4 套、Reranker 分数键名未定 | `generation_score` / `similarity_score` / `raw_scores.{tiger,e5}` / `fusion_score`；Reranker 仅文本示例 `Product P001: 5.82` | `三:L362`、`三:L541`、`三:L775`、`三:L771`、`四:L187-190` | 跨模块汇总分数时无法自动化；Reranker 输出无法落 Schema |
| C12 | `memory_type` 枚举混用 | `long_term_preference`（`一:L420`）vs Long-term/Dynamic/Episodic 三层（`一:L379-409`）vs `type=long_term_preference` + `type=interaction`（`五:L650-661`） | `一:L379-409`、`一:L420`、`五:L649-668` | Mem0 存储与 DTO 的枚举必须统一，否则过滤/检索失效 |
| C13 | Plan 协议消费方名单不一致 | `二:L682` 列 Planner/Retrieval/Reranker/Replanner/Controller；但 Judgment 输入含 "Recommendation Plan" | `二:L682` vs `五:L137`、`五:L156-159` | 文档层面的小矛盾；建议统一为"全链路唯一计划协议" |
| C14 | 硬约束表达 4 套 | 对象数组（`二:L622-624`）vs `{"max_price":30}`（`一:L614-616`、`三:L736-740`、`五:L156-159`） | 同左 | Constraint Evaluator 入参必须固定为对象数组；其余三处需改成"摘要视图"或删除 |

### 4.3 未定义接口（G 类）

| ID | 缺口 | 证据 | 影响 |
|---|---|---|---|
| G1 | `ReplannerInput.original_request` 是独立 DTO（`category/product_type/max_price`）还是引用 `Intent`+`HardConstraint`，未定义 | `二:L468-472` vs `二:L615-624` | Replanner 的"不得改原始硬约束"（`二:L570`）无法用同一结构做 diff 校验 |
| G2 | Judgment 失败态（`judgment_unavailable`）无字段承载；批量请求的封装结构未定义 | `五:L838`、`五:L188`、`五:L231-264` | 超时商品无法进入 State；`judgment_results` 会混入异构元素 |
| G3 | Reranker 的 Query 拼装算法与 Instruction→模板映射未定义 | `四:L133-145`、`二:L644-647` | 同一 Plan 可能拼出不同 Query，评测不可复现 |
| G4 | `SystemCapabilities` 未进入 State，Planner 输入装配未定义 | `二:L206-222` vs `五:L446-489` | Planner 无法校验"模块是否可用"（`二:L744`） |
| G5 | Generator 输出无系统级 caveat 字段（预算耗尽/未验证需求说明） | `五:L380` 要求 vs `五:L560-581` 结构 | 预算耗尽时无法合规输出 |
| G6 | `PlanValidationResult` 结构未定义 | `二:L735-751`、`二:L771` | 校验失败信息无法结构化落 Trace / 回灌 Replanner |
| G7 | 硬约束 per-field 结果结构未定义（只有状态语义） | `三:L748-754`、`A:L118-123` | Constraint Evaluator 无法返回可审计结果 |
| G8 | `unverified_candidates` 元素结构未定义（是否与 `Candidate` 同构、是否含 `fail` 集） | `三:L783`、`三:L752` | Controller 无法基于 `unknown` 集合做 Replan 判断 |
| G9 | `RankedProduct` 只被点名，无字段定义 | `四:L192` | Reranker→Judgment 的接口缺口（见 C11） |
| G10 | 检索/精排/Judgment/Controller/Generator/Memory 的 trace 结构未定义 | `二:L778-786`、`一:L749`、`A:L240` | 无法满足 `A:L80`「保存完整可追溯运行轨迹」 |
| G11 | Mem0 检索接口（top_k、返回结构、相似度阈值、去重规则）未定义 | `一:L624-635`、`一:L344-350` | User Context 装配不可测 |
| G12 | 澄清挂起/恢复的会话字段未定义（只有 `previous_user_messages`） | `二:L190-196`、`二:L708`、`五:L491` | `wait_for_user` 恢复时不知道等待哪个问题 |
| G13 | 用户序列的最小交互数与最大长度未定义 | `三:L212-245`、`三:L251-271` | TIGER 训练样本切分与 OOM 风险不可控 |
| G14 | 数据划分比例/时间截点未定义 | `一:L722-728`、`三:L950-966`、`A:L248` | 防泄漏要求无法落成脚本参数 |

### 4.4 未给数值的阈值与参数（T 类）

| ID | 参数 | 文档现状 | 证据 |
|---|---|---|---|
| T1 | RRF `k`（E5 内部与跨路） | 公式有、k 无数值（见 B3） | `三:L502`、`三:L691-694`、`三:L701`、`三:L717` |
| T2 | `standard`/`expanded` → 实际候选数 | "由 Python 配置映射"，映射表未给 | `二:L334`、`二:L658`、`三:L827` |
| T3 | 预算初值（replan/retrieval/api/候选） | 示例 2/4/50/100/12，且明确"仅为配置示例" | `五:L427-437` |
| T4 | `max_replan_rounds` | 字段存在、无默认值/上限 | `五:L481`、`五:L499` |
| T5 | 质量阈值（核心偏好覆盖率、证据充分率等） | "通过验证集确定，不使用未验证固定阈值" | `五:L421` |
| T6 | `required_count` 的计算规则 | 字段存在、来源未定义 | `五:L471`、`五:L397` |
| T7 | Judgment 批量大小与并发上限 | 只说"可合并批量"、`A:L159` 要并发上限 | `五:L188`、`A:L159` |
| T8 | Judgment 单商品输入长度上限 | 未定义 | `五:L142-176` |
| T9 | Reranker batch size / max_length / 截断策略 | 只说"批量"，长度列为未验证 | `四:L230`、`A:L273` |
| T10 | DeepSeek 模型 ID、温度、max_tokens、超时、重试、并发 | 全部未定义（`A:L274` 列为未验证）；接口签名只有 `generate(prompt=...)` | `A:L238-240`、`A:L274`、`二:L593-595` |
| T11 | E5 `queries` 条数上限与单条长度上限 | 校验项存在、阈值缺失 | `二:L742` |
| T12 | FAISS 索引类型与检索参数 | "根据数据规模选择"，只给了 L2 归一化+内积方案 | `三:L513-520` |
| T13 | RQ-VAE 层数 / codebook 数 / 码本大小 / 输入维度 | 全未定义，`[12,05,31]` 明确"仅为示意" | `三:L166-204`（尤其 `三:L191-194`） |
| T14 | E5 checkpoint / 向量维度 / Query-Passage 前缀 | "遵循最终选用的模型配置，并通过配置固定"（当前未选） | `三:L164`、`三:L505`、`A:L272` |
| T15 | 多样性阈值与 ILD 计算口径 | 只说"多样性不足"是 Replan 触发条件 | `五:L308`、`五:L325`、`二:L491` |
| T16 | Memory `confidence` 计算公式与权重 | 只说由证据质量/重复度/明确度/时间综合计算 | `一:L461` |
| T17 | 记忆时间衰减规则 | "可以根据后续行为更新或衰减"，无函数与半衰期 | `一:L398` |
| T18 | Top-K 的 K（最终推荐数与召回 K） | 只写 Top-K；评测另用 Recall@50/100、NDCG@10 | `五:L585`、`总:L111`、`三:L315`、`三:L913-917` |
| T19 | 用户序列最大长度 / 最小交互数 | 未定义（同 G13） | `三:L212-271` |
| T20 | 训练/验证/测试划分比例 | 只说按时间切分 | `一:L722-728`、`三:L950-966` |

### 4.5 旧项目残留、旧版文档与冗余（R 类）

| ID | 问题 | 证据 | 处置建议 |
|---|---|---|---|
| R1 | **两份 `STARTUP_PROMPT.md` 并存且内容冲突**：根目录版 295 行（旧），`docs/STARTUP_PROMPT.md` 242 行（V1.1） | 根 `STARTUP_PROMPT.md:L114`（"三个模块设计文档"，实际六份）、`:L117`（字段名 **`PlannerPlan`**，正确名 `RecommendationPlan`）、`:L152-252`（Phase 0-7 划分）；`docs/STARTUP_PROMPT.md:L3`（V1.1）、`:L65-74`（六份清单） | 以 `docs/STARTUP_PROMPT.md` 为唯一版本，根目录旧版标记废弃或删除；**任何代码/文档不得引用 `PlannerPlan` 字段名** |
| R2 | `278M` 语义易混：旧项目"278M 搜索 Reranker"≠ 本文档 `BAAI/bge-reranker-base`（278M 轻量备选） | 根 `STARTUP_PROMPT.md:L82`；`四:L45`、`四:L76-82`、`四:L92` | 排除清单里单独注明，避免"看到 278M 就删"的误伤 |
| R3 | JEV / Wikipedia 多跳 / Reader / 本地 Qwen3-8B 均为**否定式**表述，不是待实现功能 | `A:L11`、`A:L82`、`A:L264`、`总:L141`、`docs/STARTUP_PROMPT.md:L78`、`:L98-100` | 视为"禁止引入"，不视为需求；代码中不得出现对应依赖 |
| R4 | 旧方案残留词出现在"暂不纳入"清单：Bounded RSI、独立 SASRec | `总:L233-234` | 明确为 Out of Scope，不建模块 |
| R5 | 模块一以"不再训练 SASRec"作对照说明 | `一:L18` | 同上；避免有人据此恢复 SASRec 模块 |
| R6 | `README_替换说明.md` 是替换清单，不是设计文档；其"8 份文件"与实际目录分布（六份在 `docs/`，AGENTS 在根，两份 STARTUP）不一致 | `README_替换说明.md:L3` | 审计/实现时以六份 `docs/` 文档 + 根 `AGENTS.md` 为准，不引用 README 作为设计依据 |
| R7 | 文档反复引用的旧文件 `设计生成式推荐项目.txt` 在仓库中不存在 | `A:L11`、`docs/STARTUP_PROMPT.md:L78` | 无需清理；审计时不得引用其内容（也不得因缺失而阻塞） |
| R8 | 模块二 4.3 节示例（`current_preferences` 等）是 V1.1 之前的命名习惯，与 6.2 节统一 Schema 不一致 | `二:L294-309` | 视为**过时示例**（不是需求），以 6.2 节为准（同 C3） |

---

## 5. 配置项清单

> 原则：`A:L234`「模型名、API 参数、RRF 常数、Top-K、预算、超时和并发数均配置化」。

### 5.1 文档已明确给出的配置（可直接登记）

| 配置项 | 值 | 出处 |
|---|---|---|
| 数据集 | Amazon Reviews 2023 — Beauty and Personal Care | `一:L104`、`总:L11` |
| Planner / Replanner 模型 | DeepSeek API（共用同一模型服务与 Client） | `二:L44`、`A:L71`、`二:L839-841` |
| Judgment 模型 | DeepSeek API（**不是**本地 Qwen3-8B） | `五:L115-117`、`A:L77`、`A:L82` |
| Generator 模型 | DeepSeek API（与 Planner/Judgment 独立 Prompt/Schema/超时） | `五:L515`、`A:L238` |
| Reranker 主模型 | `Qwen/Qwen3-Reranker-0.6B` | `四:L204`、`四:L88-90` |
| Reranker 对照模型 | `BAAI/bge-reranker-v2-m3` | `四:L65`、`四:L91` |
| Reranker 轻量备选 | `BAAI/bge-reranker-base`（278M） | `四:L80`、`四:L92` |
| Reranker 是否微调 | 第一阶段冻结 | `四:L234`、`A:L76` |
| 记忆方案 | Mem0 OSS + Qdrant + SQLite | `一:L338-369`、`A:L69` |
| 记忆 LLM | DeepSeek API（Memory Extraction） | `一:L363` |
| 记忆 Embedding | 独立文本 Embedding 模型（具体未定） | `一:L364` |
| 事件存储 | SQLite `InteractionEvent` | `一:L465`、`五:L615` |
| 向量索引 | FAISS（具体类型未定） | `三:L509`、`三:L978` |
| 融合算法 | RRF（两级） | `三:L679-717`、`A:L125-129` |
| 硬约束执行 | 唯一 Python Constraint Evaluator | `A:L118`、`三:L754` |
| 预算档位枚举 | `standard` / `expanded` | `二:L658` |
| 召回策略枚举 | `hybrid_balanced` / `behavior_priority` / `semantic_priority` | `二:L657` |
| Plan 模式枚举 | `initial` / `replan` | `二:L656` |
| 硬约束状态枚举 | `pass` / `fail` / `unknown`（另有 `not_required` 待裁决） | `A:L118-123`、`五:L211` |
| 预算配置文件路径 | `configs/controller.yaml` | `五:L437` |
| 轨迹存储 | JSONL + SQLite | `A:L80`、`二:L848` |
| Controller 编排 | LangGraph + Python（不引入 LLM） | `五:L292-299`、`A:L78` |

### 5.2 文档未定义、必须补的配置（建议键名仅为本审计建议，需负责人确认）

| 建议配置键 | 含义 | 文档状态 | 缺失证据 |
|---|---|---|---|
| `rrf.k_e5` / `rrf.k_cross` | 两级 RRF 平滑常数 | **未定义（无任何数值）** | `三:L502`、`三:L701`、`三:L717` |
| `retrieval.budget.standard` / `.expanded`（tiger/e5 各一组） | 档位→候选数映射 | 未定义 | `二:L334`、`三:L827` |
| `retrieval.e5.top_k` | 每条 Query 的 Top-K | 未定义 | `三:L495-500` |
| `retrieval.tiger.top_k` / `beam_size` | 生成候选数与 beam | 未定义（阶段 A 才提 Beam Search） | `三:L315`、`三:L334` |
| `retrieval.e5.model` / `dim` / `query_prefix` / `passage_prefix` | E5 checkpoint 与前后缀 | 未定义（明确"遵循最终选型"） | `三:L164`、`三:L505`、`A:L272` |
| `faiss.index_type` / `metric` / `nlist` / `nprobe` | FAISS 索引类型与参数 | 未定义（仅给归一化+内积方案） | `三:L513-520` |
| `semantic_id.num_levels` / `codebook_size` / `num_codebooks` | RQ-VAE 结构 | 未定义 | `三:L166-204` |
| `tiger.max_history_len` / `min_history_events` | 序列长度约束 | 未定义 | `三:L212-271`、`三:L837` |
| `reranker.batch_size` / `max_length` / `timeout_s` / `concurrency` | 精排工程参数 | 未定义 | `四:L230`、`A:L240`、`A:L273` |
| `deepseek.<task>.model` / `temperature` / `max_tokens` / `timeout_s` / `max_retries` / `concurrency` | 三类 Task 独立参数 | 未定义（`A:L274` 未验证） | `A:L238-240`、`A:L274`、`二:L593-595` |
| `planner.max_queries` / `query_max_chars` | E5 Query 数量与长度上限 | 未定义（校验项存在） | `二:L742` |
| `controller.max_replan_rounds` / `max_retrieval_calls` / `max_api_calls` | 全局预算 | 未定义（示例非默认） | `五:L427-437` |
| `controller.required_count` | 所需有效候选数 | 未定义 | `五:L471` |
| `controller.quality_thresholds.*` | 偏好覆盖/证据充分/多样性阈值 | 未定义（应由验证集定） | `五:L421` |
| `controller.diversity_metric` / `threshold` | 多样性口径 | 未定义 | `五:L308`、`五:L325` |
| `judgment.batch_size` / `concurrency` / `max_input_tokens` | Judgment 批量与并发 | 未定义 | `五:L188`、`A:L159` |
| `generator.top_k` / `max_output_tokens` | 最终推荐数量与输出上限 | 未定义 | `五:L585` |
| `memory.retrieval_top_k` / `score_threshold` / `confidence_weights` / `decay_half_life` | Mem0 检索与置信度 | 未定义 | `一:L461`、`一:L398`、`一:L624-635` |
| `data.split.ratios` / `cutoff_ts` | 时间切分比例与截点 | 未定义 | `一:L722-728`、`三:L950-966` |
| `eval.k`（Recall@K / NDCG@K / HitRate@10） | 评测 K | 部分给出（50/100/10），未统一 | `三:L913-917`、`四:L277-280`、`总:L217-219` |

---

## 6. 旧项目内容排除清单

> 目的：防止把旧版 JEV / Wikipedia 多跳问答 / Reader 搜索项目的内容混入当前商品推荐项目（`A:L11`、`A:L264`）。

### 6.1 术语与模块（一律不得进入当前设计/代码）

| 排除项 | 原因 | 出处 |
|---|---|---|
| JEV API / JEV 检索 | 本项目不接入 JEV | `A:L82`、`总:L141`、`docs/STARTUP_PROMPT.md:L78` |
| Wikipedia 多跳问答 / 多跳搜索 / Reader 架构 | 旧项目形态，非商品推荐 | `A:L11`、`A:L264`、根 `STARTUP_PROMPT.md:L80-83` |
| 旧"278M 搜索 Reranker" | 旧项目组件；与本文档 278M 的 `bge-reranker-base` 不是同一角色 | 根 `STARTUP_PROMPT.md:L82`；对比 `四:L76-82` |
| 本地 Qwen3-8B 作为 Judgment | 第一阶段 Judgment 用 DeepSeek API | `A:L77`、`总:L141`、`五:L115-117` |
| Controller LLM | Controller 只用 LangGraph + Python | `A:L78`、`五:L299` |
| Bounded RSI 自进化 | 明确不纳入本课程设计 | `总:L233` |
| 独立 SASRec 用户兴趣建模 / 兴趣融合训练 | 不训练，改用 Mem0 | `一:L18`、`总:L234` |
| 单值 `remaining_budget: int` | 被明确否定，用多维 BudgetState | `五:L425` |
| 跨模型原始分数直接相加 | 被明确禁止，用两级 RRF | `A:L129`、`三:L685-687`、`三:L717` |
| 顶层 `retrieval_strategy` / `tiger` / `e5_faiss` | 被明确禁止的平行字段 | `A:L114`、`二:L358`、`三:L825` |
| `PlannerPlan`（字段名） | 不存在于任何 V1.1 设计文档；仅为旧版 STARTUP_PROMPT 笔误 | 根 `STARTUP_PROMPT.md:L117` |

### 6.2 文件与文档

| 文件 | 状态 | 处置 |
|---|---|---|
| `设计生成式推荐项目.txt` | 仓库中不存在；文档仅作历史背景引用 | 不创建、不恢复、不引用（`A:L11`） |
| 根目录 `STARTUP_PROMPT.md`（295 行） | 旧版，与 `docs/STARTUP_PROMPT.md`（V1.1）冲突（`PlannerPlan`、"三个模块文档"） | 建议废弃/删除，统一使用 `docs/STARTUP_PROMPT.md`（见 R1） |
| `README_替换说明.md` | 替换清单，非设计文档 | 不作为设计依据（见 R6） |
| 旧聊天记录 | `A:L9` 明确不得作为依据 | 不引用 |

### 6.3 易被误判为"旧残留"但实际合法的内容（不要删）

| 内容 | 为什么合法 | 出处 |
|---|---|---|
| `BAAI/bge-reranker-base`（约 278M） | 本文档中的"轻量备选"，用于效率对照；与旧项目"278M 搜索 Reranker"不同 | `四:L76-82`、`四:L92` |
| Mem0 中的"事件摘要" | 属于语义记忆（Episodic Memory 的可提炼摘要），但**不得**当 TIGER 行为序列 | `一:L409`、`一:L371` |
| Qwen3-Reranker-0.6B | 只是精排模型，不承担 Judgment | `A:L82`、`四:L290` |
| Amazon `review`/`rating` 事件 | 作为历史交互代理信号合法，但必须按真实来源记录、不得伪装点击/购买 | `一:L149`、`三:L245` |
| `pass`/`fail`/`unknown` 三态 | 当前项目的硬约束契约，不是旧残留 | `A:L118-123` |

### 6.4 代码层建议的禁用关键词（供 CI/lint 自查）

`jev`、`wikipedia`、`multi_hop`、`reader`、`qwen3-8b`（作为 Judgment）、`remaining_budget`（单值）、`retrieval_strategy`（顶层）、`PlannerPlan`、`click`/`purchase`（描述 Amazon review 事件时）。
依据：`A:L11`、`A:L82`、`A:L114`、`A:L264`、`五:L425`、`一:L149`。

---

## 7. 需要项目负责人确认的问题（按优先级）

| # | 问题 | 关联 ID |
|---|---|---|
| Q1 | 硬约束状态最终值域是什么？是否废弃 `not_required`？`constraint_status` 是否拆成聚合状态 + 逐字段 `constraint_results`？ | B1、C1、C2、G7 |
| Q2 | 两级 RRF 的 `k` 取多少？E5 内部与跨路是否同一个 k？是否需要 weighted RRF（第一版是否启用）？ | B3、T1 |
| Q3 | 嵌套子结构（Intent/Constraints/Preferences/Retrieval/Ranking/Clarification）的字段名与必填性，是否以 6.2 节示例为准？是否启用 `extra="forbid"`？ | B2、C3 |
| Q4 | Reranker 用哪条加载路径：`transformers` + `AutoModelForCausalLM` 取 yes/no logits，还是 `sentence-transformers` `CrossEncoder`？`max_length`/batch 取多少？ | B5、T9 |
| Q5 | `RecommendationState` 补 `hard_constraints_satisfied`/`quality_sufficient`，还是改写 Controller 代码改用 `constraint_summary`/`quality_metrics`？`action`/`stop_reason` 枚举表？ | B4、C5 |
| Q6 | `product_id` 的生成规则（是否保留父 ASIN 与子 ASIN 两列）？Catalog 是否新增 `brand`、`attributes`？ | B6、C8 |
| Q7 | `tiger_candidate_budget`/`e5_candidate_budget` 是"每轮额度"还是"剩余额度"？`standard`/`expanded` 的数值映射表？ | C6、T2、T3 |
| Q8 | `Top-K` 的 K、评测 K（Recall@50/100、NDCG@10）是否都固定进配置？`required_count` 如何计算？ | T6、T18 |
| Q9 | Judgment 的失败态字段（`judgment_unavailable`）与批量请求结构如何定义？是否需要 `schema_version`？ | G2、C2 |
| Q10 | 时间切分比例与截点规则？`InteractionEvent.timestamp` 与 Memory 时间字段统一为哪种格式（unix UTC / ISO8601）？ | C9、G14、T20 |

---

## 附录 A：本次审计的读取完整性声明

- 已完整逐行阅读：`AGENTS.md`（282 行全部）、`docs/项目总体方案.md`（264 行全部）、`docs/模块一`（763 行全部）、`docs/模块二`（890 行全部）、`docs/模块三`（1056 行全部）、`docs/Reranker 模型选型与设计.md`（304 行全部）、`docs/最终阶段`（907 行全部）。
- 交叉参考：`docs/STARTUP_PROMPT.md`（242 行）、根 `STARTUP_PROMPT.md`（295 行）、`README_替换说明.md`（28 行）。
- 本文件**不包含任何实现代码、不修改其他文件**；所有"建议"均标注为建议，所有"文档未定义"均未被臆造填补。
- 行号以审计时的工作区文件为准；若文档被后续修订，行号需重新核对。
