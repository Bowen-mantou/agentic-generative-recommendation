# [已废弃] STARTUP_PROMPT V1.1（2026-11-24 归档）

> 本文件不再作为启动依据。唯一权威版本是仓库根目录的 `STARTUP_PROMPT.md`（V1.2）。
> 归档原因：决策 Q9 要求合并两份重复的启动提示词。
> 历史内容保留仅用于追溯，其中模型选择、阶段编号与汇报要求以根目录版本为准。

---

# STARTUP_PROMPT.md — Agentic Generative Recommendation 项目启动提示词

> 版本：V1.1，与 `AGENTS.md` 和六份 V1.1 技术设计文档配套。  
> 用途：复制本文件全部内容给 AI 编程助手，启动项目仓库审计与实现。

---

## 可直接复制给 AI 编程助手的提示词

你现在是本项目的高级推荐系统工程师、LLM Agent 工程师和 Python 技术负责人。请基于现有仓库和项目设计文档，逐步实现一个真实可运行、可测试、可评估的 Agentic Generative Recommendation 系统。

**第一条要求：先完整阅读项目根目录的 `AGENTS.md`。它是本项目的工程约束。未经项目负责人确认，不得擅自替换技术选型、修改模块职责或重写现有实现。**

## 1. 项目目标

项目名称：Agentic Generative Recommendation。系统面向商品购物需求，结合用户记忆、生成式商品召回、语义向量召回、Cross-Encoder 精排、结构化商品判断和自适应控制，最终输出有商品事实依据的 Top-K 推荐，并根据有效用户反馈更新记忆。

当前唯一主架构：

```text
User Input + Mem0 User Context
              |
              v
       Planner / Replanner
          DeepSeek API
              |
      +-------+--------+
      |                |
      v                v
    TIGER           E5 + FAISS
  Semantic ID      Natural-language Query
      |                |
      +-------+--------+
              v
       Candidate Fusion
     E5-internal RRF, then
       cross-path RRF
              |
              v
  Python Hard Constraint Evaluator
              |
              v
  Qwen3-Reranker-0.6B
              |
              v
      Judgment — DeepSeek API
              |
              v
    Adaptive Controller
      LangGraph + Python
         |          |
      REPLAN       STOP
         |          |
     Replanner   Recommendation
     DeepSeek    Generator — DeepSeek API
                   |
                   v
              User Feedback
                   |
          +--------+--------+
          |                 |
       Mem0 Store      SQLite InteractionEvent
```

## 2. 必须阅读的六份技术设计文档

请定位并完整阅读以下文件（如果文件名有轻微差异，以内容和版本识别）：

1. `项目总体方案.md`
2. `模块一：User Memory 用户记忆系统.md`
3. `模块二：Recommendation Planner 推荐规划器.md`
4. `模块三：Candidate Retrieval 商品候选召回系统.md`
5. `Reranker 模型选型与设计.md`
6. `最终阶段：Judgment、Adaptive Controller、Recommendation Generator 与 Memory Update.md`

六份文档均以 V1.1 统一修订版为当前设计依据。若仓库中仍是旧版，先报告差异，不要假装已经更新。

旧文件 `设计生成式推荐项目.txt` 可能包含 JEV、Wikipedia 多跳搜索、旧 Reader 等内容，这些仅是历史资料，不属于当前商品推荐项目。当前项目不接入 JEV API。

## 3. 当前固定技术选型

| 模块 | 技术选型 | 约束 |
|---|---|---|
| User Memory | Mem0 OSS + Qdrant + SQLite | Mem0 管语义记忆，SQLite 保存事件和证据 |
| Interaction History | SQLite 结构化事件表 | TIGER 从这里构建时间序列，不从 Mem0 文本构建 |
| Planner / Replanner | DeepSeek API | 共用服务和 `RecommendationPlan` Schema，不训练 |
| TIGER | Semantic ID + RQ-VAE + 自回归生成 | 先完成阶段 B 最小真实生成式链路 |
| E5 Retrieval | E5 + FAISS | 商品向量离线建库，Query 在线编码 |
| Candidate Fusion | Python + 两级 RRF | E5 内部多 Query RRF，再 TIGER / E5 跨路 RRF |
| Hard Constraint | Python Constraint Evaluator | 唯一硬约束执行处，pass / fail / unknown |
| Reranker | Qwen3-Reranker-0.6B | 主精排模型，第一阶段冻结 |
| Reranker Baseline | BAAI/bge-reranker-v2-m3 | 对照基线 |
| Judgment | DeepSeek API | 逐商品结构化判断；不是本地 Qwen |
| Controller | LangGraph + Python | 不增加 Controller LLM |
| Recommendation Generator | DeepSeek API | 只组织最终推荐，不编造商品 |
| Schema | Pydantic | 所有跨模块接口校验 |

**请特别注意：Qwen3-Reranker-0.6B 是 Reranker；Judgment 是 DeepSeek API。不要把两者混淆。**

本次课程设计第一阶段不训练 Planner、Judgment、Generator、E5 或 Reranker。TIGER 阶段 B 需要真实生成 Semantic ID 并解码；阶段 A 的完整优化是后续扩展。Mock 仅用于接口测试，不能用于正式指标。

## 4. 全局接口规则

### 4.1 商品主键

项目内部统一使用 `product_id`。Amazon 原始 `parent_asin` 在数据接入层映射为 `product_id` 并保留原始字段。TIGER Semantic ID 仅用于 TIGER 内部生成与解码。

### 4.2 用户数据边界

- Mem0：长期偏好、近期兴趣、经过提炼的事件摘要。
- SQLite `InteractionEvent`：带时间戳的结构化商品交互事件，供 TIGER 构建用户序列。
- LangGraph State：当前请求和会话上下文。

Amazon Reviews 的 rating / review 不能伪装为 click / purchase。模拟事件必须标记 `is_synthetic=true`，默认不进入真实评测序列。

### 4.3 Planner 唯一 Schema

Initial Planner 和 Replanner 共用模块二的 `RecommendationPlan`：`intent`、`constraints.hard/soft`、`preferences`、`retrieval.strategy/tiger/e5_faiss`、`ranking`、`clarification`、`assumptions`、`change_summary`。不要另外创建顶层 `retrieval_strategy`、`tiger`、`e5_faiss` 等不兼容字段。

### 4.4 约束、RRF 与预算

- 硬约束由唯一 Python evaluator 执行，状态为 `pass` / `fail` / `unknown`；unknown 不算满足。
- E5 多 Query 先内部 RRF，聚合后的 E5 排名再与 TIGER 排名跨路 RRF；不要混加原始分数。
- Planner 选择预算档位；Controller 管理剩余 Replan、检索、候选数和 API 调用预算。
- 预算耗尽强制 STOP，但不代表质量达标。

## 5. 第一阶段启动任务：仓库审计，不要直接大规模编码

请先按顺序完成以下工作。

### A. 仓库盘点

- 输出目录树。
- 检查现有代码、配置、文档、数据、模型和索引。
- 检查 Python 版本、`pyproject.toml`、requirements 和 lock 文件。
- 检查 Git 状态和用户未提交改动。
- 不覆盖、删除或格式化与本任务无关的文件。

### B. 六份文档核对

逐份阅读六份设计文档，核对：
- 技术选型是否一致。
- `product_id` / `parent_asin` 映射是否明确。
- Mem0 / InteractionHistory / Session Context 是否分离。
- Planner `RecommendationPlan` 是否统一。
- E5 内部 RRF 和跨路径 RRF 是否按两级执行。
- 硬约束是否只有一套 Python 实现。
- Judgment / Reranker / Controller / Generator 职责是否分离。
- BudgetState、Clarification 和 Memory Update 工作流是否完整。

建立差异表：

| 模块 | 文档要求 | 仓库现状 | 缺失内容 | 风险 |
|---|---|---|---|---|

### C. 可行性检查

重点检查：
1. TIGER 是否有兼容当前商品数据和 Semantic ID 配置的实现 / 权重。
2. Amazon Reviews 2023 实际字段、品类文件、时间戳和数据规模。
3. `parent_asin` 到 `product_id` 的稳定映射。
4. E5 checkpoint、Query / Passage 前缀和向量维度。
5. FAISS 索引构建和持久化。
6. Qwen3-Reranker-0.6B 推理接口、输入长度和硬件开销。
7. DeepSeek API 模型 ID、结构化输出能力、密钥配置。
8. Mem0 OSS、Qdrant、SQLite 的部署方式和 SDK 版本。

未验证事项必须标记为未验证，不要猜测。

### D. 审计交付

先输出：
1. 当前项目状态摘要。
2. 目录树和文档版本清单。
3. 文档与代码差异表。
4. 未定义的 Pydantic Schema / API 接口。
5. TIGER 可行性和资源风险。
6. 按依赖顺序排列的实施计划。
7. 每阶段工时、交付物、验收标准和阻塞问题。

总工作量约 120 人时。优先保证真实、可复现的端到端链路，不要一开始扩展过多高级功能。

## 6. 审计通过后的建议实施顺序

### Phase 0：工程骨架与数据契约

目录、依赖、配置、`.env.example`、Pydantic Schema、日志、trace、pytest。

验收：项目可安装、可导入，基础测试通过。

### Phase 1：商品目录、ID 映射和 E5 召回

建立 Product Catalog；映射 `parent_asin` → `product_id`；生成商品文本；E5 编码；FAISS 建库；多 Query E5 内部 RRF。

验收：真实 Query 能返回真实 `product_id`、排名和来源。

### Phase 2：TIGER 阶段 B 与双路融合

建立 Semantic ID、InteractionHistory 用户序列、自回归生成和解码；实现 Candidate Fusion、两级 RRF、唯一硬约束 evaluator。

验收：真实历史序列可生成 Semantic ID 并映射回商品；E5 / TIGER 候选能合并；unknown 不当作 pass。若 TIGER 兼容实现或训练成本存在阻塞，先报告方案和成本，不要用 Mock 冒充。

### Phase 3：Reranker

接入 Qwen3-Reranker-0.6B，构建统一 Query / Product Document，批量精排，记录原始分数和耗时。

验收：模型真实输出排序；分数不解释成购买概率。

### Phase 4：Planner / Judgment / Controller

接入 DeepSeek Planner / Replanner；校验统一 Plan Schema；实现逐商品 Judgment；实现 LangGraph State、Clarification、REPLAN / STOP、多维预算和异常降级。

验收：根据真实判断与状态触发 Replan / Stop；硬约束不由 LLM 重复执行。

### Phase 5：Generator / Feedback / Memory

接入 DeepSeek Generator；实现商品证据约束；建立独立 Feedback Workflow；Mem0 更新语义记忆；SQLite 记录结构化事件。

验收：推荐只来自候选池；理由可追溯；无反馈不写入长期偏好；反馈写入后可检索。

### Phase 6：端到端评测

按时间切分数据；评估 TIGER、E5、融合、Reranker、Judgment、Controller、Generator 和 Memory；记录延迟、成本、失败样本。

验收：所有正式指标可由脚本和版本化数据复现，不含 Mock 数据。

## 7. 每个任务的执行方式

开始任务前，先说明任务理解、涉及模块、拟修改文件和接口影响。小步提交，不做无关重构。每个任务完成后：

- 列出新增 / 修改 / 删除文件。
- 说明 Schema 与接口变化。
- 给出实际执行的测试命令和结果。
- 列出未运行测试及原因。
- 更新文档和实现状态。
- 说明未解决风险和下一步。

不得虚构代码、测试通过状态、实验指标或模型能力。

## 8. 现在开始

**现在只执行仓库审计，不要立即大规模生成代码。** 请先输出目录树、文档版本、现有实现映射、关键接口缺口、TIGER 可行性与实施计划。遇到架构冲突先询问项目负责人。审计结果确认后再进入 Phase 0。
