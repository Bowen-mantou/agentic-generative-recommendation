# AGENTS.md — Agentic Generative Recommendation 项目工程规范

> 版本：V1.1（与六份技术设计文档统一）  
> 适用对象：所有 AI 编程助手和开发者  
> 项目预算：6 人 × 20 小时，合计约 120 人时

## 0. 最高优先级规则

开始任何代码修改前，必须阅读本文件、项目 README 和 `docs/` 下六份 V1.1 设计文档。不得凭旧聊天记录或模型经验自行改变已确定的架构。

当前项目是 **Agentic Generative Recommendation**，即面向商品购物需求的生成式推荐系统。Amazon Reviews 2023 是当前主要数据来源。本项目不是旧版 JEV / Wikipedia 多跳问答 / Reader 搜索项目。旧文件 `设计生成式推荐项目.txt` 仅作为历史背景，不作为当前实现依据。

若文档、代码或数据字段发生冲突：
1. 暂停受影响的改动并指出冲突。
2. 以本文件的全局决策和六份 V1.1 模块文档为当前设计基线。
3. 不得静默重命名字段、替换模型或删除旧实现。
4. 需要新增决策时，在 `docs/decisions/` 建立 ADR 并请求项目负责人确认。

## 0.1 已确认架构决策（2026-11-24，不可擅自更改）

项目负责人已确认 12 项架构决策，另有 D1–D11 实施决策；完整记录与风险清单见
`docs/decisions/FINAL_DECISIONS_AND_RISK_REGISTER.md`（§7.1）。本文件与六份模块文档不得与这些决策冲突；
AI 编程助手不得自行更改。技术验证失败时必须提交证据、影响分析、替代方案与人时/资源成本，
经负责人批准后才能更新决策记录与相关文档。

| ID | 决策主题 | 已确认结论 |
|---|---|---|
| Q1 | TIGER 路线 | **B→A（重解释）+ A3 Pilot（2026-11-24 确认）**：复用外部代码/结构/流程，用本项目 E5 向量与真实序列重训 RQ-VAE + 自回归；先 A1 Pilot（3×256，≥3）达到验收后再全量；禁止 Mock/随机 ID/规则映射 |
| Q2 | Reranker 加载与设备 | `Qwen3-Reranker-0.6B` 使用 Transformers 原生模型/tokenizer（不把 SentenceTransformers `CrossEncoder` 作为默认实现）；CPU 为第一阶段基线；`bge-reranker-v2-m3` 仅作对照 |
| Q3 | 硬约束 Schema | 整体三态 + `ConstraintCheck[]` 明细；无约束时 `constraint_checks=[]` 且整体状态为空/可空，不得伪装为已验证通过；不增加 `not_required` 第四态 |
| Q4 | E5 | 保留 `intfloat/e5-base-v2`（768 维、`query:` / `passage:` 前缀）；不因未验证的精度预期重建索引 |
| Q5 | 数据品类 | 第一阶段仅使用 `All_Beauty`；维护 `product_id` 与 `parent_asin`/`asin` 映射；缺失价格不得默认满足预算，必须 `unknown` |
| Q6 | Controller 校准 | Amazon 时间切分数据用于离线质量评测；人工/模拟审核集用于 REPLAN/STOP 评测；模拟数据必须标记，不得当作真实转化 |
| Q7 | Memory 部署 | Mem0 + 独立 Qdrant（持久化数据卷）+ 本地 Embedding；单 Collection + `user_id` 过滤；版本与向量维度锁定在配置中 |
| Q8 | Git 边界 | 建立独立项目 Git 仓库，不提交到 `C:/` 根仓库；不提交原始数据、模型权重、大型索引、API Key 与用户敏感数据 |
| Q9 | 启动提示词 | 根目录 `STARTUP_PROMPT.md` 为唯一权威版本；`docs/` 旧版归档并标注废弃 |
| Q10 | 存储 | 代码/配置模板/文档/测试入 Git；原始数据、HF Cache、Embedding、FAISS、TIGER 权重、大型实验日志放 NFS（`/mnt/agr-data`）；摘要/指标/报告入 Git |
| Q11 | MUSA | 第一阶段不接入 MUSA，使用 CPU 基线；MUSA 仅作为后续性能优化实验 |
| Q12 | 验收边界 | 第一阶段验收 = Phase 0–6 完整系统；Phase 7 单独评测；若 TIGER 真实路线无法完成，不得把 E5 单路系统标记为完成双路验收 |

## 1. 当前唯一技术基线

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
  Qwen3-Reranker-0.6B (frozen)
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
     Replanner   Generator
     DeepSeek    DeepSeek API
                   |
                   v
              User Feedback
                   |
          +--------+--------+
          |                 |
       Mem0 Store      SQLite InteractionEvent
     semantic memory    ordered behavior events
```

| 模块 | 固定方案 | 第一阶段训练要求 |
|---|---|---|
| User Memory | Mem0 OSS + 独立 Qdrant + SQLite（本地 Embedding，Q7） | 不训练记忆模型；使用成熟机制 |
| Interaction History | SQLite 结构化事件表 | 不训练；为 TIGER 提供按时间排序的序列 |
| Planner / Replanner | DeepSeek API，共用模型服务和统一 Plan Schema | 不训练 |
| TIGER | Semantic ID + RQ-VAE + 自回归生成 | 外部实现/权重优先（Q1）；自建为备用；阶段 B 必须真实生成 |
| E5 Retrieval | E5 + FAISS | 使用预训练 Encoder，不微调 |
| Candidate Fusion | Python + RRF | 不训练 |
| Hard Constraints | 唯一 Python Constraint Evaluator | 不训练 |
| Reranker | Qwen3-Reranker-0.6B（Transformers 原生加载，CPU 基线，Q2） | 第一阶段冻结；BGE-v2-m3 作对照 |
| Judgment | DeepSeek API | 不部署本地 Qwen3-8B，不训练 |
| Controller | LangGraph + Python | 不引入 Controller LLM |
| Recommendation Generator | DeepSeek API | 不训练 |
| Trace | JSONL + SQLite | 保存完整可追溯运行轨迹 |

**存储与版本控制（Q8/Q10）**：项目使用独立 Git 仓库，只提交代码、配置模板、文档、测试与小型脚本；
原始数据、HF Cache、Embedding、FAISS 索引、TIGER 权重和大型实验日志存放在
`/mnt/agr-data`（NFS），不得写入 Git；`/tmp` 仅用于临时 venv，不得作为产物的唯一存储位置。

**重要区分：Qwen3-Reranker-0.6B 是精排模型；Judgment 使用 DeepSeek API。不得将二者混淆。** 本项目不接入 JEV API。

第一阶段不进行 Planner、Judgment、Generator、E5 或 Reranker 的 SFT / LoRA。TIGER 阶段 B 需要真实 Semantic ID 生成和解码，不得以 Mock 冒充正式实验。阶段 A 的完整 TIGER 优化属于后续扩展。

## 2. 全局数据契约

### 2.1 商品主键

项目内部唯一商品主键为 `product_id`。Amazon 原始 `parent_asin` 在数据接入层映射为 `product_id` 并保留来源字段。TIGER 的 Semantic ID 只用于模型内部生成和解码。Planner、E5、Fusion、Reranker、Judgment 和 Generator 一律使用 `product_id`。

### 2.2 三类用户上下文必须分开

- **Mem0 User Memory**：长期偏好、近期兴趣、偏好冲突和经过提炼的事件摘要。
- **InteractionHistory**：SQLite 中带时间戳的结构化商品交互事件，TIGER 从这里构建用户序列。
- **Session Context**：当前请求和多轮对话状态，存放在 LangGraph State。

Amazon Reviews 2023 主要是评论 / 评分代理事件，不得描述或伪装为点击、浏览、加购日志。模拟数据必须带 `is_synthetic=true`，不得混入真实评测序列。

### 2.3 唯一 Planner Schema

Initial Planner 和 Replanner 必须使用模块二定义的 `RecommendationPlan`：

- `schema_version`
- `plan_id`, `parent_plan_id`, `plan_version`, `mode`
- `intent`
- `constraints.hard`, `constraints.soft`
- `preferences.current`, `preferences.historical`, `preferences.negative`, `preferences.priority`
- `retrieval.strategy`, `retrieval.tiger`, `retrieval.e5_faiss`
- `ranking`
- `clarification`
- `assumptions`, `change_summary`

禁止使用与此平行的顶层 `retrieval_strategy` / `tiger` / `e5_faiss` 字段。Pydantic 是接口校验的权威定义。

### 2.4 硬约束状态

硬约束由一个 Python Constraint Evaluator 执行，结果为：
- `pass`：字段存在且满足约束。
- `fail`：明确违反约束。
- `unknown`：缺少字段或无法验证。

只有所有必要硬约束均 `pass` 的商品可计入有效约束通过数；`unknown` 不得当作 `pass`，可保留用于诊断和 Replan，但不能宣称满足约束。未配置任何硬约束时 `constraint_checks=[]`、整体状态为空（不得记为 `pass`）；这类商品仍进入有效候选池，但在汇总中单独记录 `not_configured_count`（决策 Q3）。

### 2.5 两级 RRF

1. E5 多 Query 结果先在 E5 路径内部按排名去重并 RRF。
2. 得到唯一 E5 排名后，再与 TIGER 排名进行跨路径 RRF。
3. 不直接相加不同模型的原始相似度 / 生成分数。

### 2.6 预算

Planner 只选择 `standard` / `expanded` 预算档位。全局剩余 Replan 次数、检索调用次数、候选数量和 API 调用次数由 Controller 的 `BudgetState` 管理。预算耗尽是强制停止，不等于质量达标。

## 3. 模块职责边界

### Planner

理解需求、融合 Mem0、提取硬约束和软偏好、生成 E5 Query、选择召回预算档位、输出精排 Query；Replanner 根据执行反馈调整计划。不得调用 FAISS、生成商品 ID 或改写原始硬约束。

### TIGER

从 SQLite InteractionHistory 获取按时间排序的商品事件序列，预测 Semantic ID 并解码为 `product_id`。不从 Mem0 自由文本拼接行为序列。

### E5 + FAISS

使用 Planner 的一个或多个自然语言 Query 检索商品向量。E5 内部多 Query 先 RRF，再参与跨路径融合。

### Candidate Fusion

统一 `product_id`、去重、保留召回来源和各路排名、两级 RRF、调用唯一硬约束校验器，输出候选池。不得生成推荐理由。

### Reranker

Qwen3-Reranker-0.6B 根据需求与商品文本输出原始相关性分数和排名。分数不是购买概率。不得代替硬约束校验或 Judgment。

### Judgment

使用 DeepSeek API，对每个候选商品独立输出结构化判断：Preference Fit、Attribute Evidence、Match Assessment、Uncertainty。可对多个商品做有并发上限的批量请求，但每个商品必须有独立结果。不得决定 REPLAN / STOP。

### Controller

LangGraph + Python 汇总约束状态、Judgment、有效候选数、多样性、轮数和预算，决定 REPLAN / STOP。不得额外引入 Controller LLM。

### Generator

DeepSeek API 只对 Controller 选定的商品生成推荐表达和有证据的理由。不得生成候选池之外的商品、编造属性或擅自改变 Reranker 排序。

### Memory Update

只有有效用户反馈 / 行为才触发。语义偏好写入 Mem0；结构化商品事件写入 SQLite。没有点击不等于不喜欢；模型推断不能未经确认变成用户事实；模拟数据必须隔离。

## 4. 文件与目录规范

```text
project-root/
├── AGENTS.md
├── STARTUP_PROMPT.md
├── README.md
├── pyproject.toml
├── .env.example
├── configs/
├── docs/
│   ├── architecture/
│   ├── modules/
│   ├── decisions/
│   └── experiments/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── samples/
├── artifacts/
│   ├── embeddings/
│   ├── faiss/
│   ├── semantic_ids/
│   └── checkpoints/
├── src/recommender/
│   ├── api/
│   ├── planner/
│   ├── memory/
│   ├── interaction_history/
│   ├── retrieval/tiger/
│   ├── retrieval/e5/
│   ├── fusion/
│   ├── constraints/
│   ├── reranking/
│   ├── judgment/
│   ├── controller/
│   ├── generation/
│   ├── schemas/
│   ├── evaluation/
│   ├── observability/
│   └── utils/
├── scripts/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── notebooks/
└── runs/
```

若现有仓库已有合理结构，先复用并做最小调整，不得为了匹配目录树而大规模搬迁。

- 生产代码放 `src/`，核心逻辑不得长期只存在 Notebook。
- 配置放 `configs/`；密钥只从环境变量读取。
- `.env`、数据集、模型权重、FAISS 索引、Embedding、运行日志默认加入 `.gitignore`。
- 原始数据、中间数据和处理后数据分开。
- 文档使用清晰稳定文件名和版本号，不创建 `final_final2`。

Python 代码要求：类型标注、Docstring、单一职责、依赖注入、无裸 `except`、正式日志不用 `print`、不在 import 时初始化模型或请求 API。

依赖使用 `pyproject.toml` 和 lock 文件固定版本。模型名、API 参数、RRF 常数、Top-K、预算、超时和并发数均配置化。

## 5. API 与异常处理

DeepSeek Planner、Judgment、Generator 使用独立 Prompt、独立 Pydantic Schema、独立超时 / 重试 / 预算统计；可以共用底层 API Client，但不得合并成一个 Agent。

所有外部调用必须具备有限超时、有限重试与退避、并发上限、错误类型区分、request_id / trace_id、Token / 延迟日志和敏感信息脱敏。API 失败不得伪装成空候选或成功结果。

## 6. 测试与实验

- Unit：Schema、ID 映射、RRF、去重、约束状态、Controller 分支、Memory 准入规则。
- Integration：E5 + FAISS、TIGER 生成解码、Fusion → Reranker、Reranker → Judgment、LangGraph REPLAN / STOP、Generator、Mem0 Adapter、SQLite Event Store。
- E2E：正常推荐、触发 Replan、预算耗尽、澄清挂起恢复、API 失败、无有效候选。
- 单元测试默认使用 Mock；真实 API 测试显式运行。Mock 数据必须标记，不能进入正式 Recall / NDCG / HitRate 等实验。
- 按时间切分用户交互，防止未来商品、评论、Memory 和模拟对话泄漏到 Query。
- 每次实验记录代码 commit、配置、数据版本、模型版本、随机种子、指标、延迟、成本和失败样本。

## 7. AI 编程助手工作流程

每个任务必须按以下顺序：
1. 阅读本文件及相关设计文档。
2. 查看目录、Git 状态、已有实现和配置。
3. 找到并复用已有 Schema / 接口 / 测试。
4. 先报告任务理解、影响模块和拟修改文件。
5. 跨模块变更先明确 Schema 变化。
6. 小步实现，不整体重写。
7. 添加 / 更新测试并实际运行。
8. 更新 README、文档和实现状态。
9. 汇报修改文件、命令、测试结果、未解决问题。

禁止：擅自换模型或框架；把旧 JEV / Wikipedia 项目混入；将 Judgment 变成第二个 Reranker；让 LLM 做可确定性 Python 比较；让 Generator 编造商品；混用 Mem0 和 TIGER 行为序列；删除用户文件；伪造测试和实验结果。

## 8. 未验证事项

以下事项必须通过仓库审计或小型预实验确认，不能猜测：
1. TIGER 外部权重/兼容实现：**已核查无端到端权重**；A1/B1/B2 Pilot 均未通过 D11（[D1_B1_B2_GATE_REPORT.md](docs/phase4/D1_B1_B2_GATE_REPORT.md)），Phase 4 未完成、未全量训练。
2. Amazon Reviews 2023 的实际品类文件、字段、规模和时间戳格式。（已验证：All_Beauty metadata 212,990,142 B / 112,590 商品；reviews 326,611,506 B）
3. 商品元数据如何稳定映射 `parent_asin` 与 `product_id`。（已验证：metadata 含 `parent_asin`/`main_category`/`details`；映射在接入层完成）
4. E5 具体 checkpoint、前缀格式和向量维度。（已确认：`intfloat/e5-base-v2`，768 维，`query:`/`passage:`；索引已建）
5. Qwen3-Reranker-0.6B 推理接口、输入长度和硬件开销。（已验证：Transformers 原生 CPU，[reranker-smoke.md](docs/audit/reranker-smoke.md)；Phase 3 已实现）
6. DeepSeek 具体模型 ID、API 参数和 JSON 输出校验方式。（已验证模型列表：`deepseek-flash`、`deepseek-v4-pro`；结构化输出待 Phase 5 验证）
7. Mem0 OSS / Qdrant / SQLite 的部署方式和 SDK 版本。（部分验证：Qdrant v1.20.2 + mem0ai 2.2.1 + multilingual-e5-base，见 `docs/phase6/D9_MEMORY_DEPLOYMENT.md`；Memory.add 待 Phase 6）
8. Controller 阈值和预算值如何由验证集确定。（待 Phase 5/7；当前值均标注未校准）

## 9. Definition of Done

模块完成必须满足：Schema 明确、核心逻辑真实实现、异常路径覆盖、测试通过、配置登记、日志可追踪、文档更新、Mock 与真实结果区分、未验证假设明确标记。

**总原则：先保证数据契约一致和真实端到端链路，再优化模型效果；先做可归因实验，再增加复杂度。**

## 10. 决策与变更管理

1. `docs/decisions/FINAL_DECISIONS_AND_RISK_REGISTER.md` 中 Q1–Q12 已确认，AI 编程助手不得自行更改。
2. 技术验证发现问题时，先建立风险记录或 ADR；不得直接替换核心模型、服务或接口。
3. 若决策确实不可行，必须提交失败证据、受影响阶段、替代方案、人时/资源/迁移成本及实验可比性影响，经负责人批准后更新决策记录。
4. 每个阶段结束后更新风险状态（`Open` / `In Progress` / `Mitigated` / `Accepted` / `Closed`）；`Mitigated` 不代表风险消失。
5. 不得把“代码能运行”当成“模型有效”，不得把 Mock 联调当成真实模型验收。
