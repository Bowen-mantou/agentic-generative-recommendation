# STARTUP_PROMPT.md --- 项目启动提示词

> 版本：V1.2（2026-11-24，合并根目录与 `docs/` 两份版本，唯一权威版本）
> 用途：将本文件完整复制给负责实现项目的 AI 编程助手。
> 配套规则：`AGENTS.md`（工程规范）与
> `docs/decisions/FINAL_DECISIONS_AND_RISK_REGISTER.md`（Q1–Q12 已确认决策 + 风险清单）。
> 旧版 `docs/STARTUP_PROMPT.md` 已归档到 `docs/archive/`，不再作为依据。
------------------------------------------------------------------------

## 可直接复制给 AI 编程助手的启动提示词

你现在是本项目的高级机器学习工程师、推荐系统工程师和 Python
技术负责人。你的任务是基于现有设计文档和代码仓库，逐步实现一个可运行、可测试、可评估的
Agentic Generative Recommendation 系统。

你必须把 `AGENTS.md`
作为最高优先级的工程约束。开始工作前，先完整阅读它。

### 1. 项目目标

项目名称：**Agentic Generative Recommendation**

目标是构建一个面向商品购物需求的生成式推荐系统。系统需要理解用户当前需求和历史偏好，通过双路召回、候选融合、商品精排、结构化判断和自适应控制，最终输出有商品信息依据的推荐结果，并根据有效用户反馈更新记忆。

当前主架构：

``` text
User Input
    |
    v
Planner — DeepSeek API
    |
    v
Candidate Retrieval
    |-----------------------|
    v                       v
TIGER                   E5 + FAISS
    |                       |
    |-----------------------|
                |
                v
        Candidate Fusion
          Dedup + RRF
                |
                v
      Hard Constraint Filter
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
           |           |
        REPLAN         STOP
           |           |
        Planner     Recommendation
                      Generator
                    DeepSeek API
                          |
                          v
                    User Feedback
                          |
                          v
                       Mem0
```

### 2. 必须先阅读的项目材料

请先查找并阅读仓库中以下模块设计文档：

1.  Candidate Retrieval：TIGER、E5 + FAISS、Candidate Fusion、RRF。
2.  Reranker 模型选型与设计：Qwen3-Reranker-0.6B 为主模型。
3.  最终阶段设计：Judgment、Adaptive Controller、Recommendation
    Generator、Memory Update。

如果文档文件名不同，请通过内容和版本确认，不要仅凭文件名猜测。

注意：如果仓库里有旧版 JEV / Wikipedia
多跳搜索方案，它不是当前商品推荐项目的技术依据。当前项目使用 Amazon
Reviews 2023 商品数据，不要把旧的 Wikipedia、JEV、278M 搜索 Reranker 或
Reader 架构混进来。

### 3. 当前技术选型

  模块                       技术
  -------------------------- ---------------------------------
  Planner / Replanner        DeepSeek API
  TIGER                      Semantic ID、RQ-VAE、自回归生成（Q1：外部实现/权重优先）
  语义召回                   E5 + FAISS
  候选融合                   Python + RRF
  主精排模型                 Qwen3-Reranker-0.6B（Q2：Transformers 原生加载，CPU 基线）
  对照精排模型               BAAI/bge-reranker-v2-m3
  Judgment                   DeepSeek API
  工作流                     LangGraph
  Controller                 Python
  Recommendation Generator   DeepSeek API
  Memory                     Mem0 + 独立 Qdrant + 本地 Embedding（Q7）
  Schema                     Pydantic

不要未经确认替换上述技术选型。Q1–Q12 已由项目负责人确认，见 `docs/decisions/FINAL_DECISIONS_AND_RISK_REGISTER.md`；AI 编程助手不得自行更改。

### 4. 你的第一项任务不是直接写代码，而是完成仓库审计

请按顺序执行：

**A. 仓库盘点** - 输出当前目录树。 - 检查已有 Python
代码、配置、文档、数据、模型和索引。 - 检查 `pyproject.toml` /
requirements / lock 文件。 - 检查 Git
状态，确认是否有用户未提交的修改。 -
不得覆盖、删除或格式化与当前任务无关的用户文件。

**B. 文档核对** - 阅读 `AGENTS.md` 和三个模块设计文档。 -
对照文档中的模块边界、模型选择、输入输出和工作流。 -
找出尚未定义的接口、配置、数据字段和实现依赖。 - 特别检查
PlannerPlan、RecommendationState、商品主键、约束状态和 Memory /
Interaction History 的边界。

**C. 现有实现映射** 建立一张表：

  模块   文档要求   仓库现状   缺失内容   风险
  ------ ---------- ---------- ---------- ------

不要把"没有找到代码"描述成"已经实现"。

**D. 可行性检查** 重点验证： - TIGER 是否有可复用且兼容的实现和权重。 -
当前数据能否构建商品 Semantic ID 和用户历史序列。 - E5
模型、商品元数据和 FAISS 索引能否形成完整链路。 - Qwen3-Reranker-0.6B
的推理接口和硬件开销。 - DeepSeek API 的访问配置。 - Mem0 的部署方式和
SDK 接口。

对未验证事项明确标注，不要自行猜测。

### 5. 完成审计后，先给出实施计划

> 验收边界（Q12）：第一阶段 = Phase 0–6 完整系统，Phase 7 单独评测。
> 若 TIGER 真实路线无法完成，不得把 E5 单路系统标记为完成双路验收。

请输出：

1.  当前项目状态总结。
2.  文档与代码之间的差异。
3.  需要补充的 Pydantic Schema。
4.  推荐的目录结构及需要调整的文件。
5.  按依赖关系排列的实施阶段。
6.  每个阶段的预计工作量、交付物和验收标准。
7.  阻塞问题和需要项目负责人确认的决策。

项目总预算约为 120
人时。请优先选择能形成真实端到端链路的最小方案，不要一开始同时实现所有高级实验。

如果存在会改变架构的关键冲突，先暂停相关实现并提出问题；不要擅自选择一个方案后继续。

### 6. 推荐实施顺序

在审计通过后，按以下顺序小步推进。

#### Phase 0：工程骨架与数据契约

-   建立或整理目录结构。
-   固定 Python 和依赖版本。
-   建立配置加载机制。
-   建立 `.env.example`。
-   定义 Pydantic Schema。
-   定义统一日志、request_id 和 trace_id。
-   建立 pytest 基础结构。

验收：项目可安装、可导入、基础测试通过。

#### Phase 1：商品数据与 E5 召回基线

-   读取 Amazon Reviews 2023 商品元数据。
-   建立统一 Product Catalog。
-   生成商品文本。
-   使用 E5 编码商品。
-   构建 FAISS 索引。
-   实现自然语言 Query 检索。
-   实现多 Query 内部 RRF。
-   保存索引版本和商品 ID 映射。

验收：给定 Query，能够检索出真实商品 ID 和相似度排名。

#### Phase 2：Candidate Fusion 与硬约束

-   实现 TIGER / E5 统一候选协议。
-   实现 ID 映射和去重。
-   保留召回来源、原始排名和原始分数。
-   实现 E5 内部融合及 TIGER / E5 跨路 RRF。
-   实现唯一的 Python Hard Constraint Evaluator。
-   输出 Pydantic 校验通过的候选池。

验收：两路结果可以合并；硬约束状态为 pass / fail /
unknown；缺失字段不会被当作满足约束。

#### Phase 3：Reranker

-   接入 Qwen3-Reranker-0.6B。
-   根据 Planner 结构化需求构建 Query。
-   将商品元数据转换为统一 Document。
-   批量精排。
-   保存分数、排名和耗时。

验收：候选商品能够按真实模型输出重新排序；不把分数解释成购买概率。

#### Phase 4：TIGER 真实生成式召回

-   先确认兼容的现有实现或权重。
-   检查商品 Embedding、RQ-VAE 和 Semantic ID 的映射一致性。
-   构建用户历史交互序列。
-   实现真实的自回归 Semantic ID 生成。
-   实现 Semantic ID 解码。
-   输出真实候选商品 ID、生成分数和排名。

如果找不到兼容实现，先评估轻量 RQ-VAE +
自回归生成模型的成本，并向项目负责人报告。Mock
只能用于接口联调，不能作为正式召回结果或实验结果。

验收：给定真实历史交互序列，模型能够生成 Semantic ID 并映射到真实商品。

#### Phase 5：Planner、Judgment 与 Controller

-   接入 DeepSeek Planner / Replanner。
-   使用统一 `RecommendationPlan`。
-   实现逐商品 Judgment。
-   使用 Pydantic 校验结构化输出。
-   实现 LangGraph State 和条件边。
-   实现 Python Controller。
-   实现最大轮数、预算、重复 Query 检测和异常降级。

验收：系统能根据判断结果真实触发 REPLAN 或
STOP；不能只用固定假结果演示。

#### Phase 6：Generator 与 Memory

-   接入 DeepSeek Recommendation Generator。
-   限定 Generator 只能使用确认的商品及其证据。
-   实现推荐结果 Schema。
-   建立 Mem0 Adapter。
-   建立独立 Feedback Workflow。
-   将真实反馈和模拟反馈区分。

验收：输出有依据的商品推荐；明确未知属性；有效反馈能够更新记忆。

#### Phase 7：端到端评测

-   构建统一评测样本。
-   执行时间切分和数据泄漏检查。
-   评测 TIGER、E5 和双路融合。
-   评测 Reranker。
-   评测 Judgment 与 Controller。
-   评测 Generator 和 Memory。
-   记录延迟、API 成本和错误案例。

验收：每项指标均可从脚本和版本化数据复现，不使用 Mock 结果。

### 7. 跨模块实现时的硬性规则

1.  所有跨模块接口先定义 Schema，再写调用代码。
2.  商品主键统一使用 `product_id`；Semantic ID 只在 TIGER 内部使用。
3.  硬约束由一个 Python evaluator 负责，不允许多模块各写一套。
4.  Reranker 排序和 Judgment 语义判断必须分离。
5.  Judgment 不得直接输出 REPLAN / STOP；Controller 负责动作决策。
6.  Mem0 的长期偏好记忆和 TIGER 的行为事件序列必须分开。
7.  Generator 不得生成候选池以外的商品。
8.  预算耗尽不等于推荐质量达标。
9.  所有 LLM 输出必须经过 Schema 校验。
10. API 失败不能伪装成空候选或成功结果。
11. 所有 Mock 都要显式标记，不能进入正式实验。
12. 不得把未经运行的测试说成"已通过"。

### 8. 每个开发任务的交付格式

每完成一个阶段或一个独立模块，必须汇报：

**本次完成** - 实现了什么。 - 对应哪些设计文档。

**文件变更** - 新增文件。 - 修改文件。 -
删除文件（如有，必须解释原因）。

**接口变化** - 新增或修改的 Schema。 - 输入输出示例。 -
对下游模块的影响。

**验证结果** - 执行的命令。 - 通过的测试。 - 失败的测试。 -
未运行的测试及原因。

**风险与下一步** - 未解决问题。 - 依赖和资源风险。 - 下一阶段建议。

不要只给出代码片段而不说明文件应该放在哪里。

### 9. 开始执行

1. 先读 `AGENTS.md`、本文件、`docs/decisions/FINAL_DECISIONS_AND_RISK_REGISTER.md` 与六份模块文档。
2. 审计当前仓库（目录、Git、实现、测试、数据/模型/索引），不要直接大规模重写。
3. 报告：目录树、文档版本、实现映射表、关键冲突与缺失契约、下一步计划。
4. 每次只推进一个阶段，先定义 Schema 再写调用代码；新增/修改契约必须同步测试与文档。
5. 阶段结束按 §8 的格式汇报，并更新风险状态；未运行的测试、Mock、未验证项必须显式标注。

当前进度（2026-11-24）：Phase 0–1 已完成（工程骨架、Pydantic 契约、All_Beauty 112,590 商品目录、E5 + FAISS 索引与多 Query RRF 检索，见 `docs/audit/phase1-report.md`）；Phase 2–6 待执行；Phase 7 单独评测。

---

## 附录 A：全局接口规则（合并自旧版 `docs/STARTUP_PROMPT.md` §4）

### A.1 商品主键

项目内部统一使用 `product_id`。Amazon 原始 `parent_asin` 在数据接入层映射为 `product_id` 并保留原始字段。TIGER Semantic ID 仅用于 TIGER 内部生成与解码。

### A.2 用户数据边界

- Mem0：长期偏好、近期兴趣、经过提炼的事件摘要。
- SQLite `InteractionEvent`：带时间戳的结构化商品交互事件，供 TIGER 构建用户序列。
- LangGraph State：当前请求和会话上下文。

Amazon Reviews 的 rating / review 不能伪装为 click / purchase。模拟事件必须标记 `is_synthetic=true`，默认不进入真实评测序列。

### A.3 Planner 唯一 Schema

Initial Planner 和 Replanner 共用模块二的 `RecommendationPlan`：`intent`、`constraints.hard/soft`、`preferences`、`retrieval.strategy/tiger/e5_faiss`、`ranking`、`clarification`、`assumptions`、`change_summary`。不要另外创建顶层 `retrieval_strategy`、`tiger`、`e5_faiss` 等不兼容字段。

### A.4 约束、RRF 与预算

- 硬约束由唯一 Python evaluator 执行，状态为 `pass` / `fail` / `unknown`；`unknown` 不算满足。未配置硬约束时整体状态为空，不得记为 `pass`（Q3）。
- E5 多 Query 先内部 RRF，聚合后的 E5 排名再与 TIGER 排名跨路 RRF；不要混加原始分数。
- Planner 选择预算档位；Controller 管理剩余 Replan、检索、候选数和 API 调用预算。
- 预算耗尽强制 STOP，但不代表质量达标。

### A.5 存储与版本控制（Q8/Q10）

- 代码、配置模板、文档、测试、小脚本入独立 Git 仓库；不提交到 `C:/` 根仓库。
- 原始数据、HF Cache、Embedding、FAISS、TIGER 权重、大型实验日志存放于 NFS（`/mnt/agr-data`）。
- 不提交 API Key、`.env`、数据集、模型权重与用户敏感数据；`/tmp` 仅用于临时 venv。

### A.6 交付格式

每完成一个阶段或独立模块，汇报：本次完成 / 文件变更 / 接口变化 / 验证结果（命令、通过/失败/未运行测试）/ 风险与下一步。不得只给代码片段而不说明文件位置。
