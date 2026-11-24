# 模块二：Recommendation Planner 推荐规划器

## 基于 DeepSeek API 的生成式推荐规划与动态重规划机制

**项目名称：** Agentic Generative Recommendation

**模块名称：** Recommendation Planner

**文档版本：** V1.1（统一 Plan Schema 修订版）

**基础模型：** DeepSeek API

**工作流框架：** LangGraph

**输出形式：** Structured JSON Recommendation Plan

---

# 一、模块定位与设计目标

Recommendation Planner 是整个生成式推荐系统的核心规划模块，负责理解用户当前购物需求，结合 User Memory 中的历史偏好，生成结构化、可执行的推荐计划，并根据后续推荐执行结果动态调整推荐策略。

本模块采用 DeepSeek API，不进行本次课程设计范围内的模型训练。

与传统推荐系统直接根据用户行为生成商品列表不同，本项目引入生成式 Planner，使系统能够在推荐之前理解用户需求、整合用户偏好，并自主规划商品召回和精排策略。

核心目标：

1. 理解用户当前的自然语言购物需求。
2. 结合 User Memory，形成完整的用户推荐上下文。
3. 提取商品类别、使用场景、硬性约束和软性偏好。
4. 根据用户需求动态规划 TIGER 和 E5 + FAISS 的召回策略。
5. 生成适用于后续 Reranker 的排序目标。
6. 根据召回、精排和 Judgment 的执行反馈重新规划推荐策略。

本模块的核心研究问题：

**生成式 Planner 能否通过理解用户需求、融合历史记忆和动态调整召回策略，提高生成式推荐系统的推荐质量与检索效率？**

# 二、整体设计方案

Planner 采用混合式动态规划机制，由 Initial Planner 和 Replanner 两种工作模式组成。

两者使用同一个 DeepSeek 模型、同一套 API 接口和统一的输出 Schema，通过不同的 Mode 区分工作方式。

## 2.1 整体架构

```text
              User Input
                   |
                   v
              User Memory
                   |
                   v
             Session History
                   |
                   v
           System Capabilities
                   |
                   v
        Recommendation Planner
              DeepSeek API
                   |
             Check Mode
             /         \
            /           \
       Initial Mode   Replan Mode
            \           /
             \         /
                   |
                   v
       Structured Recommendation Plan
                   |
                   v
           Plan Validation
              Pydantic
                   |
                   v
           LangGraph Execution
                   |
          +--------+--------+
          |                 |
          v                 v
        TIGER            E5 + FAISS
       生成式召回          向量召回
          |                 |
          +--------+--------+
                   |
                   v
           Candidate Fusion
                   |
                   v
               Reranker
                   |
                   v
               Judgment
                   |
                   v
              Controller
                   |
             +-----+-----+
             |           |
           REPLAN       STOP
             |           |
             v           v
        Replanner    Final Generator
             |
             v
       New Recommendation Plan
             |
             v
       下一轮 LangGraph 执行
```

## 2.2 设计原则

采用以下设计原则：

1. **生成式规划：** 使用 LLM 理解自然语言需求并生成推荐计划。
2. **动态召回：** 根据用户需求和历史偏好调整两路召回的侧重点。
3. **结构化输出：** 使用统一 JSON Schema，避免自然语言计划难以执行。
4. **规划与执行分离：** Planner 只生成计划，LangGraph 负责执行。
5. **规划与控制分离：** Planner 负责制定策略，Python Controller 决定是否重新规划。
6. **统一模型：** Initial Planner 和 Replanner 共用 DeepSeek API。
7. **可追踪性：** 保存每次规划输入、输出和计划变化，为后续模型训练积累轨迹数据。

# 三、Planner 输入设计

Planner 接收四类核心输入。TIGER 所需的原始、按时间排序的 `InteractionHistory` 由执行层从 SQLite 独立读取；Planner 可以读取其统计摘要，但不得把 Mem0 自由文本记忆当作 TIGER 行为序列。

| 输入 | 具体内容 | 来源 |
|---|---|---|
| Current Request | 用户当前购物需求 | User Input |
| User Memory | 长期偏好、近期兴趣、相关事件摘要 | Mem0 |
| Session History | 当前会话上下文和历史反馈 | LangGraph State |
| System Capabilities | 可用召回模块、商品字段、执行预算 | System Config |

## 3.1 Current Request

用户当前提出的自然语言购物需求。

示例：

```text
我想买一款适合日常使用的面霜，
预算在30美元以内，希望质地清爽，
最好没有明显香味。
```

Planner 需要理解用户的商品类别、使用场景和具体需求。

## 3.2 User Memory

从前一个模块的 Mem0 中获取用户相关记忆。

示例：

```json
{
  "long_term_preferences": [
    "fragrance_free",
    "lightweight"
  ],
  "short_term_interests": [
    "moisturizing"
  ],
  "negative_preferences": [
    "strong_fragrance"
  ]
}
```

Planner 需要结合当前需求与历史记忆，但不能让历史推断覆盖用户当前明确表达的要求。

偏好优先级：

1. 当前用户明确需求。
2. 用户明确表达的长期偏好。
3. 多次历史行为推断的偏好。
4. 单次历史行为推断的偏好。

历史偏好默认作为软性排序信号，不自动转化为硬性过滤条件。

## 3.3 Session History

保存当前会话中的补充信息。

例如：

```json
{
  "previous_user_messages": [
    "我更看重保湿效果，不需要特别高端的品牌。"
  ]
}
```

用于理解用户在多轮交互中不断补充和修改的需求。

## 3.4 System Capabilities

描述当前系统实际支持的功能。

示例：

```json
{
  "retrieval_modules": [
    "TIGER",
    "E5_FAISS"
  ],
  "available_product_fields": [
    "title",
    "category",
    "price",
    "brand",
    "features",
    "description"
  ],
  "reranker_available": true,
  "judgment_available": true
}
```

Planner 必须根据实际能力制定计划，不得生成系统无法执行的操作。

# 四、Planner 输出设计

Planner 采用完整结构化 JSON 输出，统一包含六个核心部分。

## 4.1 Intent Understanding：用户需求理解

提取用户的购物意图、商品类别和使用场景。

```json
{
  "intent": "product_recommendation",
  "category": "Beauty",
  "product_type": "moisturizer",
  "usage_scenario": "daily_use"
}
```

## 4.2 Constraint Extraction：约束提取

区分硬性约束与软性偏好。

**Hard Constraints：**

用户明确提出且必须满足的条件，例如价格上限、商品类别等。

**Soft Preferences：**

用户希望尽可能满足的条件，例如质地、品牌、外观和历史偏好。

示例：

```json
{
  "hard_constraints": [
    {
      "field": "price",
      "operator": "lte",
      "value": 30,
      "unit": "USD",
      "source": "current_request"
    }
  ],
  "soft_preferences": [
    {
      "attribute": "lightweight",
      "source": "current_request",
      "priority": "high"
    },
    {
      "attribute": "fragrance_free",
      "source": "user_memory",
      "priority": "medium"
    }
  ]
}
```

硬性约束最终由 Python 规则执行，Planner 负责提取和表达约束，不直接负责商品过滤。

## 4.3 Preference Fusion：用户偏好融合

将当前购物需求与历史 Memory 结合，生成统一的推荐偏好。

示例：

```json
{
  "current_preferences": [
    "lightweight",
    "moisturizing"
  ],
  "historical_preferences": [
    "fragrance_free"
  ],
  "negative_preferences": [
    "strong_fragrance"
  ],
  "priority": [
    "price",
    "moisturizing",
    "lightweight",
    "fragrance_free"
  ]
}
```

当前需求优先，历史偏好作为个性化补充。

## 4.4 Retrieval Planning：召回策略规划

这是 Planner 与生成式推荐系统结合的核心部分。

系统使用两种召回方式：

- TIGER：根据用户历史商品交互序列生成 Semantic ID，实现生成式商品召回。
- E5 + FAISS：根据自然语言购物需求和用户偏好进行语义向量召回。

Planner 负责决定两种召回的侧重点、预算等级和 E5 查询内容。

第一版设置三种召回策略：

| 策略 | TIGER | E5 + FAISS | 适用场景 |
|---|---|---|---|
| Hybrid Balanced | 标准预算 | 标准预算 | 普通购物需求 |
| Behavior Priority | 扩展预算 | 标准预算 | 更重视历史行为偏好 |
| Semantic Priority | 标准预算 | 扩展预算 | 更重视当前自然语言需求 |

其中，标准预算和扩展预算由 Python 配置映射到实际候选数量。

示例：

```json
{
  "retrieval": {
    "strategy": "semantic_priority",
    "tiger": {
      "enabled": true,
      "candidate_budget": "standard"
    },
    "e5_faiss": {
      "enabled": true,
      "queries": [
        "lightweight daily moisturizer",
        "fragrance-free moisturizing cream"
      ],
      "candidate_budget": "expanded"
    }
  }
}
```

以上仅展示统一 `RecommendationPlan` 中的 `retrieval` 子对象；实际 API 响应必须包含第 6.2 节定义的完整顶层 Schema，不得将 `retrieval_strategy`、`tiger` 或 `e5_faiss` 放在顶层。

第一版默认两路召回均保留基础召回能力，Planner 主要调整预算侧重点。

如果缺少有效用户行为序列，TIGER 由执行层降级处理，不允许 Planner 虚构商品 ID 或 Semantic ID。

## 4.5 Ranking Guidance：精排目标

Planner 根据用户需求生成精排查询表达和重点关注的商品属性。

示例：

```json
{
  "rerank_query": "Find daily moisturizers under $30 with lightweight texture and strong moisturizing properties, preferably fragrance-free.",

  "priority_attributes": [
    "price",
    "moisturizing",
    "lightweight",
    "fragrance_free"
  ]
}
```

Planner 不直接计算商品排序分数。

实际排序仍由预训练 Cross-Encoder Reranker 完成。

## 4.6 Clarification：需求澄清

当用户需求缺少必要信息时，Planner 可以请求补充。

示例：

```json
{
  "required": false,
  "question": null,
  "reason": null
}
```

如果需求过于模糊：

```json
{
  "required": true,
  "question": "你主要想购买日常保湿面霜，还是针对特定皮肤问题的面霜？",
  "reason": "当前商品用途不够明确"
}
```

非必要信息缺失时，不应频繁追问，可以使用合理默认值并记录假设。

# 五、Initial Planner 与 Replanner

两种模式共用同一个 DeepSeek 模型、API Client 和统一输出 Schema。

通过不同的 Mode 使用不同的 Prompt 和输入模板。

## 5.1 Initial Planner

触发时机：用户首次提出购物需求。

输入：

- Current Request。
- User Memory。
- Session History。
- System Capabilities。

主要职责：

1. 理解用户需求。
2. 提取硬性约束与软性偏好。
3. 融合历史用户记忆。
4. 决定召回策略。
5. 生成 E5 检索 Query。
6. 生成 Reranker 使用的查询表达。
7. 判断是否需要需求澄清。

输出：完整初始 Recommendation Plan。

## 5.2 Replanner

触发时机：Python Controller 判断当前推荐结果不满足目标，需要调整策略。

Replanner 不重新开始整个推荐任务，而是在保留原始需求和约束的前提下，根据上一轮执行反馈修正推荐计划。

主要职责：

1. 分析上一轮召回结果。
2. 识别推荐质量不足的原因。
3. 判断是否需要调整 TIGER 与 E5 的召回侧重点。
4. 生成新的语义检索 Query。
5. 调整召回预算。
6. 保留原始硬性约束。
7. 输出完整的新 Recommendation Plan。

## 5.3 Replanner 输入

由 Controller 汇总执行结果，形成结构化反馈。

示例：

```json
{
  "mode": "replan",

  "original_request": {
    "category": "Beauty",
    "product_type": "moisturizer",
    "max_price": 30
  },

  "previous_plan": {
    "retrieval": {
      "strategy": "hybrid_balanced",
      "e5_faiss": {
        "queries": ["daily moisturizer under $30"]
      }
    }
  },

  "execution_feedback": {
    "retrieved_candidates": 100,
    "after_deduplication": 72,
    "constraint_satisfied": 65,
    "judgment_passed": 28,
    "missing_attributes": [
      "fragrance_free"
    ],
    "diversity_issue": false
  },

  "controller_feedback": {
    "reason": "INSUFFICIENT_ATTRIBUTE_EVIDENCE",
    "suggested_direction": "SEARCH_FOR_FRAGRANCE_FREE_PRODUCTS"
  },

  "budget_state": {
    "remaining_replan_rounds": 1,
    "remaining_retrieval_calls": 2,
    "remaining_api_calls": 8
  }
}
```

不需要把所有候选商品的完整数据传给 Replanner，只提供必要的统计信息、代表性失败案例和执行反馈。

## 5.4 Replanner 输出

示例：

```json
{
  "schema_version": "1.0",
  "plan_id": "plan_002",
  "parent_plan_id": "plan_001",
  "plan_version": 2,
  "mode": "replan",
  "intent": {
    "intent": "product_recommendation",
    "category": "Beauty",
    "product_type": "moisturizer",
    "usage_scenario": "daily_use"
  },
  "constraints": {
    "hard": [
      {"field": "price", "operator": "lte", "value": 30, "unit": "USD", "source": "current_request"},
      {"field": "category", "operator": "eq", "value": "Beauty", "source": "current_request"}
    ],
    "soft": [
      {"attribute": "lightweight", "source": "current_request", "priority": "high"},
      {"attribute": "fragrance_free", "source": "user_memory", "priority": "medium"}
    ]
  },
  "preferences": {
    "current": ["lightweight", "daily_use"],
    "historical": ["fragrance_free"],
    "negative": ["strong_fragrance"],
    "priority": ["price", "lightweight", "fragrance_free"]
  },
  "retrieval": {
    "strategy": "semantic_priority",
    "tiger": {"enabled": true, "candidate_budget": "standard"},
    "e5_faiss": {
      "enabled": true,
      "queries": ["fragrance-free daily moisturizer", "unscented lightweight moisturizing cream"],
      "candidate_budget": "expanded"
    }
  },
  "ranking": {
    "rerank_query": "Daily moisturizer under $30, prioritize lightweight texture and fragrance-free evidence.",
    "priority_attributes": ["price", "lightweight", "fragrance_free"]
  },
  "clarification": {"required": false, "question": null, "reason": null},
  "assumptions": [],
  "change_summary": [
    "增加无香型商品的语义检索",
    "提高 E5 召回预算",
    "保留 TIGER 基础召回",
    "保留原始硬性约束"
  ]
}
```

该示例与第 6.2 节统一 Schema 完全一致。Replanner 不输出新的预算计数器；剩余调用次数由 Controller 的 BudgetState 管理。

Replanner 输出完整的新计划，而不是只返回需要修改的几个字段。

除非用户明确修改需求，否则 Replanner 不得擅自修改原始硬性约束。

# 六、统一接口与数据结构

## 6.1 统一调用接口

采用统一 Planner 接口：

```python
async def recommendation_planner(
    mode: str,
    input_data: dict
) -> dict:

    if mode == "initial":
        prompt = build_initial_prompt(input_data)

    elif mode == "replan":
        prompt = build_replan_prompt(input_data)

    else:
        raise ValueError("Unsupported planner mode")

    result = await deepseek_client.generate(
        prompt=prompt
    )

    validated_plan = validate_plan(result)

    return validated_plan
```

实际开发时，模型调用、Prompt 构建和 Schema 校验应分别封装，避免业务逻辑耦合。

## 6.2 统一 Plan Schema

Initial Planner 和 Replanner 使用相同的核心输出结构。

```json
{
  "schema_version": "1.0",
  "plan_id": "plan_001",
  "parent_plan_id": null,
  "plan_version": 1,
  "mode": "initial",
  "intent": {
    "intent": "product_recommendation",
    "category": "Beauty",
    "product_type": "moisturizer",
    "usage_scenario": "daily_use"
  },
  "constraints": {
    "hard": [
      {"field": "price", "operator": "lte", "value": 30, "unit": "USD", "source": "current_request"}
    ],
    "soft": [
      {"attribute": "lightweight", "source": "current_request", "priority": "high"}
    ]
  },
  "preferences": {
    "current": ["lightweight", "daily_use"],
    "historical": ["fragrance_free"],
    "negative": ["strong_fragrance"],
    "priority": ["price", "lightweight", "fragrance_free"]
  },
  "retrieval": {
    "strategy": "hybrid_balanced",
    "tiger": {"enabled": true, "candidate_budget": "standard"},
    "e5_faiss": {
      "enabled": true,
      "queries": ["lightweight daily moisturizer", "fragrance-free moisturizing cream"],
      "candidate_budget": "standard"
    }
  },
  "ranking": {
    "rerank_query": "Daily moisturizer under $30 with lightweight texture, preferably fragrance-free.",
    "priority_attributes": ["price", "lightweight", "fragrance_free"]
  },
  "clarification": {"required": false, "question": null, "reason": null},
  "assumptions": [],
  "change_summary": []
}
```

统一规则：

- `mode` 仅允许 `initial` / `replan`。
- `retrieval.strategy` 仅允许 `hybrid_balanced` / `behavior_priority` / `semantic_priority`。
- `candidate_budget` 仅允许 `standard` / `expanded`，具体数量由配置映射。
- `constraints.hard` 和 `constraints.soft` 使用结构化对象，不混用字符串与对象。
- `preferences.current`、`historical`、`negative`、`priority` 均为字符串数组。
- `parent_plan_id` 在 initial 模式下为 null，在 replan 模式下指向上一版计划。
- 剩余 API 次数、召回次数和 Replan 次数不属于 Plan，由 Controller BudgetState 管理。Replanner 输入中的 `budget_state` 仅供规划时参考，不允许 Replanner 自行增加预算。

字段说明：

| 字段 | 作用 |
|---|---|
| schema_version | 输出协议版本 |
| plan_id | 当前计划 ID |
| parent_plan_id | 上一版计划 ID |
| plan_version | 计划版本 |
| mode | initial / replan |
| intent | 用户需求理解 |
| constraints | 硬性约束和软性偏好 |
| preferences | 当前、历史、负向偏好及排序优先级 |
| retrieval | 召回策略 |
| ranking | 精排目标 |
| clarification | 是否需要追问 |
| assumptions | 规划时使用的假设 |
| change_summary | 本轮计划调整说明 |

正式开发时使用 Pydantic 定义完整嵌套数据类型、枚举和字段约束。该 Schema 是 Planner、Retrieval、Reranker、Replanner 和 Controller 之间的唯一计划协议；任何模块不得自行发明平行字段名。

# 七、Planner 与 LangGraph 的协作

LangGraph 作为整个系统的统一编排框架。

Planner 只负责生成结构化计划，LangGraph 负责调度和执行。

主要节点：

| 节点 | 职责 |
|---|---|
| load_user_context | 加载 Mem0 语义记忆、Interaction History 摘要和当前需求 |
| initial_planner | 生成初始计划 |
| clarification_gate | 根据 Plan 的 clarification.required 决定继续执行或向用户澄清 |
| wait_for_user | 等待用户补充信息后恢复当前任务 |
| validate_plan | 校验计划合法性 |
| tiger_retrieval | 执行 TIGER 召回 |
| e5_retrieval | 执行 E5 向量召回 |
| candidate_fusion | 候选合并与去重 |
| reranker | 商品精排 |
| judgment | 商品匹配与推荐合理性判断 |
| controller | 决定下一步动作 |
| replanner | 根据执行反馈生成新计划 |
| recommendation_generator | 最终推荐生成 |

Initial Planner 和 Replanner 可以使用不同的 LangGraph 节点，但共享同一套 Planner 服务、DeepSeek API Client 和 `RecommendationPlan` Schema。若 `clarification.required=true`，工作流进入 `wait_for_user`，收到补充信息后更新 Session History 并恢复执行；第一阶段可先实现最小澄清挂起/恢复能力。

TIGER 和 E5 在条件允许时并行执行，之后进入候选融合。

## 7.1 Planner 与 Controller 的职责边界

两者必须严格分离。

| Recommendation Planner | Python Controller |
|---|---|
| 理解用户需求 | 检查执行状态 |
| 融合用户记忆 | 管理召回预算 |
| 制定召回策略 | 检查候选数量 |
| 生成检索 Query | 检查硬性约束 |
| 生成精排目标 | 整合 Judgment 结果 |
| 根据反馈调整计划 | 决定 STOP / REPLAN |

Controller 决定是否调用 Replanner。

Replanner 只负责在被触发后生成新计划。

Planner 不直接控制整个系统的终止条件。

# 八、稳定性与异常处理

为了保证 API 模型输出的可靠性，设置独立的 Plan Validator。

主要检查：

1. JSON 格式是否合法。
2. 必填字段是否存在。
3. 字段类型是否正确。
4. 召回策略是否属于预定义枚举。
5. 召回预算是否在允许范围内。
6. Query 数量和长度是否符合限制。
7. Replanner 是否保留原始硬性约束。
8. 计划中的模块是否在 System Capabilities 中可用。

异常处理：

- API 超时：有限重试。
- JSON 解析失败：重新请求或执行格式修复。
- Schema 校验失败：返回错误信息并进行有限修复。
- 多次失败：降级到默认均衡召回计划。
- TIGER 不可用：保留 E5 召回。
- E5 不可用：保留 TIGER 召回。

同时设置最大 Replan 次数和全局召回预算，防止系统陷入无限规划循环。

# 九、运行轨迹与数据记录

本模块需要从第一版开始记录完整 Planner Trace，为后续模型训练和分析积累数据。

每次 Planner 调用保存：

| 字段 | 内容 |
|---|---|
| request_id | 推荐任务 ID |
| plan_id | 当前计划 ID |
| parent_plan_id | 上一轮计划 ID |
| mode | initial / replan |
| input_context | Planner 完整输入 |
| output_plan | 结构化输出 |
| validation_result | Schema 校验结果 |
| replan_reason | 重规划原因 |
| model_name | 模型名称 |
| latency | API 耗时 |
| token_usage | Token 消耗 |
| timestamp | 调用时间 |

另外保存每轮执行结果：

- TIGER 召回数量。
- E5 召回数量。
- 候选合并数量。
- Reranker 排序结果。
- Judgment 判断结果。
- Controller 动作。
- 最终推荐结果。

这样可以形成：

**User Request → Plan → Retrieval → Ranking → Judgment → Controller → Replan → Final Recommendation**

的完整运行轨迹。

这些数据后续可以用于构建 Planner SFT 训练集和独立评测集。

# 十、实验与评测

本模块重点验证生成式规划和动态重规划的实际价值。

## 10.1 Planner 独立评测

| 评测维度 | 指标 |
|---|---|
| 需求理解 | Intent Accuracy |
| 约束提取 | Constraint Extraction Precision / Recall / F1 |
| 计划合法性 | Plan Validity Rate |
| 查询质量 | Query Relevance |
| 召回规划 | Retrieval Strategy Accuracy |
| 重规划能力 | Replan Success Rate |
| 规划效率 | API Latency、Token Usage |

## 10.2 系统级对照实验

设计以下实验：

| 实验 | 方案 | 研究目的 |
|---|---|---|
| P0 | 固定均衡召回 | 基础对照 |
| P1 | Initial Planner 动态规划 | 验证动态召回策略 |
| P2 | Initial Planner + Replanner | 验证动态重规划 |
| P3 | 完整 Planner + User Memory | 验证记忆增强规划 |

主要推荐指标：

- Recall@K。
- NDCG@K。
- MRR。
- Constraint Violation Rate。
- Catalog Coverage。
- Retrieval Cost。
- Replan Success Rate。

通过消融实验分析动态规划、Replanner 和 User Memory 对最终推荐效果的贡献。

# 十一、技术选型

| 模块 | 技术方案 |
|---|---|
| Planner 模型 | DeepSeek API |
| Initial Planner | DeepSeek API |
| Replanner | DeepSeek API，共用模型 |
| API Client | Python |
| 输出格式 | Structured JSON |
| 数据校验 | Pydantic |
| Prompt 管理 | 独立 Prompt 文件 |
| 工作流编排 | LangGraph |
| 状态管理 | LangGraph State |
| 运行轨迹 | JSONL + SQLite |
| 异常处理 | Retry + Validation + Fallback |

本次课程设计不进行 Planner SFT、LoRA 或其他专项模型训练。

# 十二、项目范围与后续扩展

本次课程设计阶段：

- 使用 DeepSeek API 实现 Initial Planner。
- 使用 DeepSeek API 实现 Replanner。
- 建立统一结构化输出协议。
- 实现召回策略动态调整。
- 接入 LangGraph。
- 建立完整 Planner Trace。
- 完成规划能力和系统级实验。

后续独立模型项目可以开展：

1. 基于真实 Planner Trace 构建 SFT 数据集。
2. 使用 Qwen3 系列开源模型训练 Planner。
3. 优化复杂购物需求的拆解能力。
4. 优化召回策略选择能力。
5. 优化多轮 Replanning 能力。
6. 研究 Planner 与 Judgment Model 的协同决策。

这些工作不属于本次课程设计的必要交付范围。

# 十三、模块最终定位

Recommendation Planner 是一个由 DeepSeek API 驱动的生成式推荐规划模块。

它以用户当前购物需求和 User Memory 为基础，理解用户意图、提取商品约束、融合历史偏好，并动态规划 TIGER 生成式召回和 E5 向量召回的执行策略。

在后续推荐执行过程中，Planner 根据 Controller 提供的执行反馈，通过 Replanner 调整召回方向和推荐计划。

最终形成：

**User Understanding → Memory Fusion → Retrieval Planning → Execution Feedback → Dynamic Replanning**

的闭环规划机制。

本模块的核心价值不是简单生成搜索关键词，而是使推荐系统能够根据用户需求和实际推荐效果动态调整推荐策略，为 Agentic Generative Recommendation 提供规划能力。