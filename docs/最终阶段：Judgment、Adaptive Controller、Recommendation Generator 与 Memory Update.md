# 最终阶段：Judgment、Adaptive Controller、Recommendation Generator 与 Memory Update

**项目名称：** Agentic Generative Recommendation

**模块名称：** Adaptive Decision & Recommendation Generation

**文档版本：** V1.1（统一接口与记忆边界修订版）

**核心技术：** DeepSeek API + LangGraph + Python + Mem0

---

# 一、模块定位与设计目标

本模块是整个生成式推荐系统的最后一个阶段，负责对召回和精排后的商品进行进一步判断，根据判断结果动态调整检索流程，并生成最终推荐结果。

前面的 Candidate Retrieval 和 Product Reranker 解决了商品发现与排序问题，但仅依靠相关性排序还不足以保证最终推荐结果符合用户的实际需求。

因此，本模块引入结构化商品判断和自适应控制机制，使系统能够：

1. 判断候选商品是否真正符合用户的购物需求。
2. 根据商品匹配情况判断是否需要重新规划和召回。
3. 在候选商品质量满足要求时及时停止检索。
4. 生成有依据的商品推荐和推荐理由。
5. 根据用户实际反馈更新长期记忆。

本次课程设计遵循此前确定的开发原则：

**使用 DeepSeek API 和成熟开源组件搭建完整工作流，不训练 Planner、Judgment 和 Generator 等 Agent 模型。**

后续通过保存运行轨迹和用户反馈，为独立的模型训练项目积累数据。

# 二、整体架构

```text
                 Product Reranker
                  Qwen3-Reranker
                         |
                         v
                  Judgment Model
                    DeepSeek API
                         |
                         v
                 Adaptive Controller
                   LangGraph + Python
                         |
                  +------+------+
                  |             |
                REPLAN         STOP
                  |             |
                  v             v
              DeepSeek      Recommendation
               Planner        Generator
                  |          DeepSeek API
                  v             |
             TIGER + E5         v
                FAISS      Top-K 商品推荐
                  |        + 推荐理由
                  v             |
             Candidate          v
               Fusion      Update Memory
                  |           Mem0
                  v             |
               Reranker         v
                  |          下一轮交互
                  +----> Judgment
```

模块职责：

| 模块 | 技术方案 | 核心职责 |
|---|---|---|
| Judgment Model | DeepSeek API | 对候选商品进行结构化判断 |
| Adaptive Controller | LangGraph + Python | 汇总判断、管理状态、决定下一步动作 |
| Recommendation Generator | DeepSeek API | 生成最终商品推荐和推荐理由 |
| Update Memory | Mem0 | 根据用户反馈更新记忆 |

需要特别明确：

Judgment 和 Generator 虽然都使用 DeepSeek API，但它们是两个独立的调用任务，使用不同的 Prompt、输入结构和输出 Schema。

不需要为这两个任务分别部署本地模型。

---

# 三、Judgment Model：商品结构化判断

## 3.1 模块定位

在 Candidate Retrieval 和 Reranker 完成后，系统已经获得了一批按照相关性排序的候选商品。

但是，Reranker 的相关性分数并不能完全代表商品是否满足用户的实际购物需求。

例如：

用户希望购买：

- 30 美元以内的日常面霜。
- 质地清爽。
- 最好没有明显香味。

Reranker 可能将某款价格合适、描述相关的面霜排在第一位，但该商品并没有明确说明是否无香型。

此时需要进一步判断：

- 商品是否符合用户的核心偏好？
- 商品的哪些属性有明确依据？
- 是否存在缺失的关键信息？
- 当前候选集合是否能够满足用户的购物需求？

因此引入 Judgment Model，对候选商品进行细粒度的语义判断。

## 3.2 模型选型

采用 DeepSeek API 作为 Judgment Model。

第一阶段不进行本地部署和模型训练，直接使用 DeepSeek API 完成结构化判断。

选择 API 的原因：

1. 与 Planner 和 Generator 共用现有 API 技术路线，减少部署复杂度。
2. 无需额外准备 GPU 推理环境。
3. 能够处理用户需求、Memory 和商品描述之间的复杂语义关系。
4. 支持通过专门的 Prompt 约束输出格式。
5. 后续可以利用 API 生成的判断轨迹构建专用 Judgment Model 的训练数据。

这里的 Judgment Model 是一个逻辑模块，而不是独立训练的模型。

## 3.3 输入设计

Judgment Model 接收四类信息：

| 输入 | 内容 | 来源 |
|---|---|---|
| Current Request | 用户当前购物需求 | User Input |
| User Preferences | 用户历史偏好和记忆 | User Memory |
| Recommendation Plan | 用户需求、硬性约束和优先级 | Planner |
| Candidate Product | 候选商品及其详细信息 | Reranker / Product Catalog |

输入示例：

```json
{
  "current_request": "寻找30美元以内的清爽型日常面霜",

  "user_preferences": {
    "positive": [
      "lightweight",
      "moisturizing"
    ],
    "negative": [
      "strong_fragrance"
    ]
  },

  "constraints": {
    "max_price": 30,
    "category": "moisturizer"
  },

  "candidate": {
    "product_id": "A001",
    "title": "Daily Moisturizing Cream",
    "price": 25.99,
    "description": "Lightweight moisturizing cream",
    "attributes": [
      "moisturizing",
      "lightweight"
    ],
    "constraint_status": {
      "price": "pass",
      "category": "pass"
    }
  }
}
```

## 3.4 判断粒度

采用：

**单商品独立判断 + Controller 汇总判断结果。**

即每个候选商品都产生独立的判断记录。

例如，当前有 20 个候选商品，Judgment Model 需要对每个商品分别判断，不能只输出一个笼统的候选池评价。

工程上可以将多个独立商品判断合并在一次 API 批量请求中，以减少调用开销，但每个商品必须有独立的判断结果。

## 3.5 判断维度

第一阶段设置以下四个核心判断维度：

| 判断维度 | 具体任务 |
|---|---|
| Preference Fit | 判断商品与用户当前需求和历史偏好的匹配程度 |
| Attribute Evidence | 判断商品属性是否有商品信息作为依据 |
| Match Assessment | 综合判断商品属于直接匹配、部分匹配还是不匹配 |
| Uncertainty | 判断商品信息是否充分，是否存在无法确认的属性 |

对于价格、品类等明确的硬性约束，优先使用 Python 进行确定性检查。

例如：

```python
def check_price(product, constraints):

    max_price = constraints.get("max_price")

    if max_price is None:
        return "not_required"

    price = product.get("price")

    if price is None:
        return "unknown"

    if price > max_price:
        return "fail"

    return "pass"
```

这样可以避免将简单的数值比较交给 LLM。

## 3.6 输出设计

采用结构化 JSON：

```json
{
  "product_id": "A001",

  "preference_fit": {
    "level": "high",
    "matched_preferences": [
      "lightweight",
      "moisturizing"
    ],
    "unmatched_preferences": [],
    "unknown_preferences": [
      "fragrance_free"
    ]
  },

  "attribute_evidence": {
    "lightweight": {
      "status": "supported",
      "source": "description"
    },
    "fragrance_free": {
      "status": "unknown",
      "source": null
    }
  },

  "match_assessment": "partial_match",

  "uncertainty": "medium",

  "warnings": [
    "商品没有明确的无香型属性信息"
  ]
}
```

其中，商品属性的判断必须有原始商品信息作为依据，不能让模型自行补充不存在的属性。所有商品统一使用内部主键 `product_id`；硬约束状态来自上游 Python Constraint Evaluator，Judgment 只消费该状态，不重复校验数值约束。

## 3.7 Judgment 的职责边界

Judgment Model 只负责输出商品级别的结构化判断。

它不负责：

- 生成新的检索 Query。
- 重新执行商品召回。
- 修改 Reranker 的排序结果。
- 直接决定 REPLAN 或 STOP。

**Judgment 负责判断，Controller 负责决策。**

---

# 四、Adaptive Controller：自适应控制器

## 4.1 模块定位

Adaptive Controller 是整个推荐流程的控制中心。

它负责整合 Judgment Model 的判断结果、当前候选商品质量和系统执行状态，决定是否需要重新规划，或者结束检索并进入最终推荐生成。

采用：

- LangGraph：工作流编排、节点管理、条件边和状态流转。
- Python：硬约束校验结果读取、质量汇总、状态管理、预算控制和动作执行。

硬约束本身由 Candidate Fusion 后的唯一 Python Constraint Evaluator 执行；Controller 读取 `pass` / `fail` / `unknown` 结果，不再实现第二套价格或品类判断。

不额外引入 Controller LLM。

## 4.2 核心职责

| 职责 | 具体内容 |
|---|---|
| 状态管理 | 保存每轮规划、召回、精排和判断结果 |
| 质量判断 | 汇总商品匹配情况和属性证据 |
| 预算管理 | 控制最大重规划次数和 API 调用成本 |
| 多样性管理 | 避免最终商品过度相似 |
| 流程控制 | 决定 REPLAN 或 STOP |
| 轨迹记录 | 保存完整执行过程 |

## 4.3 核心动作

Controller 设置两个主要出口。

### REPLAN：重新规划

当当前候选商品不能充分满足用户需求时，触发重新规划。

典型情况：

- 符合用户核心偏好的商品数量不足。
- 重要商品属性缺少明确证据。
- 当前召回结果与用户关键需求存在明显偏差。
- 候选商品多样性不足。
- 当前检索方向可能遗漏更合适的商品。

执行流程：

```text
Judgment Results
       |
       v
Controller 汇总判断
       |
       v
识别当前不足
       |
       v
DeepSeek Planner Replan
       |
       v
生成新的检索计划
       |
       v
TIGER + E5 + FAISS
       |
       v
Candidate Fusion
       |
       v
Product Reranker
       |
       v
Judgment Model
```

Replan 时将以下信息传递给 Planner：

- 当前用户需求。
- 当前推荐计划。
- 已召回的商品及其来源。
- Judgment 发现的问题。
- 未得到充分满足的偏好。
- 当前剩余检索预算。

Planner 根据这些信息调整检索 Query 和召回策略。

### STOP：结束检索

当当前候选商品已经满足推荐要求时，停止继续检索，进入 Recommendation Generator。

主要条件：

- 已经获得足够数量的有效候选商品。
- 用户的硬性约束已经得到满足。
- 核心偏好已经得到较好的覆盖。
- 没有必要继续消耗检索预算。

如果预算耗尽，也必须停止循环。预算耗尽是强制终止条件，不代表质量达标。Controller 必须分别记录 `quality_satisfied` 与 `budget_exhausted`。若因预算耗尽而停止但质量未达标，Generator 只能输出硬约束已验证、证据充分的商品，并明确说明未满足或未验证的需求；不能把预算停止标记成质量 STOP。

如果没有足够可靠的候选商品，则应如实告知用户，而不是为了凑够 Top-K 强行推荐。

## 4.4 Controller 实现

采用 LangGraph 构建工作流，Python 负责具体判断。

示意代码：

```python
def controller(state):
    # 预算耗尽是强制终止，但不代表质量达标
    if state["budget_state"]["remaining_replan_rounds"] <= 0 \
       or state["budget_state"]["remaining_retrieval_calls"] <= 0:
        return "STOP_BUDGET_EXHAUSTED"

    if state["valid_product_count"] < state["required_count"]:
        return "REPLAN"

    if not state["hard_constraints_satisfied"]:
        return "REPLAN"

    if not state["quality_sufficient"]:
        return "REPLAN"

    return "STOP_QUALITY_SATISFIED"
```

正式接口可将上述两个 STOP 原因统一映射为 LangGraph 的 `STOP` 路径，但必须保留 `stop_reason` 和 `quality_satisfied`，避免把预算耗尽误报为质量达标。

以上是简化的控制逻辑。

正式实现中，还需要考虑：

- 商品属性缺失。
- 判断结果不确定。
- 重复检索。
- 候选商品多样性。
- 最大迭代轮数。

所有质量阈值通过验证集确定，不直接使用未经验证的固定阈值。

### 4.4.1 多维预算状态

Controller 不使用含义不清的单个 `remaining_budget: int`。建议统一维护：

```json
{
  "remaining_replan_rounds": 2,
  "remaining_retrieval_calls": 4,
  "tiger_candidate_budget": 50,
  "e5_candidate_budget": 100,
  "remaining_api_calls": 12
}
```

以上数字仅为配置示例，实际值由 `configs/controller.yaml` 决定。Planner 只能选择 `standard` / `expanded` 等预算档位，Controller 负责扣减全局剩余预算。

---

## 4.5 LangGraph 状态设计

建议维护统一的 RecommendationState：

```python
class RecommendationState(TypedDict):

    # 用户信息
    user_request: str
    user_memory: dict
    interaction_history: list
    session_history: list

    # Planner
    current_plan: dict

    # 召回与融合
    tiger_candidates: list
    e5_candidates: list
    fused_candidates: list

    # 精排
    ranked_products: list

    # Judgment
    judgment_results: list

    # 约束校验与候选池质量
    constraint_summary: dict
    valid_product_count: int
    required_count: int
    quality_metrics: dict

    # Controller：预算是多维状态，不使用含义不明的单个整数
    action: str
    stop_reason: str
    quality_satisfied: bool
    budget_exhausted: bool
    budget_state: dict
    current_round: int
    max_replan_rounds: int
    seen_query_hashes: list
    request_id: str
    trace_id: str

    # 最终推荐
    selected_products: list
    recommendation: dict
```

LangGraph 通过条件边连接 REPLAN 和 STOP 两条主路径。若 Planner 返回 `clarification.required=true`，应在检索前进入 Ask User / Wait for User 分支；收到补充信息后更新 Session History 并恢复 Planner。第一阶段可先实现最小的挂起与恢复接口。

REPLAN 返回 Planner 节点，STOP 进入 Recommendation Generator。

## 4.6 防止无限循环

Controller 必须设置：

1. 最大 Replan 次数。
2. 最大检索预算。
3. 重复 Query 检测。
4. 每轮候选商品数量限制。
5. API 调用失败处理。

当达到预算上限时，结束检索并进入最终结果处理。

---

# 五、Recommendation Generator：最终推荐生成

## 5.1 模块定位

Recommendation Generator 是直接面向用户的最终输出模块。

采用 DeepSeek API，不单独训练生成模型。

其主要任务是将前面模块得到的结构化结果组织成用户能够理解的商品推荐内容。

## 5.2 输入设计

Generator 接收：

- 用户当前购物需求。
- User Memory 中相关的历史偏好。
- Planner 的需求理解和约束。
- Reranker 的精排结果。
- Judgment Model 的判断结果。
- 商品原始元数据。

输入示例：

```json
{
  "user_request": "寻找30美元以内的日常面霜",

  "selected_products": [
    {
      "product_id": "A001",
      "rank": 1,
      "title": "Daily Moisturizing Cream",
      "price": 25.99,

      "matched_preferences": [
        "lightweight",
        "moisturizing"
      ],

      "warnings": [
        "fragrance_free 未得到明确验证"
      ]
    }
  ]
}
```

## 5.3 输出设计

采用结构化推荐结果：

```json
{
  "recommendations": [
    {
      "rank": 1,
      "product_id": "A001",
      "title": "Daily Moisturizing Cream",
      "price": 25.99,

      "reasons": [
        "价格符合30美元以内的预算",
        "商品描述明确提到清爽质地",
        "符合用户对保湿效果的偏好"
      ],

      "limitations": [
        "商品信息未明确说明是否无香型"
      ]
    }
  ]
}
```

最终向用户展示：

- Top-K 推荐商品。
- 商品名称和价格。
- 商品核心属性。
- 推荐理由。
- 与用户偏好的匹配情况。
- 必要的商品信息限制。

## 5.4 生成约束

为保证推荐内容可靠，设置以下规则：

1. Generator 只能使用 Controller 确认的候选商品。
2. 推荐理由必须能够在商品原始信息或 Judgment 结果中找到依据。
3. 不得自行编造商品属性、价格或功能。
4. 不得将未知属性描述为已经确认的事实。
5. 原则上保留 Reranker 已确定的商品排序，不由 Generator 擅自调整。
6. 如果有效候选不足，应允许输出少于 K 个商品。

Generator 负责表达，不负责重新执行推荐决策。

---

# 六、Update Memory：用户记忆更新

## 6.1 模块定位

Memory Update 负责将用户在购物过程中的有效反馈转化为长期或短期记忆。

项目此前已经确定采用 User Memory 机制，因此这里不再额外训练用户兴趣建模模型。

使用 Mem0 作为语义记忆管理方案。原始、带时间戳的商品交互事件另存于 SQLite `InteractionEvent` 表，供 TIGER 构建历史序列；不能把 Mem0 的自然语言记忆直接当作 TIGER 行为序列。

## 6.2 记忆分类

第一阶段维护三类信息：

| 记忆类型 | 记录内容 | 示例 |
|---|---|---|
| Long-term Preference | 稳定的用户偏好 | 喜欢清爽型护肤品 |
| Short-term Interest | 当前购物兴趣 | 最近正在寻找面霜 |
| Interaction History | 用户实际交互事件 | Amazon review/rating 代理事件，以及后续真实点击、收藏、购买等有来源的事件；结构化存入 SQLite，不作为 Mem0 语义记忆存储 |

## 6.3 Memory 更新机制

推荐结束后，不应直接把系统生成的推荐理由写入长期记忆。

建议区分两种情况。

**情况一：仅完成推荐，没有用户反馈**

- 保存当前会话信息。
- 不自动增加长期偏好。
- 等待后续交互。

**情况二：用户产生明确反馈或有效交互**

例如用户反馈：

> 我更喜欢第一款，以后尽量不要推荐有明显香味的产品。

可以提取：

```json
{
  "memory_updates": [
    {
      "type": "long_term_preference",
      "action": "add",
      "content": "用户倾向于无香型产品",
      "source": "explicit_user_feedback",
      "target_store": "mem0"
    },
    {
      "type": "interaction",
      "action": "add",
      "event": {
        "event_type": "explicit_positive_feedback",
        "product_id": "A001",
        "content": "用户明确表示更喜欢商品A001",
        "source": "explicit_user_feedback",
        "is_synthetic": false
      },
      "target_store": "sqlite_interaction_history"
    }
  ]
}
```

将长期/短期语义偏好交给 Mem0 添加、更新和检索；将具体商品交互事件写入 SQLite `InteractionEvent`。两者通过 user_id、product_id 和 source_id 关联，但职责与存储结构分离。

## 6.4 更新原则

为了减少错误记忆污染：

1. 用户明确表达的偏好，可以更新长期记忆。
2. 点击、收藏、购买等有效行为，可以记录为交互事件。
3. 用户明确拒绝的商品，可以记录负向反馈。
4. 单纯没有点击某个商品，不能直接认定用户不喜欢它。
5. 模型推断出的偏好不能未经确认就写成用户事实。
6. 对于模拟购物对话产生的反馈，需要标记为模拟数据，不能与真实用户反馈混淆。

Memory Update 完成后，新的语义记忆会在下一轮购物需求中被 Planner 检索，并可由 Planner 融合进 E5 Query。TIGER 仍从 SQLite 的结构化 Interaction History 构建序列，不直接读取 Mem0 文本记忆。

---

# 七、LangGraph 整体工作流

建议将最终阶段与前面的 Planner、Retrieval 和 Reranker 统一编排。

节点设计：

| 节点 | 职责 |
|---|---|
| planner | DeepSeek 初始规划 |
| tiger_retrieval | TIGER 生成式召回 |
| e5_retrieval | E5 + FAISS 语义召回 |
| candidate_fusion | 候选融合 |
| reranker | Qwen3-Reranker 精排 |
| judgment | DeepSeek 逐商品结构化判断 |
| controller | Python 规则与状态决策 |
| replanner | DeepSeek 根据不足调整检索计划 |
| recommendation_generator | DeepSeek 生成最终推荐 |
| memory_update | 根据有效反馈更新 Mem0 语义记忆，并将结构化行为事件写入 SQLite |
| ask_user / wait_for_user | 处理 Planner 要求的必要澄清，并在用户回复后恢复工作流 |

整体流程：

```text
                  START
                    |
                    v
                  Planner
                    |
                    v
             Parallel Retrieval
              /             \
           TIGER           E5
              \             /
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
             +------+------+
             |             |
          REPLAN          STOP
             |             |
             v             v
          Replanner     Generator
             |             |
             v             v
          Retrieval    Final Output
                           |
                           v
                     User Feedback
                           |
                           v
                     Memory Update
                           |
                           v
                          END
```

其中，REPLAN 路径会再次经过召回、融合、精排和判断。

STOP 路径进入最终推荐生成。

Memory Update 不应在每次系统内部循环时执行，而应根据推荐后的实际用户反馈触发。仅完成推荐而没有用户反馈时，不得推断用户喜欢推荐商品，也不得写入长期偏好。模拟反馈必须保留 `is_synthetic=true`，与真实反馈隔离。

---

# 八、实验设计与评测

为了验证最后阶段各模块的实际价值，设计以下对照实验。

| 实验 | 系统方案 | 研究目的 |
|---|---|---|
| E1 | Reranker + 直接生成 | 基础推荐基线 |
| E2 | Judgment + 固定规则 Controller | 验证结构化判断 |
| E3 | Judgment + Adaptive Controller | 验证动态重规划 |
| E4 | Judgment + Controller + Memory | 验证记忆带来的个性化增益 |

## 8.1 Judgment 评测

- Preference Match Accuracy：偏好匹配判断准确率。
- Attribute Evidence Accuracy：商品属性证据判断准确率。
- Match Assessment Accuracy：商品整体匹配判断准确率。
- Uncertainty Detection：不确定性识别能力。

通过人工标注的商品匹配数据构建测试集。

## 8.2 Controller 评测

- Replan Success Rate：重新规划后的推荐质量提升比例。
- Stop Accuracy：停止时机的合理性。
- Constraint Satisfaction Rate：最终商品硬性约束满足率。
- Retrieval Cost：平均召回轮数及 API 调用成本。
- Latency：整体推荐延迟。

重点研究：

在相同检索预算下，Adaptive Controller 是否能够减少无效检索，并提升最终推荐质量。

## 8.3 Generator 评测

- Recommendation Quality：最终推荐质量。
- Reason Groundedness：推荐理由是否有商品信息支持。
- Hallucination Rate：商品属性编造比例。
- Constraint Satisfaction：最终推荐是否符合用户要求。

## 8.4 Memory 评测

- Memory Update Accuracy：记忆更新准确率。
- Preference Consistency：记忆与用户真实反馈的一致性。
- Personalization Improvement：加入记忆前后的推荐效果差异。

---

# 九、工程实现与异常处理

## 9.1 API 调用管理

Judgment、Planner 和 Generator 均使用 DeepSeek API，但应分别配置：

- 独立 Prompt。
- 独立输出 Schema。
- 超时和重试策略。
- Token 和调用预算。
- 调用日志。

不要因为它们使用同一个 API，就将其合并成一个 Agent。

## 9.2 结构化输出校验

使用 Pydantic 校验 API 返回结果。

当 JSON 格式错误或字段缺失时：

1. 尝试进行格式修复。
2. 必要时重新请求。
3. 多次失败后记录异常。
4. 不允许错误结果直接进入 Controller。

## 9.3 异常降级

| 异常情况 | 处理方式 |
|---|---|
| Judgment API 超时 | 有限重试；仍失败时将该商品标记为 judgment_unavailable / unknown，不得默认通过 |
| Planner Replan 失败 | 使用上一轮有效计划 |
| Generator 失败 | 使用结构化推荐模板输出 |
| Mem0 更新失败 | 保留反馈事件，后续重试 |
| 检索服务失败 | 保留正常召回路径 |
| 预算耗尽 | 停止检索并如实输出结果 |

---

# 十、阶段划分与最终交付

## 第一阶段：完整工作流搭建

本次课程设计不训练上述 Agent 模型。

主要工作：

1. 接入 DeepSeek Judgment API。
2. 实现逐商品结构化判断。
3. 使用 LangGraph 建立完整工作流。
4. 使用 Python 实现 Adaptive Controller。
5. 接入 DeepSeek Recommendation Generator。
6. 接入 Mem0 语义记忆，并建立 SQLite InteractionEvent 读写接口。
7. 完成 REPLAN 和 STOP 两条流程。
8. 保存完整运行轨迹。
9. 完成基础对照实验。

验收标准：

- Judgment 能够逐商品输出结构化判断。
- Controller 能够根据判断结果决定 REPLAN 或 STOP。
- REPLAN 能够重新调用 Planner 并执行新一轮召回。
- STOP 能够进入最终推荐生成。
- Generator 能够输出有依据的 Top-K 商品推荐。
- Memory 能够根据有效用户反馈更新。
- 整个工作流能够稳定运行并保存执行日志。

## 第二阶段：独立模型训练项目

在课程设计完成后，再利用运行轨迹开展模型训练。

| 模型 | 后续训练方向 |
|---|---|
| Planner | 利用规划和 Replan 轨迹进行 SFT |
| Judgment | 利用商品匹配判断数据训练专用模型 |
| Generator | 根据实际错误归因决定是否微调 |

Controller 继续使用 LangGraph + Python，不需要为了训练而训练。

Mem0 继续作为成熟记忆管理机制使用。

---

# 十一、最终技术选型总结

| 模块 | 技术选型 | 本次是否训练 |
|---|---|---|
| Planner | DeepSeek API | 否 |
| Judgment Model | DeepSeek API | 否 |
| Adaptive Controller | LangGraph + Python | 不需要 |
| Recommendation Generator | DeepSeek API | 否 |
| Memory | Mem0 + Qdrant + SQLite InteractionEvent | 不需要 |

本模块最终形成：

**Structured Judgment → Adaptive Control → Generative Recommendation → Memory Update**

通过生成式判断、程序化控制和长期记忆管理，使推荐系统不仅能够找到商品，还能够根据用户需求判断推荐质量、动态调整检索过程，并在用户反馈后持续优化个性化推荐。

最终实现一个完整的 Agentic Generative Recommendation 工作流。