# 模块一：User Memory 用户记忆系统
## 基于真实历史行为与模拟购物交互的动态用户记忆机制

**项目名称：** Agentic Generative Recommendation

**模块名称：** User Memory

**文档版本：** V1.1（统一数据边界修订版）

**核心技术：** Amazon Reviews 2023 + DeepSeek API + Mem0 OSS + Qdrant + SQLite

---

# 一、模块目标

本模块负责从用户历史行为和购物交互中提取、存储、检索和更新用户偏好，为后续生成式推荐提供持续、动态的用户上下文。

本项目不再独立训练 SASRec 或长期/短期兴趣融合模型，而是采用基于 LLM 的 Memory 机制管理用户兴趣。

核心目标：

1. 从真实历史交互中提取用户长期偏好。
2. 通过模拟购物对话构建动态交互场景。
3. 利用 Memory 框架实现记忆提取、存储、检索和更新。
4. 根据用户当前购物需求检索相关历史记忆。
5. 识别用户兴趣变化，处理新旧记忆之间的冲突。
6. 为后续 Recommendation Planner 提供结构化 User Context。

本模块重点研究：

**动态 User Memory 相比静态用户画像，能否更好地支持个性化生成式推荐？**

# 二、整体设计方案

采用两阶段记忆构建方案：

**第一阶段：历史行为初始化**

通过 Amazon Reviews 2023 提取用户历史偏好，建立初始 User Memory。

**第二阶段：模拟交互与动态更新**

根据历史行为生成用户 Persona，再利用 DeepSeek API 模拟多轮购物对话，将新的用户需求和反馈输入 Memory 系统，测试记忆更新能力。

整体流程：

```text
          Amazon Reviews 2023
                    |
                    v
       Historical Behavior Processor
                    |
          +---------+---------+
          |                   |
       行为统计            评论语义提取
          |                   |
          +---------+---------+
                    |
                    v
           User Persona
                    |
                    v
          Initial Memory
                    |
                    v
              Mem0 Store
                    |
                    |
       DeepSeek Persona Simulation
                    |
                    v
             模拟购物对话
                    |
                    v
          Conversation Processor
                    |
                    v
          Memory Write / Update
                    |
                    v
              Updated Memory
                    |
                    v
          Current Shopping Request
                    |
                    v
             Memory Retrieval
                    |
                    v
            User Context Builder
                    |
                    v
          Personalized User Context
```

需要区分：历史行为初始化和模拟交互更新是先后两个阶段，而不是将所有数据一次性写入 Memory。

# 三、数据来源与构建

## 3.1 真实历史行为数据

主数据集沿用项目整体方案：

**Amazon Reviews 2023 - Beauty and Personal Care**

主要使用：

- 用户 ID。
- 商品 ID（ASIN）。
- 用户评分。
- 评论标题。
- 评论正文。
- 评论时间。
- 商品标题。
- 商品类别。
- 商品属性。
- 商品描述。

数据处理后形成两张逻辑表。

### 用户行为表

| 字段 | 含义 |
|---|---|
| user_id | 用户 ID |
| parent_asin | Amazon 原始商品分组标识 |
| product_id | 项目内部统一商品主键，由数据接入层映射得到 |
| rating | 用户评分 |
| review_title | 评论标题 |
| review_text | 评论内容 |
| timestamp | 行为时间 |
| verified_purchase | 是否验证购买 |

### 商品信息表

| 字段 | 含义 |
|---|---|
| parent_asin | Amazon 原始商品分组标识 |
| product_id | 项目内部统一商品主键 |
| title | 商品名称 |
| category | 商品类别 |
| features | 商品特征 |
| description | 商品描述 |
| price | 商品价格 |
| details | 其他商品属性 |

通过数据接入层建立 `parent_asin` 到内部 `product_id` 的映射，并以 `product_id` 作为项目内唯一商品主键。后续 Planner、TIGER 解码、E5、Fusion、Reranker、Judgment 和 Generator 均只使用 `product_id`。原始 `parent_asin` 保留用于追溯 Amazon 数据。

Amazon Reviews 2023 的评论/评分记录在本项目中作为带时间戳的历史商品交互代理信号，不等同于点击、浏览、加购或完整购买日志。行为类型必须按实际数据来源记录为 `review` / `rating`，不能伪装成 `click` / `purchase`。

注意：Amazon Reviews 主要提供评论和评分等交互信息，不应将其直接视为完整的点击、浏览和加购日志。模拟购物对话用于补充真实数据中缺少的交互场景。

## 3.2 历史行为处理

采用 Python + DeepSeek API 的混合处理方式。

**Python 负责：**

- 用户历史记录统计。
- 商品类别统计。
- 评分分布统计。
- 用户交互时间分析。
- 高频商品属性统计。
- 行为时间排序。

**DeepSeek API 负责：**

- 评论语义分析。
- 正向偏好提取。
- 负向偏好提取。
- 商品属性偏好识别。
- 用户明确表达的使用需求识别。

不需要将每一条评论都发送给 LLM。先通过 Python 进行数据筛选和聚合，再对有价值的评论进行语义提取。

## 3.3 偏好提取示例

假设某用户存在以下历史记录：

| 商品 | 评分 | 评论内容 |
|---|---:|---|
| 无香型面霜 | 5 | 保湿效果很好 |
| 保湿乳液 | 5 | 质地不错 |
| 香味浓烈的乳液 | 2 | 气味太重 |

经过处理，得到：

```json
{
  "user_id": "user_001",
  "positive_preferences": [
    "moisturizing",
    "fragrance_free"
  ],
  "negative_preferences": [
    "strong_fragrance"
  ],
  "interested_categories": [
    "skincare"
  ]
}
```

以上仅为记忆候选，不代表所有信息都自动成为长期偏好。

系统需要结合评分、评论内容、重复证据和时间信息判断记忆是否可靠。

# 四、用户 Persona 构建

完成历史行为处理后，为每个用户建立初始 Persona。

Persona 是根据历史行为提取的用户偏好摘要，作为 Memory 初始化和模拟用户的基础。

示例：

```json
{
  "user_id": "user_001",
  "persona": {
    "preferred_categories": [
      "skincare"
    ],
    "positive_preferences": [
      "moisturizing",
      "lightweight"
    ],
    "negative_preferences": [
      "strong_fragrance"
    ],
    "known_interests": [
      "daily_skincare"
    ]
  }
}
```

Persona 必须能够追溯到历史行为证据。

不应直接把 DeepSeek 生成的所有 Persona 内容视为真实用户事实。

# 五、模拟购物对话构建

## 5.1 构建目标

真实历史评论可以反映用户过去的商品评价，但无法充分覆盖用户主动表达需求、修改偏好和纠正推荐等交互场景。

因此，利用 DeepSeek API 构建模拟购物对话。

模拟用户必须以已经构建的历史 Persona 为基础，保证对话具有一定的用户个性。

## 5.2 模拟场景

第一版设计五类场景：

| 场景 | 用户行为 | 主要测试能力 |
|---|---|---|
| 历史偏好延续 | 提出与历史偏好一致的需求 | 记忆检索 |
| 临时购物需求 | 提出一次性购物要求 | 长期与临时需求区分 |
| 新兴趣产生 | 表达之前没有出现的新兴趣 | 新记忆写入 |
| 明确负反馈 | 拒绝之前推荐的某种属性 | 负向记忆更新 |
| 偏好发生变化 | 明确表示不再喜欢以前的偏好 | 记忆冲突处理 |

## 5.3 模拟对话示例

初始 Persona：

用户喜欢保湿型护肤品，不喜欢浓烈香味。

模拟对话：

```text
User:
最近我想买一款面霜，有什么推荐？

Assistant:
你更喜欢清爽型还是滋润型的面霜？

User:
我更喜欢清爽一些的，而且不要有太浓的香味。

Assistant:
好的，我会优先考虑清爽、低香或无香型产品。

User:
对，以后推荐护肤品也尽量避免浓香型。
```

预期新增记忆：

```json
{
  "content": "用户偏好清爽型护肤品，并避免浓烈香味",
  "memory_type": "long_term_preference",
  "source": "simulated_dialogue"
}
```

模拟对话需要同时保存：

- Persona。
- 对话上下文。
- 用户输入。
- Assistant 回复。
- 预期记忆变化。
- 实际 Memory 更新结果。

其中预期记忆变化作为后续评测的参考标签。

## 5.4 模拟数据质量控制

模拟对话由 DeepSeek API 生成，但不能直接把模型输出全部当作正确标签。

需要：

1. 检查模拟对话是否符合初始 Persona。
2. 检查新偏好是否具有明确的上下文依据。
3. 检查长期偏好与临时需求是否区分正确。
4. 抽样进行人工审核。
5. 保留模拟数据标识。

所有模拟数据设置：

```json
{
  "is_synthetic": true,
  "source": "deepseek_persona_simulation"
}
```

模拟数据只用于 Memory 机制开发和受控实验，不应混入真实用户行为统计并作为真实用户反馈。

# 六、Memory 框架与技术选型

## 6.1 采用 Mem0 OSS

选择成熟的开源 Memory 框架：

**Mem0 OSS**

GitHub：https://github.com/mem0ai/mem0

官方文档：https://docs.mem0.ai/

Mem0 主要负责：

- Memory Extraction。
- Memory Storage。
- Memory Retrieval。
- Memory Update。
- Memory Deduplication。

我们不从零实现记忆提取和管理算法，而是在成熟 Memory 框架基础上完成推荐场景的适配。

## 6.2 整体技术栈

| 模块 | 技术 |
|---|---|
| 历史行为处理 | Python |
| 评论语义提取 | DeepSeek API |
| Persona 构建 | DeepSeek API + Python |
| 模拟用户 | DeepSeek API |
| Memory 框架 | Mem0 OSS |
| Memory Extraction LLM | DeepSeek API |
| Memory Embedding | 独立文本 Embedding 模型 |
| Vector Store | Qdrant |
| 原始数据与元数据 | SQLite |
| 当前会话状态 | LangGraph State |

Mem0 负责语义记忆的提取、存储、检索和更新；Qdrant 作为向量存储；SQLite 保留原始评论证据、来源映射、按时间排序的结构化交互事件及审计信息。

**重要边界：Mem0 中的语义记忆不等同于 TIGER 所需的行为序列。** TIGER 的序列必须从 SQLite 中的 `InteractionEvent` 按时间排序构建；不得从 Mem0 的自然语言记忆文本中猜测或拼接商品序列。

LangGraph Checkpointer 负责当前工作流状态持久化，不代替跨会话的 User Memory，也不代替 SQLite 中的原始行为事件表。

## 6.3 记忆类型

系统将记忆分成三个层次：

### A. Long-term Memory

长期稳定的用户偏好：

- 喜欢的商品类别。
- 品牌偏好。
- 价格敏感度。
- 商品属性偏好。
- 明确的负面偏好。

### B. Dynamic Memory

近期兴趣：

- 最近关注的商品类别。
- 近期频繁出现的商品属性。
- 当前购物趋势。
- 兴趣变化。

具有时间属性，可以根据后续行为更新或衰减。

### C. Episodic Memory

具体历史事件：

- 用户购买过什么商品。
- 用户曾经拒绝什么推荐。
- 用户曾经表达过什么反馈。
- 用户对某次推荐的评价。

当前购物需求单独存放在 LangGraph `Session Context`，不默认写入长期 Memory。Episodic Memory 可保存经过提炼的事件摘要；原始、可排序、可审计的行为事件仍保存在 SQLite。

# 七、Memory 数据结构

每条记忆需要具备统一的数据结构。

```json
{
  "memory_id": "mem_001",
  "user_id": "user_001",

  "memory_type": "long_term_preference",

  "content": "用户偏好无香型护肤品",

  "category": "skincare",
  "polarity": "positive",

  "confidence": 0.88,

  "source": {
    "source_type": "amazon_review",
    "source_ids": [
      "review_001",
      "review_002"
    ]
  },

  "evidence_count": 2,

  "created_at": "2026-11-23",
  "last_observed_at": "2026-11-20",

  "is_synthetic": false,
  "status": "active"
}
```

核心字段说明：

| 字段 | 作用 |
|---|---|
| memory_type | 记忆类型 |
| content | 记忆内容 |
| confidence | 记忆可靠程度 |
| source | 记忆来源 |
| source_ids | 原始证据 ID |
| evidence_count | 支持证据数量 |
| last_observed_at | 最近观察时间 |
| is_synthetic | 是否为模拟数据 |
| status | 当前记忆状态 |

置信度由证据质量、重复程度、明确表达程度和时间等因素综合计算，不能直接使用 LLM 自报的置信度。

## 7.1 Interaction History：独立的结构化行为事件

为 TIGER 提供按时间排序的用户商品交互序列，单独维护 `InteractionEvent`，不将其等同于 Mem0 语义记忆。

建议字段：

```json
{
  "event_id": "evt_001",
  "user_id": "user_001",
  "product_id": "A001",
  "event_type": "review",
  "rating": 5,
  "timestamp": 1795300000,
  "source": "amazon_reviews_2023",
  "source_id": "review_001",
  "is_synthetic": false
}
```

字段约束：

- `product_id` 使用项目内部统一商品主键。
- `event_type` 必须反映真实来源；Amazon 评论数据可使用 `review` / `rating`，不能推断成点击或购买。后续真实产品交互可使用 `click`、`favorite`、`purchase`；用户明确表达的商品反馈可使用 `explicit_positive_feedback` / `explicit_negative_feedback`。
- `timestamp` 用于构建时间顺序序列。
- `is_synthetic` 用于区分模拟交互。
- TIGER 第一版默认使用真实历史评论/评分代理事件；模拟事件只在单独的受控实验中启用，并且必须显式过滤和标记。

Mem0 保存语义偏好和必要的事件摘要；SQLite 保存原始 `InteractionEvent`。两者通过 `user_id` 和来源 ID 关联，但不合并成同一张数据表。

---

# 八、Memory 初始化与动态更新流程

## 8.1 第一阶段：初始化

```text
Amazon Reviews
      |
      v
Historical Behavior Processing
      |
      v
Preference Extraction
      |
      v
User Persona
      |
      v
Memory Candidate Validation
      |
      v
Mem0 Add Memory
      |
      v
Initial User Memory
```

完成后，每个用户拥有一份基于历史行为建立的初始语义 Memory；原始按时间排序的历史交互同时写入 SQLite `InteractionEvent`，供 TIGER 构建序列。

## 8.2 第二阶段：模拟交互更新

```text
Initial User Memory
        |
        v
Persona-based Simulation
        |
        v
Simulated Shopping Dialogue
        |
        v
Conversation Processing
        |
        v
Memory Candidate Extraction
        |
        v
Memory Write Decision
        |
        +----------------+
        |                |
      WRITE           SESSION ONLY
        |                |
        v                v
    Mem0 Update       Session State
        |
        v
  Updated User Memory（仅语义记忆）

结构化模拟交互事件如需用于实验，单独写入 SQLite 并保留 `is_synthetic=true`，不得默认混入真实 TIGER 序列。
```

## 8.3 记忆写入规则

并不是所有用户输入都需要写入长期 Memory。

| 用户表达 | 处理方式 |
|---|---|
| 我一直喜欢轻便型背包 | 长期记忆候选 |
| 最近开始关注露营用品 | 动态记忆候选 |
| 这次旅行想买一款红色背包 | 当前 Session |
| 以后不要给我推荐浓香型产品 | 长期负向记忆候选 |
| 我以前喜欢油润型，现在更喜欢清爽型 | 更新旧记忆 |

记忆写入应经过 Python 规则审核。

重点判断：

1. 是否明确表达了偏好。
2. 是否具有一定持续性。
3. 是否只是本次购物的临时要求。
4. 是否与已有记忆冲突。
5. 是否有足够的证据支持。

## 8.4 记忆冲突处理

当新信息与旧记忆冲突时，不应简单覆盖。

例如：

```text
Old Memory:
用户喜欢油润型护肤品。

New Memory:
用户现在更喜欢清爽型护肤品。
```

处理原则：

- 明确表达的最新偏好优先。
- 保留旧记忆作为历史事件。
- 更新当前有效偏好。
- 记录偏好变化时间。
- 避免旧记忆继续错误影响推荐。

# 九、当前购物需求与 Memory Retrieval

## 9.1 当前需求解析

用户提出：

「我想买一款 30 美元以内、适合日常使用的清爽型面霜。」

先通过 Input Understanding 提取：

```json
{
  "category": "skincare",
  "product_type": "moisturizer",
  "hard_constraints": {
    "max_price": 30
  },
  "soft_preferences": [
    "lightweight",
    "daily_use"
  ]
}
```

## 9.2 Memory Retrieval

将当前购物需求转化为记忆检索 Query。

例如：

```text
检索用户在护肤品方面的长期偏好，
重点关注面霜质地、香味和保湿效果。
```

Mem0 根据 user_id 检索相关记忆。

## 9.3 User Context Construction

最终合并：

- 长期偏好。
- 近期兴趣。
- 相关历史事件。
- 当前购物需求。

输出：

```json
{
  "user_id": "user_001",

  "long_term_preferences": [
    "fragrance_free",
    "lightweight"
  ],

  "short_term_interests": [
    "moisturizing"
  ],

  "relevant_episodes": [],

  "current_request": {
    "category": "skincare",
    "product_type": "moisturizer",
    "max_price": 30,
    "preferences": [
      "lightweight",
      "daily_use"
    ]
  }
}
```

最终交给后续 Recommendation Planner。TIGER 的行为序列由独立 `InteractionHistory` 接口从 SQLite 获取；Planner / E5 使用 Mem0 返回的语义 User Context。

上下文优先级：

**当前用户明确需求 > 明确表达的长期偏好 > 多次历史行为推断 > 单次历史行为推断。**

当前需求不能被旧记忆覆盖。

# 十、实验与评测

本模块不仅需要实现 Memory，还需要验证其有效性。

## 10.1 Memory 独立评测

| 评测维度 | 评测目标 | 指标 |
|---|---|---|
| Memory Extraction | 是否正确提取偏好 | Precision / Recall / F1 |
| Memory Retrieval | 是否找到相关记忆 | Recall@K / MRR |
| Memory Update | 是否正确更新偏好 | Update Accuracy |
| Conflict Resolution | 是否正确处理新旧偏好 | Conflict Resolution Accuracy |
| Temporal Handling | 是否避免过期记忆干扰 | Stale Memory Rate |

## 10.2 推荐效果评测

进一步将 Memory 接入推荐系统，验证它对推荐效果的影响。

| 实验 | 系统 | 目的 |
|---|---|---|
| M0 | 不使用 Memory | 基础对照 |
| M1 | 静态 User Profile | 静态画像对照 |
| M2 | Mem0 动态 Memory | 验证动态记忆 |
| M3 | Mem0 + 写入准入规则 | 验证记忆质量控制 |
| M4 | Mem0 + 写入规则 + 时间及冲突处理 | 完整方案 |

主要指标：

- Recall@10。
- NDCG@10。
- 用户偏好匹配率。
- Constraint Violation Rate。
- Memory Retrieval Recall。
- Memory Update Accuracy。

所有对照实验使用相同的商品库、候选生成和推荐模型，避免其他模块变化影响 Memory 的实验结论。

# 十一、数据划分与防止数据泄漏

本模块必须采用时间顺序划分数据。

1. 使用时间截点之前的真实历史记录初始化 Persona 和 Memory。
2. 仅基于截点之前的信息生成模拟对话。
3. 使用模拟对话进行受控的 Memory 更新实验。
4. 保留截点之后的真实行为作为独立评测数据。
5. 测试期间不能提前将未来行为写入 Memory。

同时，所有模拟数据必须标记为 synthetic，不能混入真实行为统计。

# 十二、最终交付物

本模块最终需要提供：

1. `parent_asin` 到统一 `product_id` 的映射与校验。
2. SQLite `InteractionEvent` 表及按时间构建用户行为序列的接口。
3. Mem0 语义记忆与 Interaction History 分离的读写接口。
4. Amazon Reviews 用户历史行为处理脚本。
5. 用户 Persona 构建脚本。
6. DeepSeek 模拟购物对话生成脚本。
7. 模拟对话质量校验脚本。
8. Mem0 OSS 初始化与配置。
9. Memory Extraction 模块。
10. Memory Write / Update 模块。
11. Memory Retrieval 模块。
12. User Context Builder。
13. 统一 Memory 数据结构。
14. Memory 运行轨迹与来源记录。
15. Memory 独立评测脚本。
16. Memory 对推荐效果的消融实验。

# 十三、模块最终定位

User Memory 不再是一个静态用户画像生成器，而是一个能够持续接收用户信息、提取偏好、管理历史记忆并根据当前需求检索相关信息的动态记忆系统。

本模块采用真实历史行为初始化、模拟购物对话持续更新的方式，构建可持续演化的用户记忆。

最终向生成式推荐系统提供：

**Long-term Preferences + Dynamic Interests + Episodic Memories + Current Shopping Context**

为后续 Planner 的需求理解、商品召回和个性化推荐提供统一的用户上下文。