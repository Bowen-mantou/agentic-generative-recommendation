# 模块三：Candidate Retrieval 商品候选召回系统

## TIGER 生成式召回 + E5 向量召回 + Candidate Fusion

**项目名称：** Agentic Generative Recommendation

**模块名称：** Candidate Retrieval

**文档版本：** V1.1（统一数据与融合协议修订版）

**核心技术：** TIGER + Semantic ID + E5 + FAISS + RRF

**编排框架：** LangGraph

---

# 一、模块定位与设计目标

Candidate Retrieval 是生成式推荐系统的核心召回模块，负责根据用户历史交互和当前购物需求，从商品库中生成或检索一批候选商品，为后续 Product Reranker 提供候选集合。

本项目采用双路召回架构：

1. TIGER：基于 Semantic ID 的生成式商品召回。
2. E5 + FAISS：基于自然语言需求的语义向量召回。
3. Candidate Fusion：合并两路召回结果，形成统一候选商品池。

两路召回分别利用不同的信息来源：

- TIGER 主要利用从 SQLite `InteractionHistory` 读取的、按时间排序的用户商品交互事件序列，预测用户下一步可能感兴趣的商品。
- E5 + FAISS 主要利用 Planner 根据当前购物需求和 Mem0 User Memory 生成的自然语言 Query，检索语义相关的商品。

Mem0 语义记忆与 TIGER 行为事件序列是两种独立数据：TIGER 不从 Mem0 自由文本中构造序列；模拟事件默认不进入真实序列评测。

通过融合两路结果，兼顾用户历史行为兴趣与当前购物需求。

本模块的核心研究问题：

**生成式商品召回与语义向量召回是否具有互补性？两路召回融合能否提升候选覆盖率、推荐准确率和长尾商品召回能力？**

# 二、整体架构

```text
                User Input
                    |
                    v
          Recommendation Planner
                    |
          +---------+----------+
          |                    |
          v                    v
    Retrieval Strategy     E5 Queries
          |                    |
          v                    |
   User Interaction           |
      Sequence                 |
          |                    |
          v                    v
       TIGER                E5 Encoder
    生成式召回                  |
          |                    v
          |                FAISS Index
          |                    |
          v                    v
    Semantic ID           Product IDs
      Decoding                  |
          |                    |
          v                    |
      Product IDs               |
          |                    |
          +----------+---------+
                     |
                     v
              Candidate Fusion
                     |
          +----------+----------+
          |                     |
          v                     v
       ID Mapping             Dedup
          |                     |
          +----------+----------+
                     |
                     v
              RRF Fusion
                     |
                     v
             Constraint Filter
                     |
                     v
           Unified Candidate Pool
                     |
                     v
               Product Reranker
```

两路召回可以通过 LangGraph 并行执行，之后统一进入 Candidate Fusion。

# 三、TIGER：生成式商品召回

## 3.1 模块定位

TIGER 是本项目的生成式推荐路径。

其核心思想是：

将商品编码成多级 Semantic ID，再训练生成模型根据用户历史交互序列预测下一个商品的 Semantic ID。

与传统向量召回直接计算 Query 和商品之间的相似度不同，TIGER 将推荐问题转化成自回归序列生成任务。

本项目采用标准的 Next-Item Prediction 范式：

**根据用户历史交互商品序列，预测下一个可能感兴趣的商品 Semantic ID。**

## 3.2 商品 Semantic ID 构建

首先需要为商品库中的每个商品构建 Semantic ID。

### 第一步：商品信息处理

从 Amazon Reviews 2023 商品元数据中提取：

- 商品标题。
- 商品类别。
- 商品描述。
- 商品特征。
- 品牌及其他可用属性。

将商品信息组织成统一的商品文本表示。

示例：

```text
Product ID: A001

Title:
Daily Moisturizing Cream

Category:
Beauty / Skincare

Description:
Lightweight moisturizing cream,
suitable for daily use.

Features:
Fragrance-free, moisturizing.
```

### 第二步：商品向量化

使用预训练文本 Embedding 模型，将商品文本编码成稠密向量。

```text
Product Text
     |
     v
Text Embedding Model
     |
     v
Product Embedding
```

商品向量作为后续 RQ-VAE 的输入。

具体 Embedding 模型可以根据实验资源选择，不要求与 E5 在线检索使用的 Query Encoder 完全绑定。

### 第三步：RQ-VAE 量化

使用 Residual Quantized Variational Autoencoder（RQ-VAE）对商品向量进行多级离散量化。

基本过程：

1. 将商品向量输入编码器。
2. 通过多级码本进行残差量化。
3. 将连续向量转换成多个离散编码。
4. 得到商品对应的 Semantic ID。

示例：

```text
Product Embedding
        |
        v
      RQ-VAE
        |
        v
Residual Quantization
        |
        v
  Semantic ID

[12, 05, 31]
```

这里的 Semantic ID 仅为示意，不代表实际训练结果。

最终建立商品映射表：

| Product ID | Semantic ID |
|---|---|
| A001 | [12, 05, 31] |
| A002 | [08, 14, 22] |
| A003 | [03, 17, 09] |

Semantic ID 需要能够映射回真实商品 ID，供后续生成结果解码。

## 3.3 用户历史交互序列构建

Semantic ID 构建完成后，需要将用户历史交互转换成生成模型能够使用的序列。

本项目使用 Amazon Reviews 2023 中的用户评论、评分和时间信息构建历史交互序列。

首先：

1. 根据 user_id 从 SQLite `InteractionEvent` 表读取历史事件。
2. 只选择当前实验允许的事件类型和时间截点内的数据，并按 timestamp 排序。
3. 根据统一 `product_id` 映射到 Semantic ID。
4. 形成用户历史商品序列。

例如：

```text
原始用户历史：

Item A → Item B → Item C

转换为 Semantic ID：

[12, 05, 31]
[08, 14, 22]
[12, 05, 29]
```

最终得到：

```text
User History:

[
  [12, 05, 31],
  [08, 14, 22],
  [12, 05, 29]
]
```

注意：Amazon Reviews 提供的是评论和评分等交互数据，并非完整的点击、浏览和加购日志。因此本项目将其作为历史商品交互的代理信号，事件类型按实际来源记录为 `review` / `rating`，不将其描述成完整用户行为日志。每条事件需保留 `is_synthetic` 标记；第一版 TIGER 真实评测默认不混入模拟事件。

## 3.4 生成式推荐模型

采用自回归生成模型预测下一个商品的 Semantic ID。

训练样本构建：

```text
用户历史序列：

Item A → Item B → Item C → Item D

训练样本：

Input:
Item A → Item B

Target:
Item C

Input:
Item A → Item B → Item C

Target:
Item D
```

将商品替换为对应的 Semantic ID 序列。

模型学习：

```text
User History Semantic IDs
             |
             v
      TIGER Generator
             |
             v
     Next Semantic ID
             |
             v
       ID Decoder
             |
             v
        Product ID
```

模型训练目标是预测下一个商品对应的 Semantic ID。

训练过程中采用自回归生成目标，使模型能够逐级生成目标商品的多个离散编码。

## 3.5 TIGER 两阶段实施方案

考虑到本次课程设计需要优先打通整个推荐系统，因此不在第一阶段投入大量时间完整复现 TIGER。

采用 B → A 两阶段实施。

### 阶段 B：最小可运行生成式召回

目标：先实现真正能够生成 Semantic ID 的商品召回链路。

主要工作：

1. 完成商品数据预处理。
2. 使用预训练 Embedding 构建商品向量。
3. 使用现有兼容实现或轻量 RQ-VAE 构建 Semantic ID。
4. 构建用户历史 Semantic ID 序列。
5. 使用兼容的现有实现或轻量自回归生成模型完成 Next-Item Prediction。
6. 实现 Semantic ID 解码。
7. 输出 Top-K 候选商品。

模型实现优先级：

- 优先寻找能够适配当前商品数据和编码方式的现有实现。
- 如果没有合适的预训练权重，则实现轻量级 RQ-VAE 和生成模型。
- 不要求第一阶段完整复现原论文的全部训练配置和优化细节。

阶段 B 的目标是形成一条真实可运行的生成式召回链路，而不只是一个接口模拟。

在最初的工程联调阶段，可以使用 Mock 输出测试 LangGraph 接口，但 Mock 结果必须明确标记，不能用于计算正式召回指标或证明生成式推荐有效。

### 阶段 A：完整 TIGER 训练与优化

阶段 B 完成后，进一步开展：

1. 优化 RQ-VAE 训练。
2. 优化 Semantic ID 码本和量化配置。
3. 正式训练自回归生成模型。
4. 优化 Beam Search 和候选生成。
5. 研究不同 Semantic ID 长度的影响。
6. 研究不同码本配置对推荐效果的影响。
7. 开展长尾商品和冷启动商品实验。

阶段 A 主要研究 Semantic ID 表示和生成模型对推荐效果的影响。

## 3.6 TIGER 输入输出

| 项目 | 内容 |
|---|---|
| 输入 | 用户历史交互商品序列 |
| 数据表示 | Semantic ID 序列 |
| 模型 | 自回归生成模型 |
| 训练目标 | 下一商品 Semantic ID |
| 在线输出 | 多个生成的 Semantic ID |
| 解码 | Semantic ID → Product ID |
| 最终输出 | 候选商品 ID、生成分数和排名 |

建议统一输出：

```json
{
  "retrieval_source": "TIGER",
  "candidates": [
    {
      "product_id": "A001",
      "semantic_id": [12, 5, 31],
      "generation_score": 0.92,
      "rank": 1
    },
    {
      "product_id": "A002",
      "semantic_id": [8, 14, 22],
      "generation_score": 0.86,
      "rank": 2
    }
  ]
}
```

其中分数由实际生成模型的解码过程产生，不是 LLM 人工生成的分数。

## 3.7 TIGER 在整体系统中的职责边界

第一版采用标准 TIGER 行为序列生成范式。

TIGER 的主要输入是从 SQLite `InteractionHistory` 构建的用户历史商品交互序列，不直接接收 Planner 生成的自然语言 Query。

Mem0 User Memory 不直接转换成 TIGER 序列。TIGER 只使用有来源、有时间戳的结构化交互事件；当前自然语言需求和显式语义偏好则主要由 E5 召回路径利用。

Planner 可以决定 TIGER 的召回预算优先级，但不能自己生成 Semantic ID 或虚构用户历史商品序列。

后续可以探索 Memory-conditioned TIGER，但不属于第一版必要实现范围。

# 四、E5 + FAISS：语义向量召回

## 4.1 模块定位

E5 + FAISS 负责根据用户当前购物需求和 User Memory，在商品库中检索语义相关的商品。

与 TIGER 根据历史商品序列预测下一个商品不同，E5 主要处理自然语言需求。

例如：

用户提出：

```text
我想买一款适合日常使用的面霜，
预算在30美元以内，希望质地清爽，
最好没有明显香味。
```

Planner 结合 User Memory 后，生成检索 Query：

```text
lightweight daily moisturizer,
fragrance-free skincare cream
```

E5 将 Query 转换为向量，FAISS 根据向量相似度找到相关商品。

## 4.2 离线阶段：商品向量构建

首先从商品元数据构建统一商品文本。

示例：

```text
Product ID: A001

Title:
Daily Moisturizing Cream

Category:
Beauty / Skincare

Brand:
Brand A

Description:
Lightweight moisturizing cream,
suitable for daily use.

Features:
Fragrance-free, moisturizing.
```

随后：

```text
Product Catalog
      |
      v
Product Text Processing
      |
      v
E5 Passage Encoder
      |
      v
Product Embeddings
      |
      v
FAISS Index
```

每个商品只需要在离线阶段编码一次。

当商品元数据更新时，再对对应商品重新编码并更新索引。

## 4.3 在线阶段：Query 检索

Planner 根据用户当前需求和历史偏好生成一个或多个检索 Query。

示例：

```json
{
  "queries": [
    "lightweight daily moisturizer",
    "fragrance-free moisturizing cream",
    "daily skincare cream with strong moisturizing effect"
  ]
}
```

在线检索流程：

```text
Planner Queries
       |
       v
E5 Query Encoder
       |
       v
Query Embeddings
       |
       v
FAISS Similarity Search
       |
       v
Top-K Product IDs
```

对于多条 Query，分别执行向量检索，再在 E5 路径内部进行商品去重和排名合并。第一版采用内部 RRF，不直接相加不同 Query 的原始相似度分数。具体流程：

1. 每条 Query 独立检索 Top-K，并保留该 Query 下的商品排名。
2. 按 `product_id` 去重，记录商品出现过的 Query 和各自排名。
3. 使用 `RRF_E5(d) = Σ_q 1 / (k + rank_q(d))` 计算 E5 内部融合分数。
4. 按内部融合分数降序形成唯一 E5 候选列表，并重新编号为 E5 总排名。

E5 的具体 Query / Passage 前缀格式遵循最终选用的模型配置，并通过配置固定。

## 4.4 FAISS 检索

将所有商品向量建立 FAISS 索引。

第一版优先使用简单、容易验证的相似度检索方案。

对于使用余弦相似度的配置：

1. 对商品向量进行 L2 归一化。
2. 对 Query 向量进行 L2 归一化。
3. 使用内积计算相似度。
4. 根据相似度返回 Top-K 商品。

索引类型根据数据规模选择，先建立准确检索基线，再考虑更高效的近似最近邻索引。

## 4.5 E5 输入输出

| 项目 | 内容 |
|---|---|
| 输入 | Planner 生成的自然语言 Query |
| 商品表示 | 商品标题、类别、描述和属性 |
| Encoder | E5 |
| 向量索引 | FAISS |
| 检索方式 | 向量相似度检索 |
| 输出 | Product ID、相似度分数和排名 |

统一输出：

```json
{
  "retrieval_source": "E5_FAISS",
  "candidates": [
    {
      "product_id": "A001",
      "similarity_score": 0.91,
      "rank": 1
    },
    {
      "product_id": "A003",
      "similarity_score": 0.87,
      "rank": 2
    }
  ]
}
```

## 4.6 User Memory 的作用

E5 是两路召回中主要利用自然语言 User Memory 的路径。

例如：

当前需求：

```text
我想买一款面霜。
```

User Memory：

```json
{
  "positive_preferences": [
    "lightweight",
    "moisturizing"
  ],
  "negative_preferences": [
    "strong_fragrance"
  ]
}
```

Planner 融合后生成：

```text
lightweight moisturizing cream,
fragrance-free,
suitable for daily skincare
```

因此，E5 检索不仅考虑当前购物需求，也利用历史用户偏好。

但需要注意：

- 当前明确的硬性约束由后续 Python 规则执行。
- 历史偏好主要作为语义检索和精排信号。
- 商品缺失的属性不能仅凭向量相似度推断为满足要求。

# 五、Candidate Fusion：双路候选融合

## 5.1 模块定位

Candidate Fusion 负责将 TIGER 和 E5 + FAISS 返回的商品候选进行统一处理，形成后续 Reranker 可以直接使用的商品候选池。

主要工作：

1. 商品 ID 统一映射。
2. 候选商品去重。
3. 召回来源记录。
4. 融合分数计算。
5. 硬性约束过滤。
6. 统一候选格式输出。

## 5.2 第一步：商品 ID 统一

TIGER 输出 Semantic ID，需要通过商品映射表解码为统一内部商品主键 `product_id`。

E5 + FAISS 直接返回 `product_id`。Amazon 原始 `parent_asin` 在数据接入层映射为 `product_id` 并保留原始字段用于追溯。

两路结果及后续所有模块只使用 `product_id`；Semantic ID 仅属于 TIGER 内部表示。

建立：

```text
Semantic ID → Product ID
```

映射关系，确保两路召回可以比较同一个商品。

## 5.3 第二步：候选去重

假设 TIGER 返回：

| 商品 | 排名 |
|---|---:|
| A | 1 |
| B | 2 |
| C | 3 |
| D | 4 |

E5 返回：

| 商品 | 排名 |
|---|---:|
| B | 1 |
| E | 2 |
| F | 3 |
| A | 4 |

合并后：

```text
TIGER: A, B, C, D
E5:    B, E, F, A

Union:

A, B, C, D, E, F
```

A 和 B 只保留一个商品实体，但需要记录两条召回来源。

例如：

```json
{
  "product_id": "A",
  "retrieval_sources": [
    "TIGER",
    "E5_FAISS"
  ],
  "source_ranks": {
    "tiger": 1,
    "e5": 4
  }
}
```

不能因为去重而丢失商品的来源信息和原始排名。

## 5.4 第三步：RRF 融合排序

第一版采用 Reciprocal Rank Fusion（RRF）作为融合基线。

原因：

TIGER 输出的是生成模型的生成分数，E5 输出的是向量相似度分数。

两种分数来自不同的模型和计算过程，不能直接进行简单加权求和。

RRF 通过排名进行融合，不要求两路模型的原始分数处于相同数值空间。

计算公式：

\[
RRF(d)=\sum_{m\in\{TIGER,E5\}}
\frac{1}{k+rank_m(d)}
\]

其中：

- \(d\)：候选商品。
- \(m\)：召回来源。
- \(rank_m(d)\)：商品在对应召回路径中的排名。
- \(k\)：平滑常数。

示例：

商品 A：

- TIGER 排名第 1。
- E5 排名第 4。

商品 B：

- TIGER 排名第 2。
- E5 排名第 1。

RRF 根据两路排名计算融合分数，得到统一的候选顺序。

第一版采用两级 RRF：先在 E5 路径内对多条 Query 的结果进行 RRF，得到唯一 E5 排名；再将该 E5 排名与 TIGER 的生成排名进行跨路径 RRF。跨路径计算使用 `RRF(d) = 1/(k + rank_TIGER(d)) + 1/(k + rank_E5(d))`，缺失来源的项记为 0。不得把 TIGER 生成分数和 E5 向量相似度直接相加。

后续可以进一步实验：

- 不同 RRF 参数。
- Weighted RRF。
- 不同召回来源配额。
- 根据 Planner 策略动态调整融合权重。

这些优化不属于第一版必要实现。

## 5.5 第四步：硬性约束过滤

RRF 排序后，根据 Planner 提取的 Hard Constraints 使用 Python 规则过滤。

例如：

```json
{
  "hard_constraints": {
    "category": "Beauty",
    "max_price": 30
  }
}
```

商品必须满足：

- 商品类别符合要求。
- 价格不超过预算。

每项硬约束的校验结果统一为 `pass` / `fail` / `unknown`：

- `pass`：字段存在且满足约束，进入 `eligible_candidates`。
- `fail`：明确违反约束，从可推荐候选中剔除，并保留诊断记录。
- `unknown`：缺少必要字段或无法确定，不得当作满足约束；保留在 `unverified_candidates` 供 Controller 判断是否需要补充检索或说明限制。

只有全部必需硬约束均为 `pass` 的商品才能计入有效候选数量。过滤后如果候选数量不足，交由 Controller 决定是否扩大召回或触发 Replan。硬约束规则由一个 Python Constraint Evaluator 实现，其他模块只读取结果，不重复实现价格或品类比较。

软性偏好不应直接作为硬性过滤条件。

例如用户历史上喜欢无香型商品，但本次没有明确要求无香型，则应优先作为后续 Reranker 的排序信号。

## 5.6 最终候选池格式

统一输出：

```json
{
  "request_id": "req_001",

  "candidates": [
    {
      "product_id": "A001",
      "fusion_score": 0.0325,
      "constraint_status": "pass",
      "retrieval_sources": ["TIGER", "E5_FAISS"],
      "source_ranks": {"tiger": 1, "e5": 4},
      "raw_scores": {"tiger": 0.92, "e5": 0.91},
      "product_metadata": {
        "title": "Daily Moisturizing Cream",
        "price": 25.99,
        "category": "Beauty"
      }
    }
  ],
  "unverified_candidates": []
}
```

最终商品信息通过 Product Catalog 根据 Product ID 获取。

Candidate Fusion 不负责生成商品推荐理由，也不负责最终商品精排。

# 六、Planner 与双路召回的协作

Recommendation Planner 负责决定两路召回的侧重点。

第一版设置三种召回策略：

| 策略 | TIGER | E5 + FAISS | 适用场景 |
|---|---|---|---|
| Hybrid Balanced | 标准预算 | 标准预算 | 普通购物需求 |
| Behavior Priority | 扩展预算 | 标准预算 | 更关注历史行为 |
| Semantic Priority | 标准预算 | 扩展预算 | 更关注当前需求 |

Planner 输出来自模块二统一 `RecommendationPlan` 的 `retrieval` 子对象：

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

该片段仅展示 `retrieval` 子对象；不得把它作为完整 Planner 输出。完整计划必须遵循模块二第 6.2 节 `RecommendationPlan` Schema。

具体候选数量通过 Python 配置映射。

Planner 不直接决定任意召回数量，也不直接执行 FAISS 或 TIGER。

LangGraph 负责读取计划并调度两路召回节点。

## 6.1 召回降级

如果用户没有足够的历史交互记录：

- TIGER 无法正常生成时，不伪造历史序列。
- E5 + FAISS 继续根据当前需求和 User Memory 召回。
- 系统记录 TIGER 不可用的原因。

如果某一路召回服务发生故障：

- 保留另一条正常工作的召回路径。
- 将异常信息写入运行轨迹。
- 由 Controller 根据候选数量和质量决定是否继续。

# 七、与 LangGraph 的接口设计

LangGraph 负责调度召回和融合节点。

建议采用以下节点：

| 节点 | 职责 |
|---|---|
| build_user_sequence | 构建用户历史商品序列 |
| tiger_retrieval | 执行 TIGER 生成式召回 |
| e5_retrieval | 执行各条 E5 + FAISS Query 检索 |
| e5_query_fusion | E5 路径内部去重与 RRF 合并 |
| candidate_id_mapping | Semantic ID 解码及统一 product_id 映射 |
| candidate_deduplication | 商品去重 |
| candidate_fusion | RRF 融合 |
| constraint_filter | 硬性约束过滤 |
| candidate_validation | 检查候选池有效性 |

两路召回可以并行执行。

推荐的调用关系：

```text
Planner
   |
   v
LangGraph State
   |
   +-------------------+
   |                   |
   v                   v
TIGER Retrieval    E5 Retrieval
   |                   |
   +---------+---------+
             |
             v
      Candidate Fusion
             |
             v
      Constraint Filter
             |
             v
      Candidate Pool
             |
             v
        Reranker
```

# 八、实验设计与评测

本模块需要独立评测两路召回及其融合效果。

## 8.1 召回对照实验

| 实验 | 召回方案 | 目的 |
|---|---|---|
| E1 | E5 + FAISS | 向量召回基线 |
| E2 | TIGER | 生成式召回基线 |
| E3 | TIGER + E5 | 验证双路召回互补性 |
| E4 | TIGER + E5 + RRF | 验证融合策略 |
| E5 | TIGER + E5 + RRF + Reranker | 验证精排效果 |

## 8.2 评测指标

| 指标 | 作用 |
|---|---|
| Recall@50 | Top-50 候选覆盖能力 |
| Recall@100 | Top-100 候选覆盖能力 |
| Candidate Recall | 候选集合覆盖真实目标的比例 |
| MRR | 目标商品在候选排序中的位置 |
| NDCG@10 | 排序质量 |
| Candidate Coverage | 候选商品覆盖情况 |
| Long-tail Recall | 长尾商品召回能力 |
| Latency | 召回耗时 |

其中：

\[
Recall@K=\frac{|C_K\cap G|}{|G|}
\]

其中：

- \(C_K\)：召回的 Top-K 商品集合。
- \(G\)：评测样本对应的真实目标商品集合。

## 8.3 双路召回互补性分析

除了比较总体 Recall，还需要统计：

1. TIGER 独有召回商品数量。
2. E5 独有召回商品数量。
3. 两路重复召回商品数量。
4. 双路融合新增的目标商品覆盖量。
5. 不同用户历史长度下的召回效果。
6. 长尾商品的独立召回表现。

特别关注：

**联合召回是否能够覆盖单一路径无法召回的目标商品。**

如果两路召回的结果高度重合，那么双路架构带来的实际价值可能有限。

## 8.4 数据划分与防止数据泄漏

TIGER 使用用户历史交互预测下一商品，采用时间顺序划分训练集、验证集和测试集。

对每个用户：

- 训练阶段只使用训练时间段的历史交互。
- 验证阶段使用验证时间段的目标商品。
- 测试阶段使用测试时间段的目标商品。

E5 商品索引使用允许访问的商品目录构建，但评测 Query 不能使用目标商品的标题、描述或未来评论作为输入。

当前购物 Query 可以由基于历史截点之前信息构建的模拟购物对话提供。

所有 Memory 和模拟对话也必须遵守时间截点，不能提前读取测试期用户行为。

这样才能公平比较 TIGER 和 E5 的召回能力。

# 九、技术选型

| 模块 | 技术方案 |
|---|---|
| 商品文本处理 | Python |
| 商品 Embedding | 预训练文本 Embedding |
| Semantic ID | RQ-VAE |
| 生成式推荐 | 轻量自回归生成模型 |
| 生成式召回 | TIGER |
| 向量编码 | E5 |
| 向量索引 | FAISS |
| 候选融合 | RRF |
| 数据存储 | SQLite / 商品映射表 |
| 工作流编排 | LangGraph |
| 接口校验 | Pydantic |

# 十、项目阶段与最终交付

## 第一阶段：最小可运行链路（B）

必须完成：

1. 商品信息预处理并建立 `parent_asin` → `product_id` 映射。
2. 商品 Embedding。
3. Semantic ID 构建。
4. 用户历史 Semantic ID 序列构建。
5. 最小可运行的自回归生成模型（真实训练或可验证的兼容权重，不以 Mock 代替正式召回）。
6. TIGER 商品生成与解码。
7. E5 商品向量构建。
8. FAISS 索引。
9. E5 在线 Query 检索。
10. Candidate Fusion。
11. RRF 融合。
12. 统一候选池接口。
13. LangGraph 双路召回集成。

阶段 B 的核心验收标准：

- TIGER 能够根据真实用户历史序列生成 Semantic ID。
- Semantic ID 能够映射回真实商品。
- E5 能够根据自然语言 Query 检索商品。
- 两路结果能够统一合并和去重。
- 统一候选池能够交给 Product Reranker。
- 正式实验使用真实模型输出，而非 Mock 数据。

## 第二阶段：完整 TIGER 优化（A）

主要工作：

1. RQ-VAE 训练优化。
2. Semantic ID 码本配置实验。
3. 自回归生成模型训练优化。
4. Beam Search 优化。
5. 长尾商品召回实验。
6. 冷启动商品实验。
7. 不同生成参数的消融实验。

## 最终交付物

1. 商品数据处理脚本和统一 `product_id` 映射。
2. 商品 Embedding 构建模块。
3. RQ-VAE Semantic ID 编码模块。
4. Semantic ID 与商品 ID 映射表。
5. 用户历史交互序列构建模块。
6. TIGER 生成式召回模块。
7. E5 商品向量构建模块。
8. FAISS 商品索引。
9. E5 在线检索服务。
10. Candidate Fusion 模块。
11. RRF 融合实现。
12. LangGraph 召回节点。
13. 双路召回实验报告。
14. 长尾和冷启动实验报告。

# 十一、模块最终定位

Candidate Retrieval 采用 TIGER 与 E5 + FAISS 双路召回架构。

TIGER 根据用户历史商品交互序列生成 Semantic ID，实现基于历史行为的生成式商品推荐。

E5 + FAISS 根据用户当前购物需求和 User Memory 进行语义向量检索，实现基于自然语言需求的商品召回。

Candidate Fusion 将两路结果统一映射到商品 ID，通过去重和 RRF 排序形成统一候选商品池。

最终形成：

**Behavior-based Generative Retrieval + Semantic-based Vector Retrieval + Candidate Fusion**

该模块使系统同时具备基于历史行为的生成式推荐能力和基于当前自然语言需求的商品检索能力，为后续精排、结构化判断和自适应推荐提供候选商品基础。