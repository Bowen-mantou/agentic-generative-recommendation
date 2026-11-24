# Reranker 模型选型与设计

## 1. 模块定位

**文档版本：** V1.1（跨模块职责修订版）

Reranker（精排模型）负责对召回阶段得到的候选商品进行二次排序，根据用户需求与商品信息之间的语义匹配程度，计算相关性分数，并输出重新排序后的商品列表。

在生成式推荐系统中，Reranker 位于 Planner 和最终推荐结果之间：

**Planner → Candidate Retrieval → Candidate Fusion → Python Hard Constraint Filter → Qwen3-Reranker → Judgment → Controller → Recommendation Generator**

各模块职责如下：

| 模块 | 核心职责 |
|---|---|
| Planner | 理解用户需求，提取商品属性、预算和个性化偏好 |
| Retriever | 从商品库中召回候选商品 |
| Reranker | 对候选商品进行精细化相关性排序 |
| Recommendation Generator | 结合排序结果生成推荐理由和最终推荐结果 |

Reranker 不负责生成检索计划、商品召回、硬约束过滤或最终推荐决策。其核心任务是：**在已经通过硬约束校验的候选商品中，根据当前需求和商品信息计算相关性分数并排序。** 商品是否满足语义偏好、属性证据是否充分由后续 Judgment 进一步判断；是否继续检索由 Controller 决定。

## 2. 模型选型需求

考虑到本项目采用生成式推荐架构，Reranker 不应仅仅判断 Query 和商品描述之间的表面语义相似度，还需要关注用户需求中的具体属性和个性化偏好。

主要需求包括：

1. **语义匹配能力**：理解自然语言购物需求与商品描述之间的关联。
2. **属性匹配能力**：识别品牌、价格、品类、功能、材质等商品属性。
3. **复杂需求理解**：处理多个偏好条件组合形成的商品筛选需求。
4. **指令适配能力**：能够根据 Planner 输出的任务指令调整排序侧重点。
5. **推理效率**：能够在较小的候选商品集合上完成批量精排。
6. **开源与可训练性**：支持本地部署，后续能够使用商品领域数据进行微调。

## 3. 候选模型对比

经过初步筛选，选择以下模型作为候选：

| 模型 | 参数量 | 特点 | 定位 |
|---|---:|---|---|
| Qwen3-Reranker-0.6B | 0.6B | 指令感知、多语言、长文本理解 | 首选 |
| BAAI/bge-reranker-v2-m3 | 约0.6B | 成熟的多语言 Cross-Encoder | 对照基线 |
| BAAI/bge-reranker-base | 约278M | 轻量、推理成本较低 | 轻量备选 |

### 3.1 第一选择：Qwen3-Reranker-0.6B

选择 Qwen3-Reranker-0.6B 作为项目的主力精排模型。

主要原因：

- **指令感知能力**：支持自定义 Instruction，可以根据商品推荐任务设计专门的排序指令。
- **多语言能力**：支持100多种语言，适合处理不同语言的用户需求和商品信息。
- **模型规模适中**：0.6B 参数量，在模型表达能力和推理开销之间具有较好的平衡。
- **部署方便**：官方支持 Sentence Transformers 的 CrossEncoder 接口，可以直接输入 Query-Product Pair 并输出相关性分数。
- **后续可优化**：可以利用商品领域的相关性标注和用户反馈进行领域微调。

模型采用基于生成式模型的判断方式，通过 yes/no 对应 Token 的分数计算相关性，而不是传统的单层分类头结构。

官方模型仓库采用 Apache-2.0 License。

### 3.2 第二选择：BAAI/bge-reranker-v2-m3

作为对照基线，选择 BAAI/bge-reranker-v2-m3。

主要优势：

- 具有成熟的 Cross-Encoder 精排架构。
- 支持多语言检索和相关性判断。
- 部署和推理方式相对简单。
- 可以直接输入 Query 和商品文本，输出相关性分数。

通过与 Qwen3-Reranker 对比，可以验证指令感知型 Reranker 在商品推荐场景中是否具有实际收益。

### 3.3 轻量备选：BAAI/bge-reranker-base

如果推理资源或延迟成为主要瓶颈，可以考虑使用 BGE-Reranker-Base。

该模型约278M 参数，属于 XLM-RoBERTa-base 系列的 Cross-Encoder，具有较低的模型部署成本。

但考虑到项目希望研究生成式模型与个性化需求的结合，暂时不将其作为首选。

## 4. 最终模型选择

综合模型能力、部署成本和后续实验空间，确定：

| 模块 | 模型选择 | 用途 |
|---|---|---|
| 主模型 | Qwen3-Reranker-0.6B | 商品精排 |
| 对照模型 | BAAI/bge-reranker-v2-m3 | 通用精排基线 |
| 轻量备选 | BAAI/bge-reranker-base | 效率对照 |

Qwen 官方公布的部分通用检索评测结果如下：

| 模型 | MTEB-R | CMTEB-R |
|---|---:|---:|
| Qwen3-Reranker-0.6B | 65.80 | 71.31 |
| BGE-Reranker-v2-m3 | 57.03 | 72.16 |

可以看到，Qwen3 在英文 MTEB-R 上表现更好，而 BGE 在中文 CMTEB-R 上略有优势。

需要注意，这些是通用检索基准结果，并非商品推荐任务的直接实验结果，最终选型仍需要通过本项目的数据集进行验证。

## 5. 输入输出设计

为了让 Reranker 更好地利用 Planner 提供的用户需求信息，采用以下输入结构：

### 5.1 Instruction

定义排序任务：

```text
Rank products according to the user's
shopping requirements and preferences.

Prioritize explicit requirements and
relevant product attributes.

Consider the user's stated preferences
when comparing candidate products.

Do not assume missing product attributes
are satisfied.
```

### 5.2 Query

使用 Planner 生成的结构化需求构建排序 Query。

例如：

```text
User Intent:
Find a lightweight daily moisturizer.

Must-have Constraints:
- Category: Moisturizer
- Price: Under $30

Preferences:
- Fragrance-free
- Lightweight texture
- Suitable for daily use
```

其中，价格上限和品类等可确定性硬约束已经由 Candidate Fusion 后的 Python Constraint Evaluator 执行。进入 Reranker 的主候选集原则上应为硬约束 `pass` 的商品；`unknown` 商品不得被当作已满足约束。Query 中仍可保留用户约束作为语义排序上下文，但不能用 Reranker 分数代替硬约束校验。

### 5.3 Document

将候选商品转换为统一格式：

```text
Product ID: A001

Title:
Daily Moisturizing Cream

Category:
Skincare

Brand:
Brand A

Price:
$25.99

Attributes:
- Lightweight
- Fragrance-free
- Daily moisturizing

Description:
A lightweight moisturizing cream
designed for everyday skincare.
```

通过统一商品文本格式，减少不同商品信息组织方式带来的输入差异。

对于缺失的商品属性，明确保留未知状态，不根据商品名称或其他字段自行推断。商品属性文本必须来自 Product Catalog；不要为了提高相关性而让文本构造器补写未提供的属性。

### 5.4 输出

Reranker 为每个 Query-Product Pair 输出一个相关性分数，并保留统一内部主键 `product_id`：

```text
Product P001: 5.82
Product P002: 3.17
Product P003: -1.24
```

按照分数降序排列，得到精排结果，并将 `product_id`、原始分数、排名和召回来源写入 `RankedProduct`。该结果随后交给 Judgment 逐商品判断；Reranker 的排名不是最终是否可推荐的唯一依据。

注意：Qwen3-Reranker 默认输出原始相关性分数，并非经过校准的概率值。分数主要用于同一 Query 下的候选商品排序，不应直接将其解释为用户购买概率。

## 6. 工程实现

采用 Sentence Transformers 的 CrossEncoder 接口进行调用。

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "Qwen/Qwen3-Reranker-0.6B",
    prompts={
        "shopping": """
        Rank products according to the user's
        shopping requirements and preferences.
        Prioritize explicit requirements and
        relevant product attributes.
        """
    },
    default_prompt_name="shopping"
)

pairs = [
    (query, product_text)
    for product_text in product_texts
]

scores = model.predict(pairs)

ranked_products = sorted(
    zip(product_ids, scores),
    key=lambda x: float(x[1]),
    reverse=True
)
```

通过批量构建 Query-Product Pair，减少逐条调用带来的额外开销。

## 7. 后续训练与优化

第一阶段暂时冻结 Qwen3-Reranker-0.6B，不立即进行模型微调。

先建立完整的商品精排基线，验证通用预训练模型在目标推荐场景中的表现。

第二阶段根据评测结果考虑商品领域微调。

训练数据可以组织为：

```text
User Query

Positive Product:
与用户需求高度匹配的商品

Hard Negative:
与用户需求相似，但在关键属性上不匹配的商品
```

Hard Negative 优先从同品类、相似品牌、相似价格区间的候选商品中挖掘，而不是简单随机采样。

训练目标可以考虑：

- Pointwise Classification：学习商品与用户需求是否匹配。
- Pairwise Ranking Loss：学习正例商品应当排在负例商品之前。
- Listwise Ranking Loss：直接优化同一用户 Query 下的候选商品排序。

具体采用哪种训练目标，需要根据标注数据规模和标签形式确定。

## 8. 实验评测方案

为了验证模型选择的合理性，设置以下对照实验：

| 实验 | 模型 | Instruction | 领域微调 |
|---|---|---|---|
| E1 | BGE-Reranker-v2-m3 | 默认 | 否 |
| E2 | Qwen3-Reranker-0.6B | 默认 | 否 |
| E3 | Qwen3-Reranker-0.6B | 商品推荐指令 | 否 |
| E4 | Qwen3-Reranker-0.6B | 商品推荐指令 | 是 |

主要评测指标：

| 评测维度 | 指标 | 目的 |
|---|---|---|
| 排序质量 | NDCG@10 | 衡量 Top-10 商品的排序质量 |
| 排序质量 | MRR@10 | 衡量首个相关商品的排名 |
| 推荐效果 | HitRate@10 | 衡量 Top-10 是否包含相关商品 |
| 推荐效果 | Precision@10 | 衡量 Top-10 商品的相关比例 |
| 效率 | Latency P50/P95 | 衡量推理延迟 |
| 效率 | Throughput | 衡量批量推理吞吐量 |

此外，召回阶段需要单独评估 Recall@K，避免将召回不足导致的问题错误归因于 Reranker。

所有模型应使用相同的候选商品集合、Query、商品文本字段和评测标签，确保对比公平。

## 9. 最终方案总结

本项目第一阶段采用 Qwen3-Reranker-0.6B 作为主力精排模型，使用 BAAI/bge-reranker-v2-m3 作为对照基线。该 Qwen3 模型仅用于 Reranker，不是 Judgment Model；Judgment 第一阶段使用 DeepSeek API。

暂时冻结模型参数，通过 Planner 提供的结构化用户需求和统一格式的商品信息进行精排。

待完成基础实验后，再根据商品领域的排序误差构造 Hard Negative，开展领域微调。

最终希望验证：

**指令感知型 Reranker 能否更有效地利用生成式 Planner 提取的用户需求与个性化偏好，从而提升商品推荐的排序质量。**

### 参考资料

1. [Qwen3-Reranker-0.6B 官方模型仓库](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
2. [BAAI/bge-reranker-v2-m3 官方模型仓库](https://huggingface.co/BAAI/bge-reranker-v2-m3)
3. [Qwen3 Embedding 与 Reranking 技术论文](https://arxiv.org/abs/2506.05176)