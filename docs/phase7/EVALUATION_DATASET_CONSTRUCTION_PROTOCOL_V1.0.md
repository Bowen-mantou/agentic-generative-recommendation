# Evaluation Dataset Construction Protocol

## 生成式推荐系统统一评测数据构建规范

-   版本：V1.0
-   日期：2026-11-24
-   状态：方案已达成一致，待工程实现与验证
-   适用阶段：Phase 4--7
-   数据范围：Amazon Reviews 2023 -- All_Beauty
-   Query 模型：DeepSeek PFlash（实际模型 ID 由配置固定）

> 本规范用于构建统一、可追溯、无未来信息泄漏的评测数据，公平比较
> TIGER-only、E5-only 和
> Hybrid。本文中的离线标签是下一次评论/交互商品，不代表真实购买转化。

------------------------------------------------------------------------

## 1. 目标与已确认方案

核心研究问题：

**在相同用户、相同历史信息、相同预测目标和相同候选预算下，TIGER 能否为
E5 召回提供额外有效候选，并提升 Hybrid 效果？**

  项目          已确认方案
  ------------- ------------------------------------------------
  主评测集      统一用户时间截点评测集
  Query 生成    Python 规则画像 + DeepSeek PFlash 自然语言改写
  Query 审核    DeepSeek PFlash 独立审核调用
  人工参与      异常审核 + PASS 样本随机抽检
  Profile V1    商品元数据 + Rating；默认不使用评论全文
  Target 隔离   Generator 和 Auditor 均不得接触 Target
  召回对照      TIGER-only、E5-only、E5 + TIGER
  评测目标      同一条样本的同一个下一交互商品
  数据范围      All_Beauty，不扩展 Q5 品类

### 1.1 基本原则

1.  三路召回必须使用相同 `sample_id`、用户历史和 Target。
2.  Profile 和 Query 只能使用 `cutoff_time` 之前的信息。
3.  Target 标签独立存储，不进入 Profile、Generator 或 Auditor 的输入。
4.  每条 Profile Fact 和 Query 都能追溯来源、模型版本和 Prompt 版本。
5.  批量生成和审核由 DeepSeek PFlash 完成；人工不负责批量生产。
6.  无法构造可信 Query 的样本可以标记不可用，不得强行生成。
7.  最终测试集、Query、审核结果和 Manifest 均需版本化。

------------------------------------------------------------------------

## 2. 总体数据生产架构

``` text
Amazon Reviews 2023 / All_Beauty
              |
              v
      Interaction Snapshot
              |
              v
     Temporal Sample Builder
              |
       +------+------+
       |             |
       v             v
  History Data   Target Labels
       |
       v
  Python Profile Extractor
       |
       v
  Structured User Profile
       |
       v
  DeepSeek PFlash - Query Generator
       |
       v
  Synthetic Query
       |
       v
  DeepSeek PFlash - Query Auditor
       |
       v
  Python Schema / Leakage Validator
       |
       +-----------------------------+
       |              |              |
       v              v              v
      PASS          REVIEW          FAIL
       |              |              |
       |              v              v
       |         Human Review    Regenerate Once
       |                             |
       |                       Still Failing
       |                             |
       +--------------+--------------+
                      |
                      v
             Final Evaluation Dataset
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       TIGER-only   E5-only     Hybrid
```

  模块                      实现                  职责
  ------------------------- --------------------- -------------------------------
  Temporal Sample Builder   Python                构造时间截点、历史和 Target
  Profile Extractor         Python 规则           提取画像事实
  Query Generator           DeepSeek PFlash API   生成自然语言购物 Query
  Query Auditor             DeepSeek PFlash API   独立审核 Query
  Schema Validator          Python                检查结构、字段和类型
  Leakage Validator         Python                检查时间和 Target 隔离
  Human Review              人工                  异常与抽样质检
  Dataset Freezer           Python                固化版本、Manifest 和校验结果

------------------------------------------------------------------------

## 3. Dataset A：统一用户时间截点评测集

### 3.1 样本定义

每条样本表示：在某个时间截点，系统只能看到用户过去的交互历史，并预测该用户下一次商品交互。

``` json
{
  "sample_id": "sample_000001",
  "user_id": "user_001",
  "cutoff_time": "2025-01-01T00:00:00Z",
  "history_item_ids": [
    "product_001",
    "product_002",
    "product_003"
  ],
  "history_event_ids": [
    "event_001",
    "event_002",
    "event_003"
  ],
  "profile_id": "profile_000001",
  "query_id": "query_000001",
  "is_synthetic": true,
  "split": "test",
  "dataset_version": "v1.0"
}
```

Target 不应直接写入提供给召回模型的样本文件。Target 标签独立存储：

``` json
{
  "sample_id": "sample_000001",
  "target_product_id": "product_004",
  "target_event_id": "event_004",
  "target_timestamp": "2025-01-05T12:00:00Z"
}
```

标签文件只授权给离线评测和指标计算程序。Generator、Auditor 和 Profile
Extractor 不得读取。

### 3.2 时间规则

``` text
history_event.timestamp <= cutoff_time
target_event.timestamp > cutoff_time
```

必须满足：

-   历史事件来自截点之前；
-   Target 是截点之后定义的下一次有效商品交互；
-   用户、商品和事件 ID 可映射到统一 Schema；
-   历史长度满足 TIGER 的输入要求；
-   不使用 Target 的未来评论、评分或行为统计构造 Profile；
-   不使用截点之后产生的 Memory 或用户画像。

如果一个用户有多个时间截点，必须分别保存每条样本的
`cutoff_time`。统计置信区间时按用户分组抽样。

### 3.3 复用现有 TIGER 测试集

优先审计现有约 2,685 条 TIGER 测试样本，检查能否恢复：

-   `sample_id`；
-   `user_id`；
-   `cutoff_time`；
-   `history_item_ids`；
-   `target_product_id`；
-   Target 时间；
-   商品目录映射；
-   原始 split 和数据版本。

上述字段可恢复、时间关系正确且商品映射有效，才允许复用。不能仅凭现有
Recall 结果反推测试样本。

如果无法恢复完整信息，重新构建统一时间切分，并在同一新测试集上重新评估
TIGER、E5 和 Hybrid。

### 3.4 重复交互与新商品子集

主评测保留真实发生的下一次交互，不因 Target 已在历史中出现就擅自删除。

同时分别报告： - `all_next_interactions`：全部符合条件的样本； -
`novel_item`：Target 此前未出现在该用户历史中的样本。

------------------------------------------------------------------------

## 4. Dataset B：Python 规则画像

### 4.1 第一版输入范围

只使用 `cutoff_time` 之前的：

-   商品 ID、Title、Category、Brand；
-   Details 中可用的商品属性；
-   Rating；
-   Timestamp。

**Profile V1 默认不使用历史评论全文。**
后续如需使用评论文本，应作为独立消融实验，不能悄悄改变 V1 定义。

### 4.2 画像类别

  画像类型              提取方式                     示例
  --------------------- ---------------------------- ------------------
  Category Interest     统计历史品类交互及正向反馈   Moisturizer
  Brand Interest        统计品牌交互和评分           Brand A
  Attribute Interest    汇总正向交互商品属性         Fragrance-free
  Negative Signals      汇总低评分商品及可识别属性   Strong fragrance
  Unknown Preferences   历史中没有足够证据的偏好     Budget unknown

### 4.3 Rating 规则

  Rating   初始解释
  -------- --------------
  4--5     正向交互信号
  3        中性或不确定
  1--2     负向交互信号

注意： - 有交互不等于喜欢； - 单次高评分不等于稳定偏好； -
低评分只能支持与该商品或已知属性有关的负向信号； -
缺失价格不能推断预算； - 缺失属性不能由模型补全。

### 4.4 Profile Schema

``` json
{
  "profile_id": "profile_000001",
  "sample_id": "sample_000001",
  "user_id": "user_001",
  "cutoff_time": "2025-01-01T00:00:00Z",
  "profile_version": "v1.0",
  "category_interests": [
    {
      "fact_id": "fact_001",
      "value": "Moisturizer",
      "support_count": 5,
      "source_event_ids": ["event_001", "event_002"],
      "confidence": "high"
    }
  ],
  "brand_interests": [],
  "attribute_interests": [],
  "negative_signals": [],
  "unknown_preferences": ["budget"]
}
```

`confidence` 必须由 Python 按确定性规则生成，不允许由 LLM
随意赋值。支持次数、评分、证据数量和置信度阈值在配置文件中定义，并在
Pilot 后校准。每条画像事实必须有来源事件、计算规则版本和 `fact_id`。

------------------------------------------------------------------------

## 5. Dataset C：DeepSeek PFlash Query Generator

### 5.1 职责与边界

Generator 只负责把结构化画像改写成自然、简洁的购物需求，不负责预测
Target 或推断未来行为。

允许输入： - `sample_id`、`cutoff_time`、`profile_id`； -
各类画像事实及证据元数据。

禁止输入： - `target_product_id`； - Target 商品标题、描述和属性； -
Target Event； - 未来评分、评论和交互。

**代码层面必须隔离 Target。** 不能只依靠 Prompt 文字要求模型忽略
Target。

### 5.2 输出 Schema

``` json
{
  "query_id": "query_000001",
  "sample_id": "sample_000001",
  "query": "I'm looking for a gentle, fragrance-free moisturizer.",
  "used_fact_ids": ["fact_001", "fact_003"],
  "status": "generated",
  "is_synthetic": true,
  "generator_model": "configured_deepseek_pflash_model",
  "prompt_version": "query_generator_v1",
  "generation_metadata": {
    "temperature": 0,
    "request_id": "request_xxx"
  }
}
```

### 5.3 Prompt 约束

1.  只能使用输入画像事实；
2.  不得创造没有证据支持的品牌、价格和功能偏好；
3.  不得使用时间截点之后的信息；
4.  不得要求或推测 Target 商品；
5.  不得将历史交互商品直接解释为当前明确购买意图；
6.  画像不足时生成宽泛 Query，或返回 `insufficient_profile`；
7.  Query 自然、简洁，像正常购物需求；
8.  每条样本只生成一个主 Query；
9.  返回 `used_fact_ids`；
10. 输出符合 JSON Schema。

### 5.4 生成与复现

-   每条样本默认只生成一个主 Query；
-   不得根据 Target 命中情况挑选 Query；
-   Query 离线生成并缓存；
-   模型标识、Prompt 版本、Temperature 固定；
-   验证集用于 Prompt 调试，Prompt 冻结后再生成测试集；
-   测试集生成后不得根据测试指标反复修改 Prompt；
-   画像不足时允许样本标记不可用。

每条 Query 记录模型版本、Prompt Hash、Profile 版本、请求
ID、生成时间、`used_fact_ids`、延迟和 Token 用量/成本（接口可提供时）。

------------------------------------------------------------------------

## 6. Dataset D：DeepSeek PFlash Query Auditor

### 6.1 独立审核原则

Generator 和 Auditor 可以使用同一 DeepSeek PFlash
模型系列，但必须使用不同 Prompt、独立调用和独立输出 Schema。Auditor
不读取 Generator 的解释或内部推理，只根据原始 Profile 事实检查 Query。

两者可能共享模型偏差，因此 Auditor
不是第三方裁判，必须保留人工抽样校准。

### 6.2 审核维度

  维度            检查内容
  --------------- --------------------------------
  Groundedness    Query 是否有 Profile 事实支持
  Hallucination   是否添加不存在的偏好或商品属性
  Contradiction   是否与历史正负向信号明显矛盾
  Specificity     是否过于宽泛或无证据地过度具体
  Naturalness     是否像正常购物需求
  Completeness    是否保留重要且有证据的需求

### 6.3 输出 Schema

``` json
{
  "audit_id": "audit_000001",
  "query_id": "query_000001",
  "sample_id": "sample_000001",
  "groundedness": "pass",
  "hallucination": false,
  "contradiction": false,
  "query_quality": "good",
  "unsupported_claims": [],
  "supported_fact_ids": ["fact_001", "fact_003"],
  "audit_status": "PASS",
  "reason": "",
  "auditor_model": "configured_deepseek_pflash_model",
  "audit_prompt_version": "query_auditor_v1"
}
```

### 6.4 审核状态

  状态     含义                               后续处理
  -------- ---------------------------------- ----------------------
  PASS     画像支持充分，Query 可用           进入 Python 最终校验
  REVIEW   存在边界问题或审核不确定           人工审核
  FAIL     存在明确幻觉、矛盾或严重质量问题   重新生成一次

Auditor 必须返回结构化字段和简短原因，不能只返回自然语言判断。Auditor
不接收 Target 标签或未来商品信息。

------------------------------------------------------------------------

## 7. Python Schema 与 Leakage Validator

### 7.1 Schema Validator

检查： - JSON 可解析； - 必填字段和类型正确； - `sample_id`、`query_id`
关联正确； - `used_fact_ids` 存在于当前 Profile； - Query
非空且长度符合配置； - `is_synthetic=true`； - 模型和 Prompt 版本存在。

### 7.2 Leakage Validator

检查： 1. 历史事件时间不晚于 `cutoff_time`； 2. Target 事件时间晚于
`cutoff_time`； 3. Profile 来源事件均来自历史； 4. Generator 输入 DTO
不含 Target 字段； 5. Auditor 输入 DTO 不含 Target 字段； 6. Profile
构建程序不能读取未来交互和未来评论； 7. Query 生成过程不能访问 Target
标签文件； 8. Prompt 构造不读取 Target 商品详情。

可以对 Query 做 Target 商品 ID 的精确字符串扫描。Target
完整标题精确匹配可标记异常；不能把常见品类词或共享属性词直接判为泄漏。

### 7.3 样本纳入条件

只有同时满足以下条件，样本才能进入主评测集： - 时间检查通过； - Schema
检查通过； - Profile 事实有效； - Generator 成功； - Auditor 为 PASS，或
REVIEW 经人工批准； - Leakage Validator 通过； - 商品 ID
能映射到冻结商品目录。

------------------------------------------------------------------------

## 8. 人工审核策略

### 8.1 人工工作范围

人工不承担批量 Query 生产，也不逐条审核全部样本。人工只处理： - Auditor
判为 REVIEW 的样本； - 重生成后仍 FAIL 的样本； - Python
校验发现的异常； - Generator 与 Auditor 结果冲突的样本； - PASS
样本中的随机抽检。

### 8.2 PASS 随机抽检

初始建议从 PASS 样本中随机抽取约 3%，并按品类、画像丰富度、Query
长度和模型版本分层抽样。

若测试集为 2,685 条，3% 约为 81
条。该比例是初始质量控制起点，不代表统计上足以证明所有错误都已被发现。

人工检查： - Query 是否忠实于 Profile； - 是否有无依据偏好； -
是否存在明显矛盾； - 是否自然； - 是否过宽或过窄； -
最终人工结论和错误类别。

若抽检发现系统性问题，暂停冻结，定位 Profile 规则、Generator Prompt 或
Auditor Prompt 的问题，修正后扩大抽检或针对性复审，并生成新数据版本。

### 8.3 覆盖率与选择偏差

必须报告： - 原始样本数； - Profile 可构建数； - Query 成功生成数； -
Auditor PASS 数； - REVIEW 数； - 人工批准数； - 最终有效样本数； -
不可用样本数及原因。

同时比较最终有效样本与原始样本的历史长度、品类分布和评分分布。

TIGER-only
完整测试集结果与三路共同有效子集结果应分别报告。核心增量比较必须使用三路共同有效的配对样本，避免
Query 生成失败导致样本选择偏差。

------------------------------------------------------------------------

## 9. 失败、重试和排除规则

  情况                      处理
  ------------------------- ---------------------------------------
  Generator API 超时        按 API 重试策略重试并记录
  Generator 返回非法 JSON   Schema 修复或内容级重新生成，最多一次
  Profile 信息不足          标记 `insufficient_profile`
  Auditor PASS              进入 Python 最终校验
  Auditor REVIEW            人工审核
  Auditor FAIL              重新生成一次，再独立审核
  重生成后仍 FAIL           人工审核；无法修复则标记不可用
  Leakage 检查失败          隔离样本，禁止自动 PASS
  Target 或时间字段缺失     隔离样本，不进入主评测集

每条样本记录初次结果、重试次数、失败原因、审核历史、人工结果和最终纳入/排除原因。

不得为达到预期样本量而强制生成不可信 Query。

------------------------------------------------------------------------

## 10. 三路召回对照实验

### 10.1 输入与候选预算

  实验         输入                 召回方式
  ------------ -------------------- -----------------------------------
  TIGER-only   `history_item_ids`   TIGER 生成 Semantic ID 并解码商品
  E5-only      `synthetic_query`    E5 Embedding + FAISS
  Hybrid       同一样本的两种输入   TIGER + E5 候选融合

第一版候选预算建议： - E5-only：Top-100； - TIGER-only：Top-100； -
Hybrid：E5 Top-100 + TIGER Top-100，去重后 RRF 融合，最终截断 Top-100。

三组最终候选池规模相同。Hybrid 的召回计算成本更高，必须单独报告。TIGER
无法生成足够合法候选时，记录实际数量，不得用虚假候选补齐。

### 10.2 Reranker 对照

-   三组召回使用同一个 Cross-Encoder Reranker；
-   使用相同商品特征和硬约束；
-   使用相同 Top-K 输出规则；
-   同时报告召回阶段和 Reranker 后指标；
-   不得只报告 Hybrid 结果而省略 E5-only 对照。

### 10.3 指标

**召回：** Recall@10/50/100、候选覆盖率、合法解码率。

**排序：** NDCG@K、MRR、目标商品最终排名。

**互补性：** Candidate Overlap、TIGER 独有候选数、Hybrid
新增有效候选数、Incremental Recall、Rescue Rate。

``` text
Incremental Recall@K
    = Recall_Hybrid@K - Recall_E5@K
```

``` text
Rescue Rate
    = TIGER 找回且 E5 未找回的 Target 数
      / E5 未找回的 Target 总数
```

Rescue Rate 分母为零时报告 `N/A`。

**成本：** TIGER 生成延迟、E5 检索延迟、Fusion 延迟、Reranker
延迟、峰值内存、CPU/GPU 使用、API 次数和费用。离线 Query
生成成本单独统计。

### 10.4 统计检验

-   三路按相同 `sample_id` 配对；
-   报告命中数量和绝对指标差值；
-   使用配对 Bootstrap 计算置信区间；
-   同一用户有多个测试样本时使用用户 Cluster Bootstrap；
-   置信区间跨越零时标记 `inconclusive`，不能声称已证明增益。

------------------------------------------------------------------------

## 11. 人工自然语言检索辅助集

另建小规模人工自然语言检索集，建议初始 100--300 条人工 Query
和相关商品标注，用于独立检查 E5 的自然语言语义检索能力，并辅助评估
Planner / Reranker。

该集合与 Synthetic Query
主评测集分开维护、分开报告，不能把两者指标合并。

------------------------------------------------------------------------

## 12. 文件组织规范

``` text
data/
└── evaluation/
    ├── raw/
    │   └── interaction_snapshot.parquet
    ├── samples/
    │   ├── train_samples.parquet
    │   ├── val_samples.parquet
    │   └── test_samples.parquet
    ├── labels/
    │   └── target_labels.parquet
    ├── profiles/
    │   ├── val_profiles.jsonl
    │   └── test_profiles.jsonl
    ├── queries/
    │   ├── val_queries.jsonl
    │   └── test_queries.jsonl
    ├── audits/
    │   ├── val_audits.jsonl
    │   └── test_audits.jsonl
    ├── human_review/
    │   ├── review_queue.jsonl
    │   └── human_audit.csv
    ├── reports/
    │   ├── coverage_report.json
    │   ├── leakage_check.json
    │   └── quality_report.json
    └── manifests/
        └── evaluation_manifest.json
```

Target 标签文件必须与 Query 生产目录隔离。训练集是否需要生成
Query，由具体训练任务决定；主评测构建不应无条件为所有训练样本调用 LLM。

### 12.1 Manifest

`evaluation_manifest.json` 至少记录：

``` json
{
  "dataset_version": "v1.0",
  "source_dataset": "Amazon Reviews 2023",
  "category": "All_Beauty",
  "split_method": "temporal",
  "profile_version": "v1.0",
  "query_generator": {
    "model": "configured_deepseek_pflash_model",
    "prompt_version": "query_generator_v1"
  },
  "query_auditor": {
    "model": "configured_deepseek_pflash_model",
    "prompt_version": "query_auditor_v1"
  },
  "embedding_model": "intfloat/e5-base-v2",
  "catalog_version": "configured_catalog_version",
  "random_seed": 42,
  "target_isolated": true,
  "is_synthetic_query": true
}
```

样本数量、模型版本和目录版本必须由实际构建程序填充，不得使用示例值冒充真实统计。

------------------------------------------------------------------------

## 13. 实施顺序与交付物

  -----------------------------------------------------------------------------------------------------------
  阶段                    工作                                 交付物
  ----------------------- ------------------------------------ ----------------------------------------------
  A. 数据审计             审计现有 2,685 条 TIGER              `temporal_split_audit.json`
                          样本，验证时间、历史、Target         
                          和商品映射                           

  B. Profile Pilot        抽取 100 条样本，构建并验证画像事实  `profile_pilot_report.md`

  C. Query Pilot          固定 PFlash 模型与 Prompt，批量生成  `query_generation_pilot.jsonl`
                          100 条 Query                         

  D. Audit Pilot          独立审核、Python                     `query_audit_pilot.jsonl`、`human_audit.csv`
                          校验、异常处理和随机抽检             

  E. 数据冻结             固定版本、全量生成、审核、Manifest   `evaluation_manifest.json`
                          和数据校验                           

  F. 三路评测             TIGER-only、E5-only、Hybrid 和统一   `retrieval_comparison_report.md`
                          Reranker 对照                        
  -----------------------------------------------------------------------------------------------------------

100 条 Pilot 用于验证数据流程，不代表足够支撑最终统计结论。

------------------------------------------------------------------------

## 14. 验收标准

### 数据正确性

-   每条样本有唯一 `sample_id`；
-   用户、商品和事件映射有效；
-   历史事件不晚于截点，Target 晚于截点；
-   Target 与 Query 生产流程隔离；
-   同一测试样本可同时提供给 TIGER 和 E5。

### Profile 质量

-   所有画像事实均有来源事件；
-   Rating 规则固定且版本化；
-   缺失属性不被补造；
-   Profile 只使用截点之前的信息；
-   置信度由 Python 规则生成。

### Query 质量

-   每条有效 Query 关联 `sample_id`；
-   标记 `is_synthetic=true`；
-   记录 `used_fact_ids`；
-   Generator 不接触 Target；
-   Auditor 独立审核；
-   Python Schema 和 Leakage 检查通过；
-   REVIEW 经人工处理；
-   覆盖率和排除原因有报告。

### 评测正确性

-   三路使用相同样本和 Target；
-   商品目录版本固定；
-   候选预算和 Reranker 一致；
-   报告三路指标、互补性、成本和统计区间；
-   Synthetic Query 结果不被描述为真实用户 Query 效果。

------------------------------------------------------------------------

## 15. 待工程验证事项

以下事项不阻塞协议确定，但在全量数据冻结前必须给出结果：

1.  现有约 2,685 条 TIGER 测试样本能否恢复完整时间截点、历史序列和
    Target；
2.  商品目录是否包含商品可用时间，能否构造历史时点可用的候选目录；
3.  DeepSeek PFlash 当前 API 的模型标识、结构化输出能力、限流和成本；
4.  Python Profile Extractor 的属性抽取覆盖率；
5.  Auditor 在人工抽检上的错误类型和通过率。

这些事项必须写入 Pilot 报告，不得用假设结果替代。

------------------------------------------------------------------------

## 16. 最终执行原则

**Python 负责事实提取、结构校验和数据隔离；DeepSeek PFlash
负责批量自然语言生成和自动审核；人工负责异常判断与抽样质量校准。**

最终使用同一批用户时间截点样本，对 TIGER-only、E5-only 和 Hybrid
进行配对比较，判断 TIGER 是否真正为 E5 带来增量价值。

未经评测证据支持，不得因为 TIGER 已完成训练就默认批准全量训练。
