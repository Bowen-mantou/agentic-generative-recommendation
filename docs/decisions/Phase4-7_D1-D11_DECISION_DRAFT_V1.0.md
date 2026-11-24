# Phase 4--7 待决事项技术建议与决策建议稿

-   版本：V1.0
-   日期：2026-11-24
-   状态：技术建议稿，等待项目负责人确认
-   依据：`OPEN_DECISIONS_PHASE4-7(1).md`

> 本文给出 D1--D11
> 的推荐选择、理由和实施要求。负责人确认前，均为建议，不视为已批准的正式决策。

## 1. 推荐决策总表

  --------------------------------------------------------------------------------------------------
  ID                      推荐选择                            核心理由
  ----------------------- ----------------------------------- --------------------------------------
  D1                      B：先 B1，再 B2                     分别诊断训练样本规模和 Semantic ID
                                                              表示容量，避免变量混杂

  D2                      单次实验 ≤2 小时；最多 1            限制试错成本
                          组并行；全阶段允许 1 次配置修正重试 

  D3                      A：LangGraph 正式运行               保留 Python
                                                              业务逻辑，同时满足已确认的编排架构

  D4                      C：新增 `api_failed`，记录失败组件  区分 Planner、Judgment、Generator 故障

  D5                      B：严格 Hash 检测                   当前没有 Query
                                                              原文，语义相似度配置实际无法工作

  D6                      B：Judgment 按 `product_id` 从      不扩大候选池契约，同时补足判断证据
                          Catalog 按需查询                    

  D7                      C：可配置，默认结构化失败           默认不伪造计划，同时保留显式降级能力

  D8                      C：增加 `min_judgments`，初始建议为 避免仅凭一条 Judgment
                          3                                   就判断整体质量达标

  D9                      独立 Qdrant 容器 + 本地多语言       服务独立、适配中文记忆，降低 NFS
                          Embedding；活动数据优先本地持久盘   活动数据库风险

  D10                     C+A：人工 Query 集为主，Persona     兼顾人工相关性可信度和个性化场景覆盖
                          合成 Query 为辅                     

  D11                     A + 双路增益条件                    TIGER 不仅要超过热度基线，还要证明能给
                                                              E5 带来增量价值
  --------------------------------------------------------------------------------------------------

## 2. 执行摘要

当前 A1 Pilot
的主要问题不是训练耗时，而是生成式召回效果和泛化能力不足。现有数据不支持直接扩大训练规模，也不支持直接放弃
TIGER。

建议： 1. 先通过 D1 的 B1、B2 受控实验定位问题； 2. 使用 D11
的统一测试协议判断 TIGER 是否值得全量训练； 3. Phase 5、Phase 6 继续使用
E5 候选池并行开发，不等待 TIGER； 4. 若 TIGER 未通过门槛，按 Q12
如实提交偏差说明，不得将 E5 单路描述为双路验收完成。

## 3. D1：A1 Pilot 未达门槛后的 TIGER 路线

**推荐：B------先 B1，再 B2。**

### 3.1 当前 Pilot 记录

  项目                                              当前记录
  ----------------------------- ----------------------------
  RQ-VAE 配置                                     3 级 × 256
  RQ-VAE 唯一 Semantic ID                            100,702
  ID 碰撞比例                                          17.5%
  RQ-VAE 验证集 MSE                                  1.87e-4
  RQ-VAE CPU 耗时                                      77 秒
  生成器参数量                                          470k
  生成器 CPU 耗时                                     111 秒
  TIGER 测试样本数                                     2,685
  TIGER Recall@10 / @50                        0.15% / 0.19%
  热度基线 Recall@10 / @50                     0.34% / 0.78%
  E5 用户画像 Recall@10 / @50     1.6% / 2.6%（仅 500 样本）
  序列长度 ≥3 的用户                                8,201 人
  序列长度 ≥2 的用户                               43,044 人

以上是待决文件中的实验记录，正式决策前仍应核对原始评测脚本、数据切分和指标计算口径。E5
与 TIGER 的测试样本数不同，不能直接横向比较。

### 3.2 两组受控实验

  实验   RQ-VAE 配置   用户序列门槛   相对 A1 改变的变量   主要问题
  ------ ------------- -------------- -------------------- ------------------------------
  B1     3×256         ≥2             只扩大用户序列范围   样本规模是否是主要瓶颈
  B2     4×1024        ≥3             只改变码本配置       Semantic ID 表示容量是否不足

**先执行 B1：** - 保持 RQ-VAE 为 3×256； - 用户序列门槛从 ≥3 调整为
≥2； - 固定时间边界、测试 Query 和候选商品目录； - 观察
Recall@K、NDCG@K、MRR 是否改善。

**再执行 B2：** - 恢复序列门槛 ≥3； - 将 RQ-VAE 改为 4×1024； - 使用与
A1 相同的数据时间边界和测试协议； -
观察码本变化是否改善编码质量和生成效果。

**对照约束：** - 测试集、时间切分和候选商品目录固定； -
不得同时改变码本配置和序列门槛； -
若测试用户群不同，同时报告共同测试用户子集和扩展用户群结果； -
报告训练用户数、交互事件数和序列长度分布； - 不得只比较训练 Loss 或
RQ-VAE 重建误差，必须比较下游召回效果。

如果 B1、B2 均无明显改善，停止无边界调参，结合 D11 决定是否保留 Pilot
结果并提交偏差方案。扩展数据品类前必须先修改 Q5。

## 4. D2：TIGER Pilot 资源上限

**推荐：单次实验 ≤2 小时；最多 1 组并行；整个 Pilot 阶段允许 1
次配置修正重试。**

  资源维度              建议上限
  --------------------- --------------------------
  单次实验 wall-clock   ≤2 小时
  并行实验数            最多 1 组
  配置修正重试          整个 Pilot 阶段最多 1 次
  自动超参数搜索        不允许
  全量训练              必须先通过 D11

2 小时是单次任务上限，不是要求训练满 2 小时。B1、B2
两组实验加一次必要重试，最多三次正式运行，总计算时长上限约 6 小时。

只有定位到可修复问题时才允许重试，例如配置错误、数据加载错误、训练中断或已确认的实现
Bug。不能因为指标不理想就不断调参。

每次重试记录原始失败、日志、修正内容、配置差异和实验变量变化。

## 5. D3：Phase 5 是否必须使用 LangGraph

**推荐：A------安装 LangGraph，并让正式工作流通过 LangGraph 运行。**

当前已有纯 Python 状态机和 70 个 Phase 5
单测。保留已经验证的业务逻辑，不要为了接入 LangGraph 而整体重写。

``` text
             LangGraph Runtime
                     |
             State / Transitions
                     |
       +-------------+-------------+
       |             |             |
    Planner       Judgment      Controller
       |             |             |
       +-------------+-------------+
                     |
             Existing Python Logic
```

实施要求： 1. 安装项目已声明的 LangGraph 可选依赖； 2. 增加 LangGraph
Adapter； 3. 将现有 Python 函数封装为节点调用； 4. 正式运行路径必须通过
LangGraph； 5. 保留纯 Python 状态机作为业务逻辑和回退实现； 6.
保留现有单测，并增加 LangGraph Runtime 集成测试。

LangGraph 负责编排、状态传递和条件转移；Python
负责硬约束、预算检查和具体业务判断。

## 6. D4：API 失败的 StopReason 契约

**推荐：C------新增 `api_failed`，并区分失败组件。**

``` python
class StopReason(str, Enum):
    QUALITY_SATISFIED = "quality_satisfied"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    NO_VALID_CANDIDATES = "no_valid_candidates"
    RETRIEVAL_DEGRADED = "retrieval_degraded"
    API_FAILED = "api_failed"
```

运行轨迹示例：

``` json
{
  "stop_reason": "api_failed",
  "failed_component": "judgment",
  "error_type": "timeout",
  "retryable": true,
  "attempts": 2
}
```

`failed_component` 限制为 `planner`、`judgment`、`generator`。

API 重试成功时只记录恢复事件，不应提前
STOP。只有重试耗尽且工作流无法继续时，才将最终 `stop_reason` 设为
`api_failed`。Schema 变更后同步更新 Controller、日志、测试和文档。

## 7. D5：重复 Query 检测语义

**推荐：B------只做严格 Hash 相等检测。**

当前只有 `seen_query_hashes`，但配置中有
`duplicate_query_similarity: 0.95`。由于没有 Query
原文或文本表示，该阈值实际无法执行。

建议： - 保留 `seen_query_hashes`； - 删除或标记
`duplicate_query_similarity` 为未使用配置； - 统一 Query 归一化规则； -
对归一化文本计算 SHA256； - Hash 相同则判定为重复 Query。

``` python
import unicodedata

def normalize_query(query: str) -> str:
    query = unicodedata.normalize("NFKC", query)
    query = query.strip().casefold()
    query = " ".join(query.split())
    return query
```

第一阶段不建议保存全部 Query
原文来计算语义相似度，也不建议删除停用词或排序词语，以免改变需求语义。

## 8. D6：Judgment 输入的商品元数据范围

**推荐：B------Judgment 按 `product_id` 从 Catalog 按需查询。**

保持候选池基础字段：
`product_id / title / price / main_category / brand`。

Reranker 完成排序后，由 `JudgmentEvidenceBuilder` 根据 `product_id`
批量查询 Catalog，补充： - 截断后的 `description`； -
与当前需求相关的关键 `details` 属性； - 其他可验证商品信息。

``` text
Candidate Pool
      |
   Reranker
      |
 Top-N product_id
      |
Judgment Evidence Builder
      |
Catalog Batch Lookup
      |
Truncated Description + Relevant Details
      |
DeepSeek Judgment
```

要求： - 只为进入 Judgment 的 Top-N 商品查询详情； - Description
截断，Details 按需求筛选； - 设置 Token 预算； - 保留 `product_id`
对齐； - 缺失属性标记为 `unknown`； - Judgment
不得推断缺失价格、规格或功效。

## 9. D7：Planner 失败时的降级语义

**推荐：C------支持配置化降级，默认结构化失败。**

``` yaml
planner:
  fallback:
    enabled: false
    strategy: balanced
```

默认关闭降级。Planner 多次调用失败后，返回结构化错误（例如
`PlanGenerationError`），不生成看似由 LLM 成功产生的计划。

显式开启降级时，使用确定性的系统默认均衡计划，并标记：

``` json
{
  "degraded": true,
  "plan_source": "system_default",
  "assumptions": [
    "Planner API unavailable",
    "Using deterministic balanced retrieval"
  ]
}
```

模块二原有"多次失败降级到默认均衡计划"的表述应同步修改为：默认结构化失败；可选显式启用系统默认计划；降级计划不能伪装成
Planner API 成功结果。

## 10. D8：质量 STOP 是否必须有 Judgment

**推荐：C------增加 `min_judgments`，初始建议值为 3。**

``` yaml
controller:
  min_judgments: 3
```

初始值 3 是工程起点，不代表已经校准。

允许 `quality_satisfied` STOP 前，至少满足： 1. 有效 Judgment 数量达到
`min_judgments`； 2. 候选商品通过相应硬约束； 3. 当前候选质量达到
Controller 配置条件； 4. 没有必须继续检索的明确原因。

"有效 Judgment"定义为：结构化结果解析成功、`product_id`
与输入候选对齐、字段满足 Schema。

最终 Top-K 中每个商品都必须有对应有效 Judgment。预算耗尽必须使用
`budget_exhausted`，API 故障按 D4 使用
`api_failed`，不得伪装成质量达标。

## 11. D9：Qdrant 部署与 Memory Embedding

**推荐：Qdrant 独立容器 + 本地多语言
Embedding；活动数据优先使用持久化本地磁盘，NFS 用于备份。**

### Qdrant

独立容器与 Python
应用生命周期分离，便于服务管理、多人协作、持久化和用户隔离测试。只绑定本机或受控内网地址，不要直接暴露公网。

### Embedding

建议优先评估：

``` text
intfloat/multilingual-e5-base
```

职责分离：

  模型                              用途
  --------------------------------- -----------------------------------
  `intfloat/e5-base-v2`             现有商品向量、E5 + FAISS 商品召回
  `intfloat/multilingual-e5-base`   Mem0 用户记忆向量化与检索

用户 Memory
可能包含中文或中英混合购物对话，因此建议评估多语言模型。正式部署前验证
Mem0 自定义 Embedding
接口、本地推理、向量维度、输入前缀和中文检索质量。不同模型即使维度相同，也不能混用同一
Collection 中的向量。

### 数据卷

-   Qdrant 活动数据：优先持久化本地磁盘；
-   NFS：用于备份、快照和恢复副本；
-   如果只能使用 NFS
    作为活动目录，先完成写入、重启恢复、备份恢复和并发访问测试。

候选路径 `/mnt/agr-data/qdrant`
不能未经测试就认定适合承载活动数据库文件。

## 12. D10：Phase 7 的 E5 评测 Query 构造方式

**推荐：C + A------人工标注 Query 集为主要文本检索评测集，Persona 合成
Query 为补充。**

### 人工标注 Query 集

建议初始规模 100--300 条。每条包含自然语言购物 Query、需求约束、相关
`product_id`、相关性等级和必要的标注依据。

主要用于 E5 文本召回、TIGER + E5 Fusion、Reranker、Judgment 和 Planner
检索计划质量评估。

### Persona + 模拟购物 Query

根据时间截点之前的历史行为构建 Persona，再生成模拟购物 Query。必须： -
只使用预测时间点之前的历史； - 标记 `is_synthetic=True`； -
不泄漏未来评论或目标商品； - 与人工 Query 指标分开报告。

### 三类评测协议

  ------------------------------------------------------------------------------
  评测协议                用途                    标签来源
  ----------------------- ----------------------- ------------------------------
  人工自然语言 Query      文本检索与商品相关性    人工标注

  Persona 合成 Query      个性化推荐与对话场景    历史行为 + 模拟生成

  TIGER Next-item         历史序列预测下一商品    时间切分后的真实交互代理标签
  ------------------------------------------------------------------------------

Amazon Reviews
是评论/评分交互数据，离线下一商品预测结果不能直接解释为真实购买转化率。

## 13. D11：TIGER Pilot 进入全量训练的量化门槛

**推荐：选择 A，并增加 E5 + TIGER 双路增益条件。**

### 当前基线

-   TIGER Recall@10：0.15%；
-   热度基线 Recall@10：0.34%；
-   E5 用户画像 Recall@10：1.6%，但仅 500 个样本。

E5 与 TIGER 样本数不同，不能直接用当前 1.6% 作为判断基准。

### Go 条件

**条件一：TIGER 自身超过热度基线** -
相同测试样本、时间切分和候选商品目录； - Recall@10 和 Recall@50
均高于热度基线； - 报告命中数、指标差值和配对 Bootstrap 置信区间。

**条件二：双路 Fusion 对 E5 有增益**

在完全相同测试样本上比较： 1. E5-only； 2. TIGER-only； 3. E5 + TIGER。

要求 Hybrid 相较 E5-only 有可验证的 Recall 增益，NDCG
等排序质量不能明显退化；同时报告候选覆盖率、重复率和新增有效候选数量。统计证据不足时标记
`inconclusive`，不能写成已证明有效。

**条件三：真实生成链路有效** - Semantic ID 合法； - 解码成功； -
生成商品属于真实商品目录； - 资源消耗符合预算； - 运行过程可复现。

### No-Go 处理

若 B1、B2 均未达到条件： - 不启动 TIGER 全量训练； - 保留 Pilot
模型、日志、配置和评测报告； - 标记 Phase 4 未通过完整验收； - Phase
5、Phase 6 可继续使用 E5 候选池； - 按 Q12 提交偏差说明； - 不得宣称完成
TIGER + E5 双路召回验收。

## 14. 推荐实施顺序

  ---------------------------------------------------------------------------------------------
  顺序              工作                         关联决策          交付物
  ----------------- ---------------------------- ----------------- ----------------------------
  1                 固定评测协议和基线测试样本   D1、D10、D11      数据切分与评测配置

  2                 TIGER B1                     D1、D2            B1 实验报告

  3                 TIGER B2                     D1、D2            B2 实验报告

  4                 TIGER Go / No-Go 评审        D11               全量训练批准或 No-Go 记录

  5                 LangGraph Adapter            D3                Runtime 集成与测试

  6                 StopReason 与 Query 去重契约 D4、D5            Schema、配置和单测

  7                 Judgment Evidence Builder    D6                Catalog 查询与证据构造模块

  8                 Planner 降级和 Judgment 门槛 D7、D8            Controller 配置与集成测试

  9                 Qdrant 与本地 Embedding      D9                部署配置、持久化与隔离测试

  10                完成 Phase 7 评测集          D10               人工 Query 集与合成 Query 集
  ---------------------------------------------------------------------------------------------

Phase 5、Phase 6 可先使用 E5 候选池开发和联调，不必等待 TIGER。

## 15. 负责人确认清单

以下为建议值，等待负责人确认后才转为正式决策。

  ------------------------------------------------------------------------------------------------
  ID                      建议决策                                         状态
  ----------------------- ------------------------------------------------ -----------------------
  D1                      B：B1（3×256 + 序列≥2），再 B2（4×1024 +         待确认
                          序列≥3）                                         

  D2                      单次≤2小时；最多1组并行；全阶段1次配置修正重试   待确认

  D3                      A：LangGraph 正式运行，Python 逻辑保留           待确认

  D4                      C：新增 `api_failed`，记录 `failed_component`    待确认

  D5                      B：严格 Hash 检测，删除无效相似度配置            待确认

  D6                      B：Judgment 按 `product_id` 查询 Catalog         待确认

  D7                      C：默认结构化失败，可配置开启显式降级            待确认

  D8                      C：增加 `min_judgments`，初始值建议为 3          待确认

  D9                      独立容器 + `multilingual-e5-base` 候选 +         待确认
                          本地活动数据盘                                   

  D10                     C+A：人工 Query 集为主，Persona 合成 Query 为辅  待确认

  D11                     A + 双路增益条件                                 待确认
  ------------------------------------------------------------------------------------------------

建议回复：

``` text
D1=B
D2=2小时/最多1组并行/全阶段1次修正重试
D3=A
D4=C
D5=B
D6=B
D7=C
D8=C（min_judgments=3）
D9=独立容器 + multilingual-e5-base + 本地活动数据盘
D10=C+A
D11=A+双路增益条件
```

可以只修改不同意的项目，其余按建议确认。

## 16. 确认后的同步要求

负责人确认后，AI 编程助手应： 1. 更新
`FINAL_DECISIONS_AND_RISK_REGISTER.md`； 2. 更新 Phase 4--7
对应模块文档； 3. 更新 Pydantic Schema、Enum、State 和配置； 4.
更新单元测试和 LangGraph 集成测试； 5. 更新 `STARTUP_PROMPT.md` 或
`AGENTS.md` 中相关约束； 6. 输出文件变更清单、测试结果和未解决风险； 7.
保持 Q5（All_Beauty）和 Q11（第一阶段不接入 MUSA）不变； 8.
未经批准，不得将 E5 单路替代描述为双路召回验收完成。

## 17. 最终建议

当前项目不应因为已经投入 TIGER，就默认继续全量训练；也不应因为 A1
效果较弱，就立即放弃生成式召回。

更合理的做法是：用 B1、B2 定位问题；用统一测试集和 D11
门槛决定是否继续；让 Phase 5、Phase 6 与 TIGER Pilot 并行推进；用人工
Query、合成 Query 和 Next-item 三套独立评测协议验证不同能力。

**最终目标不是证明某个模型必须被保留，而是证明
TIGER、E5、Judgment、Controller 和 Memory
各自对完整生成式推荐系统产生了什么可测量的价值。**
