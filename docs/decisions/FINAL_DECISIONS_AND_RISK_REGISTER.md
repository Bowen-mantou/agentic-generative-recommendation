# Agentic Generative Recommendation：最终决策记录与技术风险清单

-   版本：V1.0
-   决策日期：2026-11-24
-   状态：Q1--Q12 已由项目负责人确认；技术验证持续进行
-   依据：负责人上传的 `open-questions.md` 与本轮明确确认的 12 项决策

## 1. 决策摘要

本文件是当前项目的统一架构决策记录。已确认的选择不得由 AI
编程助手自行更改；验证失败时必须提交证据、影响分析和替代方案，等待负责人批准。

项目目标是在 Amazon Reviews 2023 的 `All_Beauty` 商品数据上实现完整的
Agentic Generative Recommendation。第一阶段不训练
Planner、Judgment、Generator 等 Agent
模型；通过完整运行轨迹和评测数据，为后续独立模型训练积累数据。

``` text
User Input
  -> User Memory (Mem0 + Qdrant)
  -> Planner / Replanner (DeepSeek API)
  -> Candidate Retrieval
       -> TIGER
       -> E5 + FAISS
  -> Candidate Fusion (Dedup + RRF)
  -> Python Hard Constraint Evaluator
  -> Qwen3-Reranker-0.6B (Transformers)
  -> Judgment (DeepSeek API)
  -> Adaptive Controller (LangGraph + Python)
       -> REPLAN -> Planner / Retrieval
       -> STOP -> Recommendation Generator (DeepSeek API)
  -> User Feedback
  -> Mem0 Update
```

### 12 项已确认决策

  -------------------------------------------------------------------------------
  ID                      决策主题                最终决定
  ----------------------- ----------------------- -------------------------------
  Q1                      TIGER 实现路线          B：外部实现/权重优先；A：自建
                                                  RQ-VAE + 自回归生成作为备用

  Q2                      Reranker 加载与设备     Transformers 原生加载；CPU
                                                  作为第一条稳定基线

  Q3                      Constraint Schema       整体三态 + `ConstraintCheck[]`
                                                  明细

  Q4                      E5 checkpoint           保留 `intfloat/e5-base-v2`

  Q5                      数据品类                仅使用 `All_Beauty`

  Q6                      Controller 校准         Amazon 时间切分数据 +
                                                  人工/模拟审核集

  Q7                      Memory 部署             Mem0 + 独立 Qdrant + 本地
                                                  Embedding

  Q8                      Git 边界                建立独立项目 Git 仓库

  Q9                      启动提示词              合并为唯一正式
                                                  `STARTUP_PROMPT.md`

  Q10                     存储                    NFS 存原始数据和大产物；Git
                                                  存代码与文档

  Q11                     MUSA                    第一阶段不接入 MUSA，使用 CPU
                                                  基线

  Q12                     第一阶段验收            C：完成 Phase 0--6
                                                  完整系统；Phase 7 单独评测
  -------------------------------------------------------------------------------

------------------------------------------------------------------------

## 2. 正式决策记录

### Q1. TIGER：B 优先，A 备用

1.  首先核查外部 TIGER 复现、代码和权重。
2.  核查许可证、代码可运行性、权重、商品集合兼容性、Semantic ID
    映射、数据格式和依赖环境。
3.  外部实现不兼容时，再评估自建最小链路：商品向量 → RQ-VAE → Semantic
    ID → 用户行为序列 → 自回归生成 → Semantic ID 解码。
4.  自建前必须提交训练时长、显存/设备、用户序列分布和人时估算。
5.  不允许用随机 Semantic ID、规则映射或 Mock 冒充真实生成式召回。

**重要边界：** A 是备用路线，不代表 B
失败后可以不经评估就自动启动完整自建。如果自建成本超出预算，必须提交可行性报告并由负责人重新确认。

**当前已知情况：** 仓库审计时没有 TIGER 实现和预训练权重；已有
All_Beauty 商品目录 112,590 个商品及 768 维 E5 向量；Reviews
文件尚未下载，用户历史交互序列尚未建表。

### Q2. Reranker：Transformers 原生加载，CPU 基线

-   主模型：`Qwen3-Reranker-0.6B`。
-   使用 Transformers 原生模型和 tokenizer，不把 Sentence Transformers
    `CrossEncoder` 作为默认加载实现。
-   按模型实际架构和模型卡要求构造输入，验证 Yes/No token 的打分方式。
-   第一阶段先在 CPU 上验证正确性和延迟。
-   MUSA 不作为第一阶段依赖。
-   `BAAI/bge-reranker-v2-m3` 保留为对照模型，不替代主模型。

必须验证模型架构、输入模板、Yes/No token ID、logits
取值方式、截断长度、批量大小、CPU 内存和端到端延迟。

### Q3. Constraint Schema：整体三态 + 明细数组

字段约定：

``` json
{
  "constraint_status": "pass",
  "constraint_checks": [
    {
      "field": "price",
      "status": "pass",
      "expected": "<= 30",
      "observed": 25.99
    },
    {
      "field": "category",
      "status": "pass",
      "expected": "moisturizer",
      "observed": "moisturizer"
    }
  ]
}
```

状态仅允许： - `pass`：已验证满足。 - `fail`：已验证违反。 -
`unknown`：缺少信息，无法验证。

聚合规则： - 任一适用硬约束为 `fail`，整体为 `fail`。 - 没有
`fail`，但至少一个适用硬约束为 `unknown`，整体为 `unknown`。 -
所有已配置硬约束均为 `pass`，整体为 `pass`。 -
没有配置任何硬约束时，`constraint_checks=[]`
表示"未配置"；整体状态应缺省/可空，不得伪装为已验证通过。枚举本身仍只保留三态，不增加
`not_required` 第四种状态。

硬约束由 Python 唯一执行。Judgment
可以读取约束结果，但不得重新实现价格或数值比较。

### Q4. E5：保留 `intfloat/e5-base-v2`

-   保留当前 768 维商品向量和 FAISS 索引。
-   保留 `query:` / `passage:` 前缀约定。
-   不因未经验证的精度预期重建索引。
-   只有未来明确提出多语言或精度需求，并完成对照实验后，才评估替换。

### Q5. 数据：仅 All_Beauty

-   第一阶段不扩展多品类。
-   内部商品主键统一为 `product_id`，维护与 Amazon `parent_asin` /
    `asin` 的映射。
-   缺失价格不得默认满足预算约束，必须标记 `unknown`。
-   商品元数据、评论和用户交互事件分开管理；评论文本不能直接等同于用户偏好事实。

当前数据快照：商品目录 112,590 个商品；`price` 覆盖率约
15.7%，`description` 约 17.0%，`details` 约 96%，`brand` 约
89.9%。属性缺失必须进入约束和评测设计。

### Q6. Controller：真实时间切分 + 人工/模拟审核

两类数据承担不同任务：

1.  Amazon 时间切分数据：用于离线推荐质量评测，如
    Recall@K、NDCG@K、MRR。
2.  人工/模拟审核集：用于 REPLAN/STOP
    决策评测，覆盖候选不足、硬约束失败、属性未知、质量达标、预算耗尽等场景。

规则： - 不得把下一商品预测标签直接当作 REPLAN/STOP 标签。 -
模拟反馈必须标记为模拟数据。 -
不得把模拟数据表现写成真实用户转化效果。 - 当前
`replan=2`、`retrieval_calls=6`、`api_calls=60`、`required_count=5`、`min_valid=3`
等只是初始值，未校准前不得称为最优阈值。 -
阈值、预算、验证集版本写入配置和实验记录。

### Q7. Memory：Mem0 + 独立 Qdrant + 本地 Embedding

数据边界： - Mem0：长期偏好、短期兴趣、明确表达的反馈等语义记忆。 -
SQLite InteractionHistory：有时间顺序的结构化行为事件，供 TIGER
构造用户序列。 - LangGraph State：当前请求和会话上下文。

初始部署建议： - Qdrant 独立服务并使用持久化数据卷。 - 单 Collection +
`user_id` 过滤作为初始多用户隔离方案。 - Embedding
本地运行，避免记忆读写依赖外部 Embedding API。 - Mem0、Qdrant、Embedding
的版本和向量维度锁定在配置中。

具体 Embedding checkpoint 尚未确定，需验证语言覆盖、维度、Mem0
适配接口和 Qdrant 配置。

### Q8. Git：独立项目仓库

-   项目代码、配置模板、文档、测试和小型脚本进入独立仓库。
-   不提交到已有 `C:/` 根仓库。
-   初始化前检查父目录 Git 状态，避免误纳入父仓库的无关修改。
-   不提交原始数据、模型权重、大型 FAISS 索引、API Key 或用户敏感数据。
-   实验记录代码 commit 或等效代码版本。

### Q9. STARTUP_PROMPT：唯一正式版本

-   根目录 `STARTUP_PROMPT.md` 是唯一权威版本。
-   合并现有两个版本中仍有效的仓库审计、阶段规划和汇报规范。
-   统一 Phase 0--7 编号。
-   `docs/` 不再保留内容分叉的第二份正式版本；历史版本放入 archive
    并标注废弃。
-   启动提示词要求 AI
    先审计仓库，再提出计划；未经确认不得大规模重写代码。

### Q10. 存储：NFS + Git 分离

  内容                            位置
  ------------------------------- ------------------------
  源代码、配置模板、文档、测试    Git
  Amazon 原始数据、处理后大数据   NFS
  HuggingFace Cache               NFS
  Embedding、FAISS 索引           NFS
  TIGER / RQ-VAE 权重             NFS
  大型实验日志和预测结果          NFS
  最终摘要、指标表、实验报告      Git（必要时 NFS 归档）

已知环境：`/home` 可用空间有限；`/mnt/agr-data` 是大容量
NFS，已测得可写，但备份和持久性策略未确认；`/tmp`
重启后清空，不得作为唯一产物存储位置。

关键产物应保存 manifest、数据版本、模型版本、配置和校验和。

### Q11. MUSA：第一阶段不接入

-   第一阶段使用 CPU 基线。
-   复用已有 E5-base CPU 编码和索引产物。
-   不为第一阶段增加 MUSA、FAISS ABI 和 NumPy 兼容性维护负担。
-   MUSA 可作为后续性能优化实验，不属于第一阶段验收依赖。

**范围说明：** 该决定不等于已证明 TIGER 自建训练或所有推理任务都能在 CPU
上满足时延要求。Q1 自建路线必须单独评估训练资源。

### Q12. 验收边界：Phase 0--6 完整系统

-   Phase 0：工程骨架、依赖、配置、Schema、日志。
-   Phase 1：All_Beauty 数据、E5 + FAISS。
-   Phase 2：Candidate Fusion、去重、RRF、硬约束。
-   Phase 3：Qwen3-Reranker-0.6B。
-   Phase 4：真实 TIGER 生成式召回。
-   Phase 5：DeepSeek Planner / Judgment、LangGraph Controller。
-   Phase 6：Recommendation Generator、Mem0、反馈更新和端到端集成。
-   Phase 7：离线评测、消融实验、延迟/成本分析和最终报告。

阶段性验收： 1. Phase 0--1：基础工程和单路 E5 召回可运行。 2. Phase
2--3：融合、硬约束、Reranker 可运行。 3. Phase 4--6：双路召回和完整
Agentic 推荐闭环可运行。 4. Phase 7：用可复现数据验证各模块贡献。

如果 TIGER 真实路线最终无法在预算内完成，不得把 E5
单路系统标记为完成了双路 TIGER + E5 的 Phase 0--6
验收。必须提交偏差说明并由负责人重新批准范围。

------------------------------------------------------------------------

## 3. 尚需验证的技术风险

以下是已选方案的实施风险，不是重新开放架构选择。

  ---------------------------------------------------------------------------------------------------------------------------------------------------
  风险 ID           风险                                    严重程度          验证方式与通过标准
  ----------------- --------------------------------------- ----------------- -----------------------------------------------------------------------
  R1                外部 TIGER 与 All_Beauty 商品集合不兼容 极高              核查许可证、代码、权重、Semantic ID
                                                                              映射和商品覆盖；真实用户序列可生成并解码为真实 `product_id`

  R2                自建 TIGER 训练成本超预算               极高              下载/抽样 Reviews，统计用户序列长度；估算
                                                                              RQ-VAE、生成器训练时长和设备需求；小规模训练后再批准扩展

  R3                Reviews 下载或用户序列构建存在问题      高                验证文件完整性、时间戳、重复事件、用户/商品映射；稳定生成按时间排序的
                                                                              `InteractionEvent`

  R4                Qwen3-Reranker Transformers             高                核对模型架构、官方模板、Yes/No token ID 和 logits；固定样例 sanity
                    原生打分实现错误                                          check

  R5                CPU Reranker 延迟/内存不满足要求        高                在目标服务器测 batch、候选数量、文本长度、吞吐、P50/P95 延迟和峰值内存

  R6                Constraint Schema 聚合语义不一致        高                覆盖全通过、任一失败、部分未知、无约束四类单元测试

  R7                e5-base-v2 与实际 Query 语言不匹配      中高              检查 Query
                                                                              语言；构建人工相关性小集评测当前召回；如有多语言需求再单独评估

  R8                商品价格/描述缺失影响约束判断           高                统计字段覆盖率；缺失字段输出 `unknown`；报告可验证样本和未知样本比例

  R9                Controller 阈值缺少标注数据             高                建立 REPLAN/STOP 标注规范和审核集；报告混淆矩阵、错误案例及预算消耗

  R10               Mem0、Qdrant、本地 Embedding            高                锁定版本；测试写入、检索、更新、删除、重启恢复和向量维度
                    版本或维度不兼容                                          

  R11               Qdrant 用户过滤未正确隔离记忆           高                使用多个测试用户交叉读写；验证查询、更新、删除均受 `user_id` 限定

  R12               InteractionHistory 与 Mem0 语义记忆混用 高                独立定义 SQLite 事件表和 Mem0 schema；模拟事件显式标记 `is_simulated`

  R13               NFS 备份、持久性和并发写入策略不明      高                确认备份/快照；测试重启恢复；关键模型和索引保留可恢复副本

  R14               Git 初始化误操作影响父仓库              中高              检查父目录 Git 状态；在独立目录建仓；确认 `.git` 路径和 ignore 规则

  R15               合并后的 Startup Prompt                 中                只保留根目录权威版本；全文检查 Phase 0--7、模型和接口一致
                    仍有阶段或架构冲突                                        

  R16               将"不接 MUSA"误解为 TIGER               高                Q11 限定为第一阶段 CPU 基线；Q1 自建路线单独评估设备和训练时长
                    训练资源已解决                                            

  R17               DeepSeek 模型 ID、JSON 输出、配额或 API 中高              固定模型 ID；测试结构化输出、超时、重试、限流、Token 成本和降级
                    行为变化                                                  

  R18               LangGraph                               高                对 REPLAN、STOP、预算耗尽、重复 Query、API
                    状态更新或循环边导致重复检索/无限循环                     失败做集成测试；所有路径受预算和轮数限制

  R19               时间切分或目标商品信息泄漏              极高              按时间切分；检查未来评论、目标商品信息、Memory 和 Prompt
                                                                              未泄漏测试标签；固定数据快照和随机种子

  R20               模拟反馈被误报为真实偏好/购买行为       高                模拟数据携带来源和
                                                                              `is_simulated`；真实行为指标只使用真实事件；报告中区分两类结果
  ---------------------------------------------------------------------------------------------------------------------------------------------------

### 风险状态定义

-   `Open`：尚未开始验证。
-   `In Progress`：正在验证。
-   `Mitigated`：已有缓解措施，但仍需持续观察。
-   `Accepted`：负责人接受剩余风险。
-   `Closed`：已有测试或证据证明风险已解决。

------------------------------------------------------------------------

## 4. 验证优先级与执行顺序

### Gate 1：数据与关键模型可行性

优先处理 R1、R3、R4、R5、R10、R11。

通过标准： - TIGER 外部路线有明确兼容性结论。 - Reviews
用户序列可以构建。 - Reranker 能真实打分。 - Memory 能够隔离读写。

### Gate 2：数据契约与控制逻辑

优先处理 R6、R9、R12、R18。

通过标准： - 模块间 Schema 固定。 - REPLAN / STOP 可重复测试。 -
无约束、未知约束和失败约束行为一致。 - 预算和最大轮数可以阻止无限循环。

### Gate 3：存储与可复现性

优先处理 R13、R14、R15、R17。

通过标准： - 代码可回滚。 - 数据和模型可恢复。 - AI
编码入口只有一份权威提示词。 - API 错误有明确降级路径。

### Gate 4：完整系统与评测可信度

优先处理 R7、R8、R19、R20。

通过标准： - Phase 0--6 完整链路可运行。 - Phase 7 指标可复现。 -
模拟数据和真实数据严格区分。 - 未完成能力如实披露。

------------------------------------------------------------------------

## 5. 当前初始配置值

以下配置可作为工程起点，但在校准前不得称为最优值：

  ------------------------------------------------------------------------
  配置                                        初始值 状态
  --------------------- ---------------------------- ---------------------
  最大 Replan 次数                                 2 未校准

  最大 Retrieval Calls                             6 未校准

  最大 API Calls                                  60 未校准

  Required Product                                 5 未校准
  Count                                              

  Minimum Valid                                    3 未校准
  Products                                           

  E5 内部 RRF k                                   60 初始默认

  TIGER/E5 跨路 RRF k                             60 初始默认

  E5 checkpoint                `intfloat/e5-base-v2` 已确认

  E5 向量维度                                    768 已有索引

  E5 编码设备                                    CPU 已确认基线

  Reranker                     `Qwen3-Reranker-0.6B` 已确认主模型

  DeepSeek 模型            当前配置 `deepseek-flash` 需验证模型 ID
                                                     和接口可用性
  ------------------------------------------------------------------------

所有阈值和预算必须配置化，不得散落在 Python 业务逻辑中。

------------------------------------------------------------------------

## 6. 变更管理规则

1.  本文件记录的 12 项决策视为已确认，AI 编程助手不得自行更改。
2.  技术验证发现问题时，先创建风险记录或
    ADR，不得直接替换核心模型和服务。
3.  若某项决策确实不可行，必须提交失败证据、受影响阶段、替代方案、人时/资源/迁移成本及实验可比性影响。
4.  由项目负责人批准后，更新本文件和相关模块文档。
5.  每个阶段结束后更新风险状态，并附验证记录和测试结果。
6.  `Mitigated` 不代表风险消失。
7.  不得把"代码能运行"当成"模型有效"，也不得把 Mock
    联调当成真实模型验收。

------------------------------------------------------------------------

## 7. 建议立即执行的下一步

1.  将本文件保存到
    `docs/decisions/FINAL_DECISIONS_AND_RISK_REGISTER.md`。
2.  更新 `AGENTS.md`，同步 Q1--Q12 的已确认选择和禁止擅自变更规则。
3.  合并根目录与 `docs/` 中两份
    `STARTUP_PROMPT.md`，只保留一份正式版本。
4.  优先执行 Gate 1，尤其是 TIGER 外部实现兼容性核查和 Reviews
    用户序列构建。
5.  在 Gate 1 有结论之前，不投入大量时间自建 TIGER。

**最终原则：架构决策已经确定；接下来通过可复现的工程验证降低风险。任何未经验证的能力都必须标记为待验证，不得写成已完成。**

---

## 7.1 D1–D11 补充决策（2026-11-24 确认）

项目负责人转交的 [Phase4-7_D1-D11 技术建议与决策建议稿](Phase4-7_D1-D11_DECISION_DRAFT_V1.0.md)
已作为本轮执行依据（原文标注“等待确认”，本轮按已确认处理；如有异议可逐项回退）。
要点：

| ID | 决策 |
|---|---|
| D1 | B：先 B1（3×256 + 序列 ≥2），再 B2（4×1024 + 序列 ≥3），受控实验 |
| D2 | 单次实验 ≤2 h；最多 1 组并行；全阶段 1 次配置修正重试；自动调参禁止 |
| D3 | A：安装 LangGraph 并让正式工作流经 LangGraph 运行，保留 Python 业务逻辑 |
| D4 | C：新增 `api_failed`，记录 `failed_component`（planner/judgment/generator） |
| D5 | B：严格 Hash（NFKC + casefold + 空白归一）检测，移除无效相似度配置 |
| D6 | B：Judgment 按 `product_id` 从 Catalog 按需补 description/details |
| D7 | C：Planner 降级可配置，默认结构化失败，显式降级需标记 degraded/plan_source |
| D8 | C：新增 `min_judgments`（初始 3），质量 STOP 需满足有效 Judgment 数量 |
| D9 | 独立 Qdrant 容器 + `intfloat/multilingual-e5-base`（记忆）+ 本地活动盘、NFS 备份 |
| D10 | C+A：人工 Query 集为主、Persona 合成 Query 为辅，三类协议分开报告 |
| D11 | A + 双路增益：TIGER 需超热度基线，且 Hybrid 对 E5-only 有可验证增益 |


**D1–D11 执行状态（2026-11-24）**

| ID | 状态 |
|---|---|
| D1 | A1/B1/B2 受控实验完成；**三项均未通过 D11 → No-Go**；未全量训练，Phase 4 未完成（[报告](../phase4/D1_B1_B2_GATE_REPORT.md)） |
| D2 | 遵守：单实验 <2 h、无并行训练、无自动调参；1 次配置修正 + 1 次实现 bug 修正已记录 |
| D3 | 完成：langgraph 1.2.24；正式入口走 CompiledStateGraph；9 个集成测试（服务器） |
| D4 | 完成：`api_failed` + `failed_component` + 重试成功不 STOP |
| D5 | 完成：NFKC+casefold+空白归一后 SHA-256 严格判重；移除无效相似度配置 |
| D6 | 完成：`JudgmentEvidenceBuilder`（Catalog 按需补 description/details，token 预算） |
| D7 | 完成：planner fallback 默认关闭；显式开启生成 degraded balanced 计划 |
| D8 | 完成：`min_judgments=3`；2 条不达标、3 条达标测试通过 |
| D9 | Qdrant 容器运行 + `multilingual-e5-base` 验证（中文 P50 16.2 ms）；`Memory.add` 端到端待 Phase 6 |
| D10 | 协议与模板完成（`docs/phase7/EVALUATION_PROTOCOL_D10.md`）；人工标注**尚未进行** |
| D11 | 执行完毕：条件一/二均不通过；偏差说明见 [Phase 4 报告](../phase4/D1_B1_B2_GATE_REPORT.md) |

## 8. 风险状态（执行记录）

按 §6 第 5 条维护；详细证据与逐条更新见
[`docs/audit/risk-status.md`](../audit/risk-status.md)。截至 2026-11-24（Gate 1）：

| 风险 | 状态 | 关键证据 |
|---|---|---|
| R1 | In Progress | 无 TIGER 端到端权重；6 个候选实现需改造（`docs/audit/tiger-compatibility-report.md`） |
| R2 | In Progress | 8,201 条 ≥3 用户序列；4×1024 Semantic ID 冲突 14.3%；训练资源为估算（`docs/audit/tiger-feasibility-assessment.md`） |
| R3 | Mitigated | 694,252 事件 / 631,986 用户 / 112,565 商品；0 跳过；SQLite 已构建 |
| R4 | Closed（有边界） | yes=9693/no=2152、官方模板三方一致；分数饱和不可当概率 |
| R5 | Mitigated | 512 token 单条 P50 0.41 s；20 候选 ≈ 6 s/query；峰值 RSS 3.9–5.2 GB |
| R6 | Mitigated | 约束三态 + 无约束为空状态；单测覆盖四种情形 |
| R14 | Closed | 独立 Git 仓库；凭据未入库；父仓库未改动 |
| R15 | Mitigated | 单一 STARTUP_PROMPT V1.2，旧版归档 |
| R8/R12/R13/R17 | In Progress | 字段覆盖统计、事件表、NFS、DeepSeek 模型列表已验证 |
| 其余（R7/R9/R10/R11/R16/R18/R19/R20） | Open | 见 `risk-status.md` |

Q1 状态说明：Q1-B 的“外部权重”已被核查证伪；负责人已确认 **B→A（重解释）+ A3 Pilot**。A1 Pilot 已执行：RQ-VAE 编码通过（100,702 唯一 ID，17.5% 碰撞），但生成器测试集 Recall@10 0.15% < 热度基线 0.34%、远低于 E5 用户画像 1.6%，未达进入全量训练的门槛；未启动全量训练，等待负责人选择 A2 / 数据范围 / 偏差方案（见 `docs/phase4/TIGER_PILOT_REPORT.md`）。
