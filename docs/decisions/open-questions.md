# 待负责人决策事项（Open Questions）

> 用途：集中列出**需要项目负责人决定**的事项。本文只陈述事实、可选方案与影响，
> 不含推荐或倾向性意见。
> 状态：12 项待定，1 项（Q12）关乎本轮验收边界。
> 相关文档：[审计与计划](../audit/repository-audit-and-plan.md) ·
> [字段级契约](../audit/design-contracts.md) ·
> [Phase 1 执行记录](../audit/phase1-report.md) ·
> [服务器环境](../audit/s4000-environment.md)

| ID | 待决问题 | 影响阶段 | 状态 |
|---|---|---|---|
| Q1 | TIGER 实现路线（自建 / 外部 / 暂缓） | Phase 4、双路召回 | 阻塞 Phase 4 |
| Q2 | Reranker 加载方式与运行设备 | Phase 3 | 待定 |
| Q3 | Judgment 输入的 `constraint_status` 形状 | Phase 5 | 待定 |
| Q4 | E5 checkpoint 选择 | Phase 1（需重建索引） | 已用默认值，待确认 |
| Q5 | 数据品类范围 | Phase 1/4/7 | 已用单品类，待确认 |
| Q6 | Controller 阈值与预算的校准数据 | Phase 5/7 | 待定 |
| Q7 | Mem0 / Qdrant 部署形态 | Phase 6 | 待定 |
| Q8 | 项目 Git 仓库边界 | 全流程 | 待定 |
| Q9 | 两份 `STARTUP_PROMPT.md` 的取舍 | 全流程 | 待定 |
| Q10 | 产物与原始数据的长期存储位置 | 运维 | 待定 |
| Q11 | MUSA 加速是否用于 Phase 1 编码 | Phase 1/7 | 待定 |
| Q12 | “第一阶段”的验收边界 | 全部 | 待定 |

---

## Q1. TIGER 实现路线

**已确认事实**

- 仓库内没有 TIGER 实现、没有预训练权重；六份文档也未给出 RQ-VAE 层数、码本大小、
  生成器 backbone 的具体配置（AGENTS.md §8.1 列为未验证事项）。
- 文档要求“阶段 B 完成最小真实生成链路，不得以 Mock 冒充正式实验”。
- 现有产物：All_Beauty 112,590 个商品目录 + 768 维 E5 向量，可作为 RQ-VAE 的输入候选。
- Amazon reviews 文件（326,611,506 B）尚未下载，用户历史序列尚未建表。

**可选方案（互斥）**

| 方案 | 内容 | 需要的前置核查 |
|---|---|---|
| A | 自建 RQ-VAE + 小型自回归生成模型，在 All_Beauty 上训练 | 训练时长估算、显存/设备（CPU 或 MUSA）、用户序列最小长度分布 |
| B | 引入外部 TIGER 复现或权重 | 许可证、embedding 与商品集合是否兼容、Semantic ID 映射方式、数据格式 |
| C | 暂缓 TIGER，先以 E5 单路推进 Phase 2/3/5，Phase 4 后置 | Phase 7 的 E2/E3/E4 实验改期方案 |

**影响**

- Phase 4 能否启动；模块三“双路召回”验收；Phase 7 的 TIGER/E5/融合对照实验；
  24 人时预算是否足够。
- 选择 C 时，跨路径 RRF 在正式实验中只有单路输入。

**需要负责人提供的输入**：Phase 4 是否在本轮验收范围内；外部权重/复现的可用性核查结论。

---

## Q2. Reranker 加载方式与运行设备

**已确认事实**

- 文档 §3.1 描述 Qwen3-Reranker 为“yes/no 对应 Token 的分数”，§6 使用
  `sentence_transformers.CrossEncoder(..., prompts=...)` 接口。
- 服务器 MUSA 上尚未验证 Qwen3-Reranker-0.6B；`dp` 容器没有 faiss/sentence-transformers。
- CPU 上 0.6B 模型的批量精排延迟尚未测量。

**可选方案**

| 维度 | 选项 |
|---|---|
| 加载/打分 | A. ST `CrossEncoder`（依赖该 ST 版本支持 Qwen3-Reranker）；B. `transformers.AutoModelForCausalLM` + yes/no logits 自行实现 |
| 运行设备 | CPU；MUSA（新建容器）；`dp` 容器（需另建 venv） |
| 主模型 | Qwen3-Reranker-0.6B（AGENTS 规定的主模型）；bge-reranker-v2-m3（文档规定的对照） |

**影响**

- Phase 3 的接口、批量大小、延迟与显存；Phase 7 效率指标；是否需要额外的冒烟实验。
- 若使用 MUSA，需要锁定 numpy/faiss 版本组合（见 Q11）。

**需要负责人提供的输入**：允许使用的设备与容器范围；是否允许新增长期运行的容器。

---

## Q3. Judgment 输入的 `constraint_status` 形状

**已确认事实**

- 本轮契约（已实现）为：整体三态枚举 + `ConstraintCheck[]` 明细。
- 模块三 §5.6 候选池示例是字符串 `"pass"`；最终阶段 §3.3 Judgment 输入示例是按字段的
  `{"price": "pass", "category": "pass"}` 字典；最终阶段 §3.5 的 Python 示例另有一个
  `not_required` 返回值。
- AGENTS.md §2.4 规定只有 `pass` / `fail` / `unknown` 三态。

**可选方案**

| 方案 | Judgment 收到的内容 |
|---|---|
| A | 整体三态 + `ConstraintCheck[]`（当前契约） |
| B | 逐字段 `dict[field, status]` |
| C | 两者同时传入 |

**影响**

- Phase 5 的 Judgment Prompt、输入 token 成本、输出与上游状态的一致性校验。

---

## Q4. E5 checkpoint 选择

**已确认事实**

- 当前配置与已建索引使用 `intfloat/e5-base-v2`（768 维，英文，`query: `/`passage: ` 前缀）。
- 服务器索引：112,590 向量、`IndexFlatIP`、345.9 MB；更换模型必须重建索引（CPU 编码约 17 分钟）。
- 文档未指定 E5 checkpoint，AGENTS.md §8.4 列为未验证事项。

**可选方案**：`intfloat/e5-base-v2`（当前）；`intfloat/e5-large-v2`（1024 维）；
`intfloat/multilingual-e5-base`；其他 checkpoint。

**影响**：索引维度与体积、重建时间、Phase 7 的对照实验可比性、多语言支持。

**需要负责人提供的输入**：是否要求多语言支持或更高检索精度目标。

---

## Q5. 数据品类范围

**已确认事实**

- 当前使用 `All_Beauty`：metadata 212,990,142 B、112,590 个商品、reviews 326,611,506 B
  （尚未下载）；字段覆盖：title/main_category 100%、brand 89.9%、details 96%、
  price 15.7%、description 17.0%。
- 其他品类规模更大（例如 Video_Games metadata 437,375,785 B 起）。

**可选方案**：仅 `All_Beauty`；多品类合并；其他单品类。

**影响**：数据下载与存储、catalog/索引重建时间、TIGER 训练成本、评测覆盖；
price 覆盖率会影响硬约束可用性。

**需要负责人提供的输入**：验收对品类数量与商品规模的要求。

---

## Q6. Controller 阈值与预算的校准

**已确认事实**

- 当前配置值（replan 2 次、检索 6 次、API 60 次、required_count 5、min_valid 3、
  RRF k=60 等）来自文档示例或本轮默认，`controller.quality.calibrated = false`。
- 时间切分方案、验证集来源、标注数据尚未确定。

**可选方案**：用现有 Amazon 交互按时间切分构造验证集；另建人工/合成标注集；
由负责人直接给定固定阈值与预算。

**影响**：REPLAN/STOP 的触发率、Phase 7 质量与预算结论的可信度。

**需要负责人提供的输入**：可用于校准的数据范围与划分方式。

---

## Q7. Mem0 / Qdrant 部署形态

**已确认事实**

- `dp` 容器没有 `mem0ai` / `qdrant-client`，Qdrant 尚未部署。
- AGENTS.md 固定方案为 Mem0 OSS + Qdrant + SQLite；Mem0 使用的 embedder 未指定。
- SQLite `InteractionEvent` 表结构尚未落地。

**可选方案**

| 维度 | 选项 |
|---|---|
| 向量库 | 独立 Qdrant 容器；嵌入式 Qdrant；其他兼容存储 |
| Embedder | E5（本地）；Mem0 默认 embedder |
| 多用户隔离 | 单 collection + `user_id` 过滤；每用户独立 collection |

**影响**：Phase 6 的依赖、运维成本、多用户隔离方式、与训练容器的耦合度。

**需要负责人提供的输入**：是否允许新增长期运行的服务容器。

---

## Q8. 项目 Git 仓库边界

**已确认事实**

- 项目目录未被跟踪，位于 `C:/` 根仓库（分支 `verl-migration`）之下；该根仓库有大量与本
  项目无关的未提交修改与删除。
- 本轮未执行任何 git 写操作；项目自身目前没有版本历史。

**可选方案**：在项目目录 `git init`；把项目纳入现有根仓库；保持无版本控制。

**影响**：回滚能力、协作方式、实验 commit 记录（AGENTS.md §6 要求记录代码 commit）。

---

## Q9. 两份 `STARTUP_PROMPT.md` 的取舍

**已确认事实**

- 根目录版本 295 行（Phase 0-7 编号）；`docs/STARTUP_PROMPT.md` 242 行；两者内容不同。

**可选方案**：以根目录版本为准；以 `docs/` 版本为准；合并为一份新文件并删除另一份。

**影响**：流程与阶段编号的一致性。

---

## Q10. 产物与原始数据的长期存储位置

**已确认事实**

- `/home` 191 GB，当前可用 8.1 GB（多人共享）；代码 + artifacts 位于 `/home`（约 851 MB）。
- 原始数据与 HF 缓存位于 `/mnt/agr-data`（12 TB NFS，实测可写 370 MB/s）；该 NFS 的备份与
  持久性策略未验证。
- `/tmp` 122 GB，但重启后清空（venv 位于 `/tmp`）。

**可选方案**：维持现状（raw/cache 在 NFS，产物在 home）；把产物也迁到 NFS；
申请其他持久卷。

**影响**：磁盘配额、重启后的可恢复性、多人共享时的空间竞争。

**需要负责人提供的输入**：NFS 是否可长期使用、是否有备份；是否可申请独立存储。

---

## Q11. MUSA 加速是否用于 Phase 1 编码

**已确认事实**

- CPU 全量编码：112,590 条约 17 分钟（48 线程，长度分桶后约 200–300 条/s）。
- MUSA 实测：E5-small-v2 在单卡上约 1,700–3,000 条/s；E5-base 未测。
- 当前编码器只支持 `cpu` / `cuda`；容器内需锁定 `numpy==1.25.0` + `faiss-cpu==1.9.0` 以避免 ABI 冲突。
- Phase 1 产物（索引/向量）已用 CPU 生成。

**可选方案**：保持 CPU；接入 MUSA 编码；仅在效率实验中用 MUSA。

**影响**：编码耗时、可复现性（GPU 浮点差异）、ABI 依赖、是否需要重建现有索引。

**需要负责人提供的输入**：Phase 1 是否要求使用 GPU；是否接受 MUSA 环境锁定。

---

## Q12. “第一阶段”的验收边界

**已确认事实**

- AGENTS.md 中“第一阶段”指整个不训练阶段（含完整系统 + TIGER 阶段 B），
  而根 `STARTUP_PROMPT.md` §6 用 Phase 0-7 编号。
- 本轮已交付 Phase 0 + Phase 1；Phase 2–7 尚未开始。

**可选方案**

| 方案 | 本轮验收范围 |
|---|---|
| A | 以 Phase 0 + Phase 1 为本阶段验收 |
| B | 继续完成 Phase 2 + Phase 3（融合、硬约束、Reranker）后再验收 |
| C | 继续到 Phase 6（完整系统）后验收 |

**影响**：本轮验收内容、后续排期与 120 人时预算分配。

**需要负责人提供的输入**：阶段边界的明确定义。

---

## 附：已实施且可逆的默认值（供核对，非决策请求）

| 项 | 当前值 | 修改方式 |
|---|---|---|
| 硬约束状态 | 三态 + 无 check 表示“未配置”（ADR-0003） | 改 `schemas/constraints.py`（需改契约） |
| 两级 RRF k | 60 / 60，同分按 `product_id`（ADR-0002） | `configs/retrieval.yaml` |
| E5 checkpoint | `intfloat/e5-base-v2`（ADR-0001） | `configs/models.yaml` + 重建索引 |
| 数据品类 | `All_Beauty`（ADR-0005） | 重新下载/构建 catalog |
| DeepSeek 模型 | `deepseek-flash`（Planner/Judgment/Generator 共用配置项） | `configs/models.yaml` |
| 编码设备 | CPU | `configs/models.yaml: e5.device`（MUSA 未接入） |
