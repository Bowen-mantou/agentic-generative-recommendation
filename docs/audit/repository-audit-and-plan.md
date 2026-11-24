# 仓库审计与第一阶段实施计划

> 审计基线：`AGENTS.md` V1.1 + `docs/` 六份 V1.1 设计文档
> 审计时间：2026-11-24
> 审计范围：`projects/Agentic-Generative-Recommendation/` 及其直接依赖（S4000 服务器环境）
> 证据附件：[design-contracts.md](design-contracts.md)（字段级契约与冲突行号）、
> [s4000-environment.md](s4000-environment.md)（服务器环境实测）、
> [phase1-report.md](phase1-report.md)（Phase 0/1 执行与验证记录）

---

## 0. 执行摘要

1. **审计前仓库只有文档，没有任何工程实现**：无 `pyproject.toml`/requirements/lock，
   无 `src/`、`configs/`、`tests/`、`data/`、`artifacts/`，无数据、模型权重与索引。
   8 份文档（AGENTS、两份 STARTUP_PROMPT、README 替换说明、HKU/S4000 指南、六份
   V1.1 设计文档）是唯一资产。
2. **项目不在任何 Git 版本控制内**：项目目录位于 `C:/` 根仓库（分支 `verl-migration`）
   之下且为未跟踪目录；该根仓库存在大量与本项目无关的未提交修改/删除。本轮
   **未执行任何 git 写操作**（未 add/commit/init/restore）。
3. **文档架构一致，但存在 5 类阻断/字段级缺口**（见 §4）：`constraint_status` 多出一态
   `not_required`、RRF 的 `k` 无数值、Controller 伪代码引用 State 中不存在的字段、
   Reranker 加载方式表述不一致、TIGER 无实现/无权重。
4. **可行性结论**：E5 + FAISS、Amazon 数据、DeepSeek API 已实测可用；TIGER、
   Qwen3-Reranker-0.6B、Mem0 仍需 Phase 3/4/6 前置验证。
5. **本轮已交付 Phase 0 + Phase 1**（按 STARTUP_PROMPT §6 的阶段编号）：
   工程骨架、跨模块 Pydantic 契约、配置/日志/trace、单元测试；真实
   All_Beauty 商品目录、E5 编码、FAISS 索引、多 Query 内部 RRF 检索，并在
   S4000 上完成部署与真实验证。`Phase 2-7` 尚未开始。

---

## 1. 仓库盘点

### 1.1 审计前目录树（原始状态）

```text
projects/Agentic-Generative-Recommendation/
├── AGENTS.md                              12,872 B / 282 行   V1.1
├── STARTUP_PROMPT.md                      11,027 B / 295 行   （本轮任务提示词）
├── README_替换说明.md                      1,356 B /  28 行
├── HKU-GPU与S4000使用指南.md               7,485 B / 227 行   （含明文口令，禁止外传）
└── docs/
    ├── 项目总体方案.md                    12,636 B / 263 行   V1.1
    ├── 模块一：User Memory 用户记忆系统.md  20,808 B / 762 行   V1.1
    ├── 模块二：Recommendation Planner 推荐规划器.md 25,864 B / 889 行  V1.1
    ├── 模块三：Candidate Retrieval 商品候选召回系统.md 27,753 B / 1,055 行 V1.1
    ├── Reranker 模型选型与设计.md          11,119 B / 303 行   V1.1
    ├── 最终阶段：Judgment、Adaptive Controller、
    │   Recommendation Generator 与 Memory Update.md 27,040 B / 906 行 V1.1
    └── STARTUP_PROMPT.md                  10,906 B / 242 行   V1.1（与根目录版本不同）
```

**不存在**：`pyproject.toml`、`requirements*.txt`、lock 文件、`.env.example`、
`.gitignore`、`src/`、`configs/`、`tests/`、`scripts/`、`notebooks/`、`data/`、
`artifacts/`、`runs/`、任何 Python 文件、任何模型/索引文件。

### 1.2 Python 依赖与配置

| 检查项 | 审计结果 |
|---|---|
| `pyproject.toml` / `requirements.txt` / lock | 全部不存在，依赖版本完全未固定 |
| 配置目录 `configs/` | 不存在 |
| 密钥管理（`.env.example`） | 不存在；DeepSeek key 仅由任务方口头提供 |
| 测试结构 | 不存在；无 pytest 配置 |
| 静态检查 | 无 ruff/mypy/CI 配置 |
| 日志/trace | 无任何实现 |

### 1.3 Git 状态（只读检查）

| 项目 | 结果 |
|---|---|
| 仓库根 | `C:/`（`git rev-parse --show-toplevel`），分支 `verl-migration` |
| 本项目目录 | 未跟踪：`?? new/intern/plan/projects/Agentic-Generative-Recommendation/` |
| 根仓库已有改动 | `01-search-r1/*` 修改、`04-vllm-sched/*` 大量删除等，**与本项目无关** |
| 本轮 git 操作 | **无**（未 add/commit/stash/init，未触碰无关文件） |
| 风险 | 项目自身无版本历史；一旦误删无法回滚（决策 Q8） |

### 1.4 数据、模型与索引

| 资源 | 审计结果 |
|---|---|
| Amazon Reviews 2023 原始数据 | 仓库内无；S4000 的 `/home/agr` 下也未发现缓存（env-recon §6） |
| 商品目录 / `parent_asin → product_id` 映射 | 不存在 |
| E5 / Qwen3-Reranker / TIGER 权重 | 不存在 |
| FAISS 索引 / embeddings | 不存在 |
| InteractionHistory / SQLite | 不存在 |

### 1.5 审计后目录树（Phase 0 + Phase 1 完成后）

```text
Agentic-Generative-Recommendation/
├── AGENTS.md / README.md / STARTUP_PROMPT.md / README_替换说明.md
├── pyproject.toml  .env.example  .gitignore  .gitattributes
├── configs/
│   ├── base.yaml  models.yaml  retrieval.yaml  controller.yaml  memory.yaml
│   ├── local.example.yaml     （服务器路径示例；local.yaml 为 gitignored 本地覆盖）
├── docs/
│   ├── （六份 V1.1 设计文档 + 两份 STARTUP_PROMPT）
│   ├── audit/
│   │   ├── design-contracts.md              （字段级契约提取，1006 行）
│   │   ├── s4000-environment.md             （服务器环境实测，558 行）
│   │   ├── repository-audit-and-plan.md     （本文件）
│   │   └── phase1-report.md                 （执行与验证记录）
│   ├── decisions/                           （ADR：见 §4.6）
│   └── architecture/
├── src/recommender/
│   ├── config.py  cli.py
│   ├── schemas/          （common/product/plan/retrieval/constraints/fusion/
│   │                      rerank/judgment/controller/generation/memory/
│   │                      evaluation/artifacts，13 个模块）
│   ├── catalog/          （amazon.py / text.py / build.py）
│   ├── retrieval/e5/     （encoder.py / index.py / service.py / artifacts.py）
│   ├── fusion/rrf.py
│   ├── observability/    （context.py / logging.py / trace.py）
│   ├── utils/            （errors.py / io.py）
│   └── （planner/judgment/reranking/controller/generation/memory/
│          interaction_history/evaluation/api 为 Phase 3-7 预留空包）
├── scripts/  （sync_to_server.py / bootstrap_s4000.sh / env_s4000.sh）
├── tests/unit/（109 个测试）  tests/fixtures/plans/initial_beauty.json
│   tests/integration/
├── artifacts/（faiss 索引 bundle：index.faiss / product_ids.jsonl /
│              embeddings.npy / manifest.json；catalog/ 目录结构同上）
└── runs/（JSONL trace，按 request_id 分目录）
```

---

## 2. 文档清单与版本

| # | 文档 | 版本 | 规模 | 职责 | 审计发现 |
|---|---|---|---|---|---|
| 0 | `AGENTS.md` | V1.1 | 282 行 | 最高工程约束、技术基线、数据契约 | 与六份文档一致 |
| 1 | `docs/项目总体方案.md` | V1.1 | 263 行 | 系统架构、选型总表、评测维度 | 与 AGENTS 一致 |
| 2 | `docs/模块一：User Memory…md` | V1.1 | 762 行 | Mem0 记忆、InteractionEvent、Persona/模拟对话 | `not_required` 之外无冲突 |
| 3 | `docs/模块二：Recommendation Planner…md` | V1.1 | 889 行 | 唯一 `RecommendationPlan`、Initial/Replanner | Plan Schema 完整 |
| 4 | `docs/模块三：Candidate Retrieval…md` | V1.1 | 1,055 行 | TIGER、E5+FAISS、两级 RRF、硬约束、候选池 | 候选池字段与来源命名有缺口 |
| 5 | `docs/Reranker 模型选型与设计.md` | V1.1 | 303 行 | Qwen3-Reranker-0.6B、Query/Document 模板 | §3.1 与 §6 表述需实测澄清 |
| 6 | `docs/最终阶段…md` | V1.1 | 906 行 | Judgment、Controller、Generator、Memory Update | Controller/State 不一致 |
| 7 | `docs/STARTUP_PROMPT.md` | V1.1 | 242 行 | 项目内启动说明 | 与根目录版本内容不同（Q9） |
| 8 | 根 `STARTUP_PROMPT.md` | — | 295 行 | 本轮任务提示词（Phase 0-7） | 与 `docs/` 版本需合并 |

**版本一致性结论**：六份设计文档 + AGENTS 均为 V1.1，未发现旧版残留正文；旧 JEV /
Wikipedia / Reader / 本地 Qwen3-8B 只以**否定式**出现（AGENTS L11/L82/L264、
总体方案 L141、STARTUP_PROMPT L78）。项目目录内**没有**旧项目代码或数据。

> ⚠️ 两份 `STARTUP_PROMPT.md` 内容不同：根目录 295 行版本给出 Phase 0-7 的
> 四阶段编号与审计要求；`docs/` 242 行版本是项目内启动说明。两者不冲突但重复，
> 两者内容不同，已作为待决事项 Q9 登记。

---

## 3. 现有实现映射（审计时状态）

> 原则：没有找到代码就写“未实现”，不把文档描述当作已完成实现。

| 模块 | 文档要求 | 仓库现状（审计时） | 缺失内容 | 风险 |
|---|---|---|---|---|
| 配置 / 依赖 | 固定 Python 与依赖版本、YAML 配置、`.env.example` | **无实现** | pyproject、lock、配置加载、密钥模板 | 高：无版本固定，实验不可复现 |
| Schema（跨模块） | Pydantic 为权威契约（Plan/Candidate/Judgment/State/Memory） | **无实现** | 全部 Schema | 高：各模块会各自发明字段 |
| Planner / Replanner | DeepSeek API，统一 `RecommendationPlan`，不得架空字段 | **无实现** | Prompt、Client、校验、Replanner 输入 | 中 |
| User Memory | Mem0 OSS + Qdrant + SQLite，写入规则 | **无实现** | Adapter、规则引擎、Qdrant 部署 | 中 |
| Interaction History | SQLite `InteractionEvent`，按时间排序，`is_synthetic` 隔离 | **无实现** | 表结构、ETL、查询 | 中 |
| TIGER | Semantic ID、RQ-VAE、自回归生成、解码 | **无实现、无权重、无兼容实现** | 全部 | **极高**：Phase 4 唯一硬阻塞 |
| E5 + FAISS | 商品文本 → E5 → FAISS；多 Query 内部 RRF | **无实现** | 编码器、索引、检索、映射、版本 | 中（本轮已补齐） |
| Candidate Fusion | ID 统一、去重、两级 RRF、约束调用 | **无实现** | 全部 | 中 |
| Hard Constraints | 唯一 Python Evaluator，pass/fail/unknown | **无实现** | 规则引擎、状态聚合 | 中：文档有第 4 态冲突 |
| Reranker | Qwen3-Reranker-0.6B（冻结）+ bge 对照 | **无实现** | 模型适配、模板、批量、计时 | 中高：MUSA 未验证 |
| Judgment | DeepSeek 逐商品结构化判断 | **无实现** | Prompt、批量并发、Schema 校验 | 中 |
| Controller | LangGraph + Python，REPLAN/STOP，多维预算 | **无实现** | State、条件边、预算、循环保护 | 中 |
| Generator | DeepSeek，只表达候选池内商品 | **无实现** | Prompt、证据约束、输出 Schema | 中 |
| Trace / 观测 | JSONL + SQLite，request_id/trace_id | **无实现** | 日志、trace、脱敏 | 中 |
| 评测 | 时间切分、Recall/NDCG、消融 | **无实现** | 样本构造、指标脚本、报告 | 中高 |
| 测试 | unit/integration/E2E，Mock 隔离 | **无实现** | 全部 | 高 |

**结论**：审计时项目完成度为“文档 100% / 实现 0%”。本轮补齐了 Phase 0 的全部
基础工程与 Phase 1 的 E5 召回链路（见 §6.3/§6.4 与 phase1-report）。

---

## 4. 关键架构冲突与缺失契约

> 行号级证据与完整清单见 [design-contracts.md](design-contracts.md) §4；这里只列
> 会影响实现的条目与处理决策。

### 4.1 阻断级（会改变架构或阻塞阶段）

| ID | 冲突/缺口 | 证据 | 本轮处理 | 需要确认 |
|---|---|---|---|---|
| B1 | TIGER 无兼容实现、无权重；RQ-VAE 层数/码本、生成器 backbone 全部未定义 | 模块三 §3.2-3.5；AGENTS §8.1 | 配置占位（`tiger.generator.backbone_model_id: null`），**未用 Mock 冒充** | 是（Q1） |
| B2 | Reranker 表述：§3.1 说“yes/no token 分数”，§6 用 SentenceTransformers `CrossEncoder(prompts=...)`；MUSA 上未验证 | Reranker L56-L59 vs L196-L228 | 契约先定（`RerankResult.score_interpretation` 明确“非购买概率”），Phase 3 前置冒烟 | 是（Q2） |
| B3 | `constraint_status` 三种形状：候选池 `str`（模块三 L772）、Judgment 输入 `dict`（最终阶段 L170）、Python 示例第 4 态 `not_required`（L211） | 见 design-contracts C1 | 统一三态枚举；`not_required` = 该约束**不存在 check**；Judgment 消费 `ConstraintEvaluation` | 是（Q3） |
| B4 | Controller 伪代码引用 `state["hard_constraints_satisfied"]`、`state["quality_sufficient"]`，但 State 定义里没有这两个键 | 最终阶段 L400-L404 vs L469-L482 | 放入 `QualityMetrics`（`hard_constraints_satisfied` / `quality_satisfied`），State 持 `quality_metrics` | 否（已按最小改动落地） |
| B5 | Mem0/Qdrant 部署方式、SDK 版本、Mem0 embedder、SQLite DDL 均未定义 | 模块一 §6、最终阶段 §6 | 仅配置占位 + 契约；Phase 6 处理 | 是（Q7） |

### 4.2 字段级缺口（已在本轮契约中裁决）

| ID | 缺口 | 裁决 | 位置 |
|---|---|---|---|
| C1 | RRF 的 `k` 全文无数值；两级是否同 k 未定 | 两级同 k=60，配置化（`retrieval.e5.rrf_k` / `retrieval.fusion.rrf_k`） | ADR-0002 |
| C2 | `source_ranks` 示例键为 `"tiger"` / `"e5"`，但召回源枚举写成 `E5_FAISS` | 统一用 `RecallPath` 的值：`"tiger"` / `"e5_faiss"`；契约校验键必须与来源一致 | `schemas/fusion.py` |
| C3 | `unverified_candidates` 元素结构未定义 | 复用 `FusedCandidate`（含 `constraint_status=unknown`），与 eligible/rejected 三分区 | `schemas/fusion.py` |
| C4 | 禁止顶层 `retrieval_strategy` / `tiger` / `e5_faiss` | `RecommendationPlan` `extra=forbid`，有单测覆盖 | `schemas/plan.py` |
| C5 | E5 Query 数量/长度限制无数值 | 协议上限 10 条 / 500 字符；运行上限 `max_queries=5`、`query_max_chars=300` | `config.py` |
| C6 | Budget 数值只是示例 | 采用 2 轮 replan / 6 次检索 / 60 次 API / 50-100 候选，标注未校准 | `configs/controller.yaml` |
| C7 | E5 checkpoint、维度、前缀未定 | 默认 `intfloat/e5-base-v2`（768 维，`query: `/`passage: `），配置化 | ADR-0001 |
| C8 | FAISS 索引类型未定 | `IndexFlatIP` + L2 归一化（余弦），先精确基线 | `configs/retrieval.yaml` |
| C9 | Judgment 输入的约束状态形状 | 契约先定义 `JudgmentCandidate.constraint_status`（整体三态）+ 上游 checks 可追溯 | 待 Q3 确认 |
| C10 | `InteractionEvent.event_type` 词汇表分散 | 统一枚举：review/rating/click/favorite/purchase/explicit_\*；review/rating 才允许 `rating` | `schemas/memory.py` |
| C11 | 商品文本模板（embedding/rerank）无版本 | 两个独立模板版本：`product_text.embedding.v1` / `product_text.rerank.v1` | `catalog/text.py` |
| C12 | Generator 的“理由必须有依据”无量化门槛 | 契约强制 `reasons` 非空；Phase 6 用证据引用校验 | `schemas/generation.py` |
| C13 | Memory confidence 不能直接用 LLM 自报 | 契约记录 `confidence` 但不由 LLM 决定；规则引擎在 Phase 6 | `schemas/memory.py` |
| C14 | 评测样本构造（Query 生成）未定义 | Phase 7 用截点前信息构造；本轮只定义 `RetrievalEvalSample` 并强制防泄漏 | `schemas/evaluation.py` |

### 4.3 未定义接口（需要在后续阶段先定义 Schema 再写代码）

`load_user_context` 的 Mem0 检索参数；Replanner 输入 Schema；Judgment 批量协议；
Controller 阈值来源；Memory 写入规则引擎；Feedback Workflow；Trace 字段全集；
TIGER beam-search 输出；澄清挂起/恢复持久化；SQLite DDL；评测样本生成规则。
完整列表见 design-contracts §4.4（G1-G14）。

### 4.4 无数值参数（已给出临时值并标注未校准）

| 参数 | 文档状态 | 当前值 | 校准阶段 |
|---|---|---|---|
| RRF k | 未给 | 60 / 60 | Phase 7 |
| E5 top-k per query | 示例 50 | 50 | Phase 7 |
| 候选预算 standard / expanded | 仅档位 | 50/100、100/200 | Phase 7 |
| Replan 次数 / 检索次数 / API 次数 | 示例 2/4/12 | 2/6/60 | Phase 7 |
| required_count / min_valid_candidates | 未给 | 5 / 3 | Phase 7 |
| min_mean_rerank_score / min_diversity | 未给 | `null`（未启用） | Phase 3/7 |
| DeepSeek temperature/max_tokens/timeout | 未给 | 0.2 / 4096 / 60s | Phase 5 |
| Reranker max_length / batch | 未给 | 2048 / 8 | Phase 3 冒烟后定 |

### 4.5 旧项目残留检查

项目目录内**未发现** JEV / Wikipedia / 多跳 / Reader / 本地 Qwen3-8B 的实现或依赖；
仅存在于 AGENTS/STARTUP 的否定式约束与 design-contracts 的排除清单中。
目录外（`../01-search-r1` 等）属于其他项目，本轮未读取、未修改。

### 4.6 本轮新增 ADR

| ADR | 决策 |
|---|---|
| `docs/decisions/0001-e5-checkpoint.md` | 默认 `intfloat/e5-base-v2`，配置化；维度/前缀写进 manifest |
| `docs/decisions/0002-rrf-parameters.md` | 两级 RRF 均 k=60；同分按 `product_id` 排序保证可复现 |
| `docs/decisions/0003-constraint-status.md` | 三态枚举；`not_required` 用“无 check”表达；Judgment 只消费上游结果 |
| `docs/decisions/0004-server-storage-layout.md` | `/home` 放代码与产物、`/tmp` 放 venv、12 TB NFS 放原始数据与缓存 |
| `docs/decisions/0005-data-scope.md` | 第一阶段用 `All_Beauty` 单品类建真实链路，后续可扩展多品类 |

---

## 5. 可行性检查

| 检查项 | 结论 | 证据 / 状态 |
|---|---|---|
| TIGER 是否有可复用且兼容的实现和权重 | **未找到** | 项目目录无代码；未检索外部仓库（不纳入本轮）；无权重 → Phase 4 需自建轻量方案或另行决策（Q1） |
| 当前数据能否构建 Semantic ID 与用户历史序列 | **可以** | `All_Beauty` meta 212,990,142 B / reviews 326,611,506 B；review 含 `user_id`/`parent_asin`/`timestamp`/`rating`；已建 catalog（真实运行） |
| E5 + 商品元数据 + FAISS 能否形成完整链路 | **已打通** | Phase 1 真实运行：编码 → 索引 → Query 检索 → 多 Query RRF；见 phase1-report |
| Qwen3-Reranker-0.6B 推理接口与硬件开销 | **未验证** | ST CrossEncoder 是否原生支持 Qwen3-Reranker 需冒烟；MUSA 未验证；CPU 延迟未知（Q2） |
| DeepSeek API 访问配置 | **已验证** | `GET /v1/models` 200；可用模型 `deepseek-flash`、`deepseek-v4-pro`；chat 200（env-recon §2） |
| Mem0 部署方式与 SDK | **未验证** | dp 容器无 `mem0ai`/`qdrant-client`，Qdrant 未部署；Phase 6 前置 |
| S4000 存储/网络/依赖 | **已验证** | home 可用 8.9 GB（清理 uv 缓存后）；NFS 12 TB 可写 370 MB/s；hf-mirror 3.1–3.6 MB/s；tuna 119 MB/s；无 python3-venv，用 uv（env-recon §1-3） |
| GPU/MUSA | **部分验证** | E5-small 在 MUSA 单卡跑通 ~3000 条/s（env-recon §5）；Phase 1 仍用 CPU，MUSA 接入未做 |

**未验证事项（明确标注）**：TIGER 权重来源与训练成本；Qwen3-Reranker 在 MUSA/CPU 的
吞吐与精度；Mem0/Qdrant 稳定性；All_Beauty 全量完整性（下载未完成前）；NFS 备份策略；
Controller 阈值的验证集来源。

---

## 6. 第一阶段实施计划

### 6.1 范围与原则

- **“第一阶段”的解释**：按根 `STARTUP_PROMPT.md` §6 的 Phase 0-7 编号组织；本轮交付
  **Phase 0 + Phase 1**（工程骨架 + 商品目录/E5 召回基线）。Phase 2-7 为目标计划，
  尚未执行，本文不把它们描述为已完成。
- 总预算约 120 人时；优先级：真实端到端最小链路 > 高级实验。
- 所有 Mock 显式标记且不得进入正式实验；未运行的测试不得声称通过。

### 6.2 阶段总表

| 阶段 | 内容 | 人时 | 依赖 | 交付物 | 验收标准 | 状态 |
|---|---|---|---|---|---|---|
| Phase 0 | 工程骨架、配置、Schema、日志/trace、pytest | 12 | — | `pyproject.toml`、`configs/*`、`schemas/*`、`observability/*`、`tests/` | 可安装、可导入、单元测试通过、`agr config validate` 正常 | ✅ 完成 |
| Phase 1 | 商品目录、E5 编码、FAISS、多 Query RRF、版本化 | 18 | Phase 0 | `catalog/*`、`retrieval/e5/*`、`fusion/rrf.py`、真实索引 bundle | 给定 Query 返回真实 product_id + 相似度排名；索引带版本与映射 | ✅ 完成 |
| Phase 2 | 候选协议、去重、两级 RRF、唯一硬约束 Evaluator、候选池 | 12 | Phase 1 | `fusion/`、`constraints/` | 两路可合并；pass/fail/unknown；缺失字段不当作满足 | 待开始 |
| Phase 3 | Qwen3-Reranker-0.6B 接入、批量精排、计时 | 12 | Phase 2 | `reranking/` | 真实模型重排；分数不解释为购买概率 | 待开始 |
| Phase 4 | TIGER 真实生成式召回（RQ-VAE + 自回归 + 解码） | 24 | Phase 1；数据 ETL | `retrieval/tiger/` | 真实历史序列 → Semantic ID → 真实商品 | **阻塞**（Q1） |
| Phase 5 | Planner/Replanner、Judgment、LangGraph Controller、预算 | 18 | Phase 2-3 | `planner/`、`judgment/`、`controller/` | 真实触发 REPLAN/STOP；预算与异常路径覆盖 | 待开始 |
| Phase 6 | Generator、Mem0 Adapter、Feedback Workflow | 14 | Phase 5 | `generation/`、`memory/`、`interaction_history/` | 推荐有依据；有效反馈更新记忆；模拟隔离 | 待开始 |
| Phase 7 | 端到端评测：时间切分、消融、延迟/成本 | 12 | Phase 1-6 | `evaluation/`、实验报告 | 指标可从脚本与版本化数据复现 | 待开始 |
| **合计** | | **122** | | | | 已完成 30 |

> 估算基于 6 人 × 20h 预算；Phase 4 若需外部 TIGER 复现，工作量可能翻倍（Q1）。

### 6.3 Phase 0 交付与验收（已完成）

**交付**：`pyproject.toml`（依赖分组 + pytest/ruff/mypy 配置）、`.env.example`、
`.gitignore`、`.gitattributes`、5 份 YAML + `local.example.yaml`、配置加载器
（YAML 深合并 + `AGR__` 环境覆盖 + `local.yaml` 覆盖 + 类型校验 + 路径解析）、
13 个 Schema 模块、JSON 日志与 request/trace 上下文、JSONL trace（脱敏）、
`agr` CLI（version/config/catalog/index/search）、109 个单元测试。

**验收命令与结果**：见 [phase1-report.md](phase1-report.md) §2（可复现）。
`python -m pytest -q` 全绿；`ruff check` / `mypy` 无告警；`pip install -e .` 成功；
`agr config validate` 输出合法配置与路径。

### 6.4 Phase 1 交付与验收（已完成）

**交付**：Amazon metadata 下载器（可重试、断点安全、SHA-256）、`ProductRecord`
目录构建（`parent_asin → product_id`、去重、跳过统计、manifest 版本化）、双文本模板
（embedding/rerank 版本化）、E5 编码器（懒加载、前缀、归一化、维度校验）、FAISS
`IndexFlatIP` 构建/加载/哈希校验、E5 检索服务（多 Query 内部 RRF、可解释的代表
Query、`E5PathOutput` 契约）、CLI 全链路命令。

**验收标准对照**：
1. 给定 Query 能检索出真实商品 ID 与相似度排名 → phase1-report 记录真实
   `product_id`、相似度与排名（非 Mock）。
2. 多 Query 内部 RRF → 单测 + 真实运行各记录一次。
3. 索引版本与商品 ID 映射保存 → `manifest.json`（模型/维度/哈希/条数）+
   `product_ids.jsonl`；`agr index info` 可校验。

**执行中发现并修复的数据映射问题**：初版 `build_product_record` 只读取始终为空的
`categories` 列表，导致 `main_category` 全为 `null`；复核原始记录后改为优先使用顶层
`main_category` 字段（100% 覆盖，`All Beauty` / `Premium Beauty`），补两个单测并重建
catalog（`all_beauty_v2`）。同时实测字段覆盖：title 100%、brand 89.9%、details 96%，
但 **price 仅 15.7%**、description 17.0%、features 15.4% —— Phase 2 的价格硬约束对多数
商品只能返回 `unknown`，这是数据事实而非实现缺陷。

### 6.5 后续阶段依赖顺序

```text
Phase 1 (done) ──► Phase 2 (fusion + constraints) ──► Phase 3 (reranker) ──► Phase 5 (planner/judgment/controller)
        │                                                                              │
        └──► Phase 4 (TIGER, 需 Q1 决策) ─────────────────────────────────────────────┤
                                                                                       ▼
                                                                            Phase 6 (generator/memory)
                                                                                       │
                                                                                       ▼
                                                                            Phase 7 (evaluation)
```

Phase 2 不依赖 TIGER 也可先做（TIGER 结果用契约校验的占位路径测试，但正式实验必须
等 Phase 4 真实结果）。

---

## 7. 待负责人决策事项

12 项待定事项已集中整理到 [open-questions.md](../decisions/open-questions.md)（只列事实、
可选方案与影响，不含推荐意见）。本文不再重复其内容，摘要如下：

| ID | 待决问题 | 影响阶段 |
|---|---|---|
| Q1 | TIGER 实现路线（自建 / 外部 / 暂缓） | Phase 4、双路召回 |
| Q2 | Reranker 加载方式与运行设备 | Phase 3 |
| Q3 | Judgment 输入的 `constraint_status` 形状 | Phase 5 |
| Q4 | E5 checkpoint 选择 | Phase 1（需重建索引） |
| Q5 | 数据品类范围 | Phase 1/4/7 |
| Q6 | Controller 阈值与预算的校准数据 | Phase 5/7 |
| Q7 | Mem0 / Qdrant 部署形态 | Phase 6 |
| Q8 | 项目 Git 仓库边界 | 全流程 |
| Q9 | 两份 `STARTUP_PROMPT.md` 的取舍 | 全流程 |
| Q10 | 产物与原始数据的长期存储位置 | 运维 |
| Q11 | MUSA 加速是否用于 Phase 1 编码 | Phase 1/7 |
| Q12 | “第一阶段”的验收边界 | 全部 |

---

## 8. 本轮验收物与未完成项

**验收物**：本文件、[design-contracts.md](design-contracts.md)、
[s4000-environment.md](s4000-environment.md)、[phase1-report.md](phase1-report.md)、
`src/` 全部代码、`tests/`、`configs/`、`scripts/`、服务器上的真实 catalog 与索引
bundle、`docs/decisions/` 五份 ADR 与 [open-questions.md](../decisions/open-questions.md)。

**明确未做（不得误认为完成）**：TIGER、Reranker、Planner、Judgment、Controller、
Generator、Mem0、LangGraph 工作流、端到端评测、真实 API 调用链路。
