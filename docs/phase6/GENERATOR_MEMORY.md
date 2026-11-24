# Phase 6 G5：DeepSeek Generator + Mem0 Memory Adapter 实现与真实验证

| 项 | 值 |
|---|---|
| 文档 | `docs/phase6/GENERATOR_MEMORY.md` |
| 对应任务 | 共享任务板 `task-10`（24 小时方案 G5） |
| 执行时间 | 2026-11-25 00:53 – 01:10（Asia/Shanghai） |
| 执行人 | `gen-memory`（Agent） |
| 代码基线 | 本地 `main` @ `a8717b7` + 本次新增（提交见 §9）；Lead 并行修改的 `src/recommender/e2e/`、`cli.py`、`pyproject.toml` 不在本任务写范围 |
| 服务器 | `10.123.0.45`（hostname 见 D9），venv `/tmp/agr/venv`，`HF_HOME=/mnt/his_test/group11/agr/hf`，Qdrant 容器 `agr-qdrant` @ `127.0.0.1:6333` |
| 结论 | **Mem0 Adapter 真实 add/read/双 user_id 隔离/按 ID 删除/清理全部通过（11/11 checks）；真实 DeepSeek Generator 英文 query 与中文 query 两次 smoke 均成功（无 fallback），商品 ID 均来自候选、标题/价格来自真实 catalog、每条 reason 的证据路径均真实存在。34 个新增单测通过，全量 315 测试通过。** |

> 本文件只记录**真实执行**过的命令、输出与结果。未执行的内容一律标为「未验证」，不把 Mock / 图级测试 / 离线结果当作端到端验收。
> 证据报告原始文件位于服务器 `/tmp/g5_mem0_smoke.json`、`/tmp/g5_generator_smoke.json`、`/tmp/g5_generator_smoke_zh.json`（临时证据，不进入 Git；关键数字已完整摘录在本文件）。

---

## 0. 结论摘要

| 项 | 结果 | 证据位置 |
|---|---|---|
| Memory Adapter 契约 | `MemoryHit` / `MemoryAdapter` Protocol / `create_memory_adapter` / 显式 `UnavailableMemoryAdapter` 按 task-10 实现；导入不加载 Mem0、不加载模型、不请求网络 | §1、§2 |
| 真实 Mem0 写入 | `phase6_smoke_u1` 2 条、`phase6_smoke_u2` 1 条，`infer=False`，`is_synthetic=true`，metadata 原样落库 | §5.2 |
| 真实 Mem0 读取 | 两个用户的 `search(filters={"user_id": ...})` 只返回本用户记忆；交叉 query 不泄漏他用户；E5 中文语义排序正确（u1 保湿 query 0.9175 > 洁面 0.8570） | §5.2 |
| 真实 user_id 隔离 | Mem0 原始 `get_all(filters=...)` 计数 2/1，payload user_id 与请求一致；Adapter 读取层再做一次 payload 校验 | §5.2 |
| 真实删除/清理 | 按 `memory_id` 删除生效；`delete_all` 后两个测试用户查询均为 `[]`，测试 collection 点数 3 → 0 | §5.2 |
| Mem0 版本/维度 | `mem0ai 2.2.1`、`qdrant-client 1.20.0`、collection 768 维 Cosine + `bm25` sparse 槽（dense-only） | §5.2、§6 |
| 真实 DeepSeek Generator | 英文 query：`deepseek-flash`，2066.7 ms，prompt/completion tokens 1862/293，attempts=1，无 fallback；中文 query：1743.1 ms，1862/283，同样失败 0 | §5.3 |
| 生成内容真实性 | 两次 smoke 的 product_id 均为候选子集；title/price 与真实 catalog 完全一致；每条 reason 的 `evidence_refs` 均能在候选 evidence 中解析；英文/中文理由语言与请求一致 | §5.3 |
| 语言问题（Lead 发现） | 修复前英文 query 返回中文理由；prompt 增加「与 current_request 同语言」规则 + 单测 + 英文/中文真实 smoke 复验通过 | §3.5、§5.4 |
| 测试 | 新增 34 个单测（Memory 17 + Generator 17），本地与服务器 venv 均通过；全量 315 通过；`ruff` 通过；本范围 `mypy` 通过 | §4 |
| 未验证 | 真实 E2E 由 Lead 执行（本任务未重复）；`infer=True` LLM 抽取；BM25/spaCy 混合检索；threshold 校准；Memory Update 准入规则；并发写 | §8 |

---

## 1. 集成接口（与 Lead 的 E2E 严格保持一致）

### 1.1 Memory

`src/recommender/memory/adapter.py`

```python
@dataclass(frozen=True)
class MemoryHit:
    memory_id: str
    text: str
    score: float | None
    metadata: dict[str, Any]

class MemoryAdapter(Protocol):
    def read(self, user_id: str, query: str, limit: int = 5) -> list[MemoryHit]: ...
    def add(self, *, user_id: str, text: str,
            metadata: dict[str, Any] | None = None,
            is_synthetic: bool = False) -> MemoryHit: ...
    def health(self) -> dict[str, Any]: ...

def create_memory_adapter(config: MemoryConfig) -> MemoryAdapter: ...
```

- `Mem0MemoryAdapter`（`mem0_adapter.py`）为真实实现；`UnavailableMemoryAdapter` 为显式降级实现：
  `health().available=False`、`read()` 返回 `[]`、`add()` 抛 `MemoryUnavailableError`（绝不静默假装写入成功）。
- `create_memory_adapter` 在以下情况返回 `UnavailableMemoryAdapter`，原因写入 `health()["reason"]`：
  `memory.enabled=false`、vector store provider 不是 `qdrant`、`mem0ai`/`qdrant-client` 未安装。
- 额外提供 `delete(memory_id)` / `delete_all(user_id)`，仅用于 smoke 与清理任务，不改变 E2E 必需接口。
- E2E 调用方式（Lead `src/recommender/e2e/service.py` 实际代码）：
  `create_memory_adapter(config.memory.model_copy(update={"enabled": True}))` → `health()` → `read(user_id, query, limit=5)`。

### 1.2 Generator

`src/recommender/generation/deepseek_generator.py`

```python
@dataclass(frozen=True)
class GeneratorCandidate:
    product_id: str
    title: str = ""
    price: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    judgment: dict[str, Any] = field(default_factory=dict)

class RecommendationGenerator:
    def __init__(self, client: DeepSeekClient, config: GenerationConfig) -> None: ...
    def generate(
        self, *, request_id: str, user_request: str,
        candidates: Sequence[GeneratorCandidate],
        stop_reason: StopReason | None,
        quality_satisfied: bool | None,
        budget_exhausted: bool | None,
    ) -> RecommendationOutput: ...
```

- `RecommendationGenerator` **不自行创建** DeepSeek client；client 由调用方（E2E）注入。
- Prompt 版本常量：`recommendation_generator_v1`。
- 对既有 `schemas/generation.py` 仅做向后兼容的最小扩展：
  - `RecommendationItem.evidence_refs: list[str] = []`：与 `reasons` 一一对应的证据路径，Gate 可审计 grounding；
  - `RecommendationOutput.generation_metadata: GenerationMetadata | None = None`：attempts / retries / fallback / token / latency；
  - 两个字段都有默认值，Lead 的 `_FallbackGenerator` 等既有构造无需修改。

---

## 2. Memory Adapter 设计

### 2.1 惰性初始化与显式失败

- 模块导入**不** import mem0、不加载 E5、不连接 Qdrant。
- 第一次 `health()` / `read()` / `add()` 才执行 `_build_memory()`；失败后缓存 `_init_error`，后续调用继续抛 `MemoryUnavailableError` / health 返回 `available=False`，不做无界重试。
- `MEM0_TELEMETRY=False` 在首次 import mem0 前通过 `os.environ.setdefault` 设置（D9 §7.6）。

### 2.2 本地 Embedding 与前缀（D9 §6.3/§7.3）

- 使用项目 `E5Encoder` 包装为 `E5MemoryEmbedder`，模型 `intfloat/multilingual-e5-base`，768 维，revision `d128750597153bb5987e10b1c3493a34e5a4502a`（D9 实测 commit）。
- `memory_action` 映射：`add`/`update` → `passage: `，`search` → `query: `；`embed_batch` 批量编码。
- **不共享商品向量空间**（R8）：记忆 collection 独立（`agr_user_memory`），与商品 `intfloat/e5-base-v2` FAISS 索引完全隔离。

### 2.3 Mem0 配置（实际生效值）

```text
vector_store: qdrant, host=127.0.0.1, port=6333, collection=agr_user_memory,
              embedding_model_dims=768, on_disk=false
embedder:     huggingface 占位（构造后立即替换为 E5MemoryEmbedder，避免二次加载模型）
llm:          deepseek（仅构造所需；本 Adapter 永远传 infer=False，不会调用 LLM 抽取）
history_db:   artifacts/mem0/history.db（相对项目根，Git 忽略；与 InteractionEvent SQLite 无关）
version:      v1.1（mem0 默认 API version）
```

- 占位 embedder 使用不会访问的 `huggingface_base_url`，使 `Memory.from_config` 不加载第二份 HF 模型；构造后执行 `memory.embedding_model = E5MemoryEmbedder(...)`，Mem0 的 add/search/update 都经由 `self.embedding_model`，替换立即生效（D9 §7.3 方案 1）。
- Mem0 自带 SQLite 只是 changelog，不是项目 `InteractionEvent` 存储（AGENTS.md §2.2 / R12）。

### 2.4 user_id 隔离（R11）

- `read` 固定调用 `memory.search(query, filters={"user_id": user_id}, top_k=limit)`；Mem0 2.2.1 已移除 `search(user_id=...)` 参数（D9 §7.2）。
- 返回结果再做一次 payload 级防御：若 `item["user_id"]` 存在且不等于请求用户，丢弃并记 warning，绝不把他人 payload 交给 Planner/Generator。
- `add` 的身份字段 `user_id` 只通过 Mem0 顶层参数传入；用户 metadata 中的 `user_id/agent_id/run_id/data/hash/created_at/updated_at/text_lemmatized` 会被过滤，不能覆盖身份或内部字段。

### 2.5 is_synthetic 与清理

- 每次 `add` 都写入 `metadata["is_synthetic"]`（默认 `False`，测试/模拟反馈必须显式 `True`），并随 `MemoryHit.metadata` 原样返回；本任务 smoke 的 3 条测试记忆全部标记 `is_synthetic=true`。
- 真实 smoke 中按 `memory_id` 删除 1 条、再 `delete_all(user)` 清空两个测试用户；清理后查询均为 `[]`，collection 点数回到 0。

### 2.6 health()

返回结构化信息（不包含 API key）：`available`、`backend="mem0"`、collection、vector_store 实际 host/port/dims、embedder 模型/revision/维度/前缀映射、`user_id_isolation` 说明、`history_db_path`、`versions.{mem0ai,qdrant-client}`。E2E 直接把该 dict 写入 trace。

---

## 3. Generator 设计

### 3.1 只表达、不改写

- 只允许输入候选 `product_id`（候选外 ID 触发重试，持续非法则 fallback）。
- 输出顺序强制为 Controller/Reranker 的候选顺序：模型返回顺序被忽略，绝不擅自改变排序。
- 输出条目的 `title` / `price` **一律取自 catalog 候选对象**，模型返回的同名字段被忽略，防止编造或改写商品事实。
- 最多输出 `config.generation.top_k` 条；允许少于候选数，但不得重复、不得为空（无候选时返回显式空 fallback，见 §3.4）。

### 3.2 证据路径契约（每条 reason 必须引用真实证据）

- 对每个候选，`candidate_evidence_paths()` 从 `title`、`price`、`evidence.*`、`judgment.*` 递归展开**真实存在且非空的叶子路径**（如 `evidence.details.Scent`、`judgment.preference_fit.level`）。
- Prompt 把每个候选的 `allowed_evidence` 列表原样交给模型；模型每条 reason 必须返回 `{text, evidence}`，`evidence` 必须精确命中该候选的 allowed path。
- Python 校验失败 → 一次修复重试；仍失败 → 模板 fallback。校验通过的 `evidence` 写入 `RecommendationItem.evidence_refs`（与 reasons 同序），E2E Gate 可独立复核。
- 这样把「理由有证据」从 prose 断言变成可机器校验的引用，避免把 `unknown`/缺失属性写成满足。

### 3.3 有界重试与 fallback

| 失败类型 | 重试预算 | 最终行为 |
|---|---|---|
| `DeepSeekAPIError`（含 timeout / JSON / rate limit 等） | 最多 1 次 API 重试 | 仍失败 → `RecommendationOutput(model="fallback_template")` |
| 非法 JSON / 候选外 ID / 空 reason / 证据路径不存在 / schema 失败 | 最多 1 次格式修复（回显非法输出 + 校验错误） | 仍失败 → `fallback_template` |
| 无候选且 `stop_reason is None` | 不调用 API，直接报 `ValueError` | 不伪造空结果原因 |

- 最大 API 调用次数 3（首次 + API 重试 + 格式修复，且各自预算独立）。
- Fallback 的 reasons 只用 catalog 标题/价格构造模板句，`limitations` 明确写「模板化兜底输出，不得视为真实 DeepSeek Generator 结果」；`generation_metadata.fallback_used=true` 并记录 `error_kind`。
- **Gate 口径：`model == "fallback_template"` 不算真实 Generator 通过。**

### 3.4 输入契约

- `request_id`、`user_request` 非空；重复候选按出现顺序去重；超过 top_k 截断。
- 无候选时返回空 `recommendations` 的 fallback（要求 `stop_reason` 非空，与 `RecommendationOutput` 既有校验一致）。

### 3.5 语言一致性（Lead 在真实 E2E 中发现的问题及修复）

- **问题**：第一版 prompt 要求 reason 使用中文；用户 query 为英文时，真实 E2E 返回中文理由（本任务修复前的英文 smoke 同样复现，见 §5.3）。
- **修复**：`GENERATOR_PROMPT_VERSION` 仍为规定的 `recommendation_generator_v1`，prompt 规则 3 改为：
  > 所有面向用户的文本（reasons 的 text 与 limitations）必须使用与 current_request 相同的语言：英文请求输出英文理由，中文请求输出中文理由，其他语言同理；不得混用语言，也不得固定输出中文。
  OUTPUT_SCHEMA_EXAMPLE 同步注明「与 current_request 同语言」。
- **测试**：`test_prompt_requires_reason_language_to_match_request` 断言 system prompt 含英/中双向规则。
- **真实复验**：英文 query 改为英文 reasons（§5.3 第二次 EN smoke），中文 query 仍为中文 reasons（§5.3 ZH smoke），两次均通过。

### 3.6 generation_metadata

`GenerationMetadata` 记录 `prompt_version`、`attempts`、`api_retries`、`format_retries`、`fallback_used`、`latency_ms`、`prompt_tokens`、`completion_tokens`、`candidate_count`、`error_kind`、`error`，供 E2E trace / Gate 使用。

---

## 4. 测试

### 4.1 新增测试

| 文件 | 数量 | 覆盖 |
|---|---|---|
| `tests/unit/test_memory_adapter.py` | 17 | unavailable 分支（disabled / 依赖缺失 / 非 qdrant）、lazy 初始化失败 health/read、stub Mem0 read 映射与风险 payload 丢弃、`user_id` filter 参数、`infer=False`、metadata 保留与身份字段过滤、`is_synthetic`、no-record/异常 add、delete/delete_all、health 版本与 768 维锁定、E5 前缀（search=query，add/update=passage）、非法输入 |
| `tests/unit/test_generator.py` | 17 | 成功生成、候选子集、候选外 ID、缺失/空 reason、证据路径不存在、模型乱序不改变输出顺序、模型 title/price 被 catalog 覆盖、prompt 上下文、API 重试、两次 API 失败 fallback、格式修复、持续非法 fallback、空候选、top_k/去重、语言规则、evidence path 展开、候选输入校验 |
| **合计** | **34** | 全部不依赖真实 Qdrant / 真实 DeepSeek API（stub Mem0 对象 + stub client） |

### 4.2 本地执行（2026-11-25，Windows / Python 3.13.11）

```text
python -m pytest tests/unit/test_memory_adapter.py tests/unit/test_generator.py -q
34 passed

python -m pytest -q
315 tests passed（--co 统计 315；全量跑通，无失败）

python -m ruff check src/recommender/memory src/recommender/generation \
  src/recommender/schemas/generation.py tests/unit/test_memory_adapter.py tests/unit/test_generator.py
All checks passed!

python -m mypy src/recommender/memory src/recommender/generation src/recommender/schemas/generation.py
Success: no issues found in 6 source files
```

> 说明：完整 `python -m mypy`（`files=["src/recommender"]`）当前只剩 Lead 尚未提交的 `src/recommender/e2e/service.py:536/538` 6 个 error（本任务未修改该文件，也不在写范围）。已当面报告 Lead。

### 4.3 服务器 venv 复跑（mem0 已安装环境）

```text
cd /tmp/agr/g5 && PYTHONPATH=/tmp/agr/g5/src /tmp/agr/venv/bin/python -m pytest \
  tests/unit/test_memory_adapter.py tests/unit/test_generator.py -q
34 passed
```

---

## 5. 真实验证（真实 Qdrant + 真实模型 + 真实 DeepSeek API）

### 5.1 环境（实际值）

| 项 | 值 |
|---|---|
| 服务器 | `10.123.0.45`，代码快照 `/tmp/agr/g5`（`src/` + `configs/` + `scripts/` + `tests/` 上传） |
| Python venv | `/tmp/agr/venv`（`mem0ai 2.2.1`、`qdrant-client 1.20.0`、`sentence-transformers 5.7.0`） |
| HF_HOME | `/mnt/his_test/group11/agr/hf`（显式导出；`multilingual-e5-base` 已缓存，`HF_HUB_OFFLINE=1`） |
| Qdrant | 容器 `agr-qdrant` up 9h，`127.0.0.1:6333`，运行前 `/collections` 为空库 |
| DeepSeek key | 从服务器 `~/Agentic-Generative-Recommendation/.env` 注入 shell 环境；**未打印、未写入任何文档/日志** |
| collection | `agr_user_memory`（真实配置值） |

运行命令（两次 smoke 相同前缀；`set -a` 加载密钥，但只用于进程环境）：

```bash
cd /tmp/agr/g5
set -a && . ~/Agentic-Generative-Recommendation/.env && set +a
source scripts/env_s4000.sh
export HF_HOME=/mnt/his_test/group11/agr/hf
export HF_HUB_OFFLINE=1
export MEM0_TELEMETRY=False
PYTHONPATH=/tmp/agr/g5/src /tmp/agr/venv/bin/python smoke/mem0_smoke.py
PYTHONPATH=/tmp/agr/g5/src /tmp/agr/venv/bin/python smoke/generator_smoke.py
```

> smoke 脚本仅存在于服务器 `/tmp`（未进入 Git），避免把临时证据/测试用户写进仓库。

### 5.2 真实 Mem0 round-trip（`/tmp/g5_mem0_smoke.json`）

**健康检查（真实返回）**

```json
{
  "available": true, "backend": "mem0", "collection": "agr_user_memory",
  "vector_store": {"provider": "qdrant", "host": "127.0.0.1", "port": 6333,
                   "collection": "agr_user_memory", "embedding_dim": 768, "on_disk": false},
  "embedder": {"model": "intfloat/multilingual-e5-base",
               "revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
               "embedding_dim": 768,
               "memory_action_prefixes": {"add": "passage: ", "update": "passage: ", "search": "query: "}},
  "versions": {"mem0ai": "2.2.1", "qdrant-client": "1.20.0"}
}
```

**写入**（3 条，`infer=False`，`is_synthetic=true`，总耗时 1387.7 ms，含进程内首次 E5 编码）

| user_id | text（测试记忆） | memory_id |
|---|---|---|
| `phase6_smoke_u1` | 用户对香精敏感，偏好无香精的保湿面霜，预算 20 美元以内。 | `dbf0fc7a-2cdb-4063-95a2-ddf4e4953881` |
| `phase6_smoke_u1` | 用户常用的洁面是温和型氨基酸洁面，不喜欢皂基。 | `46da9a2a-f653-4d69-bfe8-f01b5f5462c8` |
| `phase6_smoke_u2` | 用户喜欢电竞机械键盘，轴体偏好红轴，预算 500 元。 | `2efae42e-de41-4b43-bba0-f9bb52b098e4` |

**读取**（真实 Qdrant 向量检索 + `user_id` filter）

| 查询 | 用户 | 延迟 | 命中 | 最高分 |
|---|---|---|---|---|
| 无香精保湿面霜 | u1 | 22.4 ms | 2（本用户保湿/洁面） | 0.9175（保湿面霜） |
| 机械键盘 游戏 | u2 | 20.1 ms | 1（本用户键盘） | 0.8747 |
| 机械键盘 游戏（交叉） | u1 | 19.0 ms | 2（仍是 u1 的 2 条） | 0.8029（洁面） |

- u1 返回条目的 payload `user_id` 全部为 `phase6_smoke_u1`；u2 全部为 `phase6_smoke_u2`；
- 交叉 query 中 **没有** 出现 u2 的键盘记忆（隔离生效）；
- 低相关度的 u1 洁面记忆仍被返回（E5 余弦基线高 + mem0 默认 threshold=0.1），这是 R9 的已知限制（§8）。

**真实 Mem0 原始状态校验**（直接调用 `memory.get_all(filters={"user_id": ...})`）

```text
raw_counts_match_adds = true      # u1=2, u2=1
raw_identity_isolated = true      # 每行 payload user_id 与请求一致
```

**checks（11/11 全部 true）**

```text
raw_counts_match_adds, raw_identity_isolated,
u1_returns_only_u1, u2_returns_only_u2, cross_query_never_leaks_u2,
u1_top_hit_is_relevant, u2_top_hit_is_relevant,
all_synthetic_marked, scores_present,
single_delete_removed, cleanup_removed_all_test_memories
```

**清理**

- 按 `memory_id` 删除 `dbf0fc7a-…` 后，u1 查询只剩洁面 1 条；
- 对两个测试用户 `delete_all` 后，u1/u2 查询均 `[]`；
- Qdrant `agr_user_memory` `points_count`: **0（加之前）→ 3（加之后）→ 0（清理后）**；
- collection 配置保留：768 维、Cosine、`bm25` sparse 槽（未安装 fastembed，BM25 实际未写 sparse 向量，dense-only）。

**空记忆读取**（E2E 新用户场景，独立进程）

```text
available=True backend=mem0
empty_read_hits=0 latency_ms=1288.9
empty_read_ok=True
```

### 5.3 真实 DeepSeek Generator smoke

**候选（真实 catalog：`.../catalog/all_beauty_v2/products.jsonl`）**

| product_id | title（节选） | price | average_rating / rating_number |
|---|---|---|---|
| `B07N5B8WWR` | Facial Moisturizer. Preservative Free. Organic & 100% Natural Face Moisturizing Cream for Sensitive, Dry & Normal Skin… | 29.99 | 4.3 / 35 |
| `B01IAE1UVA` | Secura Moisturizing Cream [59431900] 3 oz (Pack of 6) | 21.53 | 4.4 / 7 |

English query：`I need a fragrance-free facial moisturizer for sensitive skin, unscented and under $30.`
中文 query：`我想要一款适合敏感肌的无香精保湿面霜，预算 30 美元以内。`

| 运行 | 语言 | model | fallback | 延迟 | prompt/completion tokens | attempts / retries |
|---|---|---|---|---|---|---|
| 修复前 EN（暴露语言问题） | 英文 query → **中文理由** | `deepseek-flash` | false | 3199.1 ms | 1815 / 394 | 1 / 0 |
| 修复后 EN | 英文 query → 英文理由 | `deepseek-flash` | false | 2066.7 ms | 1862 / 293 | 1 / 0 |
| 修复后 ZH | 中文 query → 中文理由 | `deepseek-flash` | false | 1743.1 ms | 1862 / 283 | 1 / 0 |

修复后 EN 真实输出（两商品各 3 条理由，完整摘录）：

```text
B07N5B8WWR ($29.99)
  - This facial moisturizer is listed as unscented and is made for sensitive, dry, and normal skin.  [evidence.details.Scent]
  - It is priced at $29.99, which fits your under-$30 budget.                                      [price]
  - It has an average rating of 4.3 from 35 ratings.                                               [evidence.average_rating]
  limitations: The title says preservative free and 100% natural, but the listing does not explicitly state
               fragrance-free beyond the unscented scent label.
B01IAE1UVA ($21.53)
  - This moisturizing cream is listed as unscented and comes as a pack of six 3 oz tubes.           [evidence.details.Scent]
  - It is priced at $21.53, which is under your $30 budget.                                         [price]
  - It has an average rating of 4.4, though from only 7 ratings.                                   [evidence.average_rating]
  limitations: The listing does not specifically state it is formulated for sensitive facial skin…
```

**独立校验（EN post-fix，9/9 true）**

```text
not_fallback_template, ids_are_candidate_subset, ranks_contiguous,
reasons_non_empty, evidence_refs_present, reason_language_matches_request,
catalog_title_price_match, evidence_refs_exist_on_candidate, metadata_complete
```

- 商品 ID ⊆ `{B07N5B8WWR, B01IAE1UVA}`；
- title/price 与 catalog 逐字段相等（模型输出的 title/price 被忽略）；
- 每条 reason 的 `evidence_ref` 都能在候选 evidence 中解析；
- 所有理由/局限为英文。中文 smoke 的 5 项 checks（not_fallback / ids_subset / reasons_non_empty / language_matches_request / evidence_refs_present）全部 true。
- 模型引用的 `4.3/35`、`4.4/7` 已单独回查 catalog 原始行，与真实值一致，未编造。

### 5.4 语言回归的来源与修复

- **发现者**：Lead 在真实 E2E（使用本任务未提交的 Generator/Memory 文件）中运行英文 query，Memory health available、Mem0 feedback write/read 隔离正常、真实 Generator 成功，但 reasons 为中文。
- **本任务**：修复 prompt 规则 3 与 schema 示例，新增单测，并重新跑 EN/ZH 两次真实 smoke（§5.3）；修复前输出关键数字与中文理由已记录在本文件 §5.3（原始 JSON 已被修复后同名运行覆盖）。
- 修复仅改变 prompt 文本，不改变接口、常量值或调用契约；Lead 的 E2E 无需改动。

### 5.5 当前 Qdrant 状态（供 Lead 参考）

任务 smoke 清理后，`agr_user_memory` 当前仅有 Lead 真实 E2E 写入的 1 条 `demo_user_002` 记忆（`is_synthetic=false`）。本任务未删除该数据（不是本任务测试用户），也未把其内容写入文档。

### 5.6 真实 E2E 集成状态（Lead 报告，本任务未重复运行）

Lead 报告：使用本任务未提交文件，真实 E2E 已跑通 Memory health、Mem0 feedback write/read 隔离、真实 DeepSeek Generator（`model=deepseek-flash`，商品 ID 来自 `selected_products` 5 条）。本任务未独立运行 `agr recommend` 全链路，因此 **E2E 整体验收状态以 Lead 的报告为准**；本文件只对 Adapter/Generator 模块的真实行为负责。

---

## 6. 配置与版本锁定

`configs/memory.yaml`（本次更新）：

```yaml
memory:
  enabled: false                       # Phase 0/1 默认关闭；CLI/E2E 显式开启
  mem0:
    vector_store:
      provider: qdrant
      host: 127.0.0.1
      port: 6333
      collection: agr_user_memory
    embedder:
      provider: huggingface
      model: intfloat/multilingual-e5-base
  sqlite:
    path: data/processed/interaction_history.db
```

- **锁定位置**：
  - collection / host / port / model：YAML（`MemoryConfig` 支持的字段）；
  - 768 维、模型 revision、前缀映射、prompt 版本：`src/recommender/memory/mem0_adapter.py` 与 `deepseek_generator.py` 常量（`MemoryConfig`/`GenerationConfig` 未暴露这些字段，本期不扩 `config.py` 以缩小写范围）；
  - 运行期真实版本：`health()["versions"]` 返回 `mem0ai` / `qdrant-client`；Qdrant server 1.20.2 由 D9 记录。
- **依赖版本**：Lead 已在并行工作中把 `pyproject.toml` 的 memory extra 从 `mem0ai>=0.1,<2` 更新为 `mem0ai>=2.2,<3`、`qdrant-client>=1.12,<2`（本任务未修改该文件，避免写冲突）。服务器 venv 实测 `mem0ai 2.2.1`、`qdrant-client 1.20.0`。

---

## 7. 对 Q7 / R8 / R11 / R12 的符合性

| 要求 | 结论 | 证据 |
|---|---|---|
| Mem0 OSS | ✅ `mem0ai 2.2.1` 真实 add/search/delete | §5.2 |
| 独立 Qdrant 服务 | ✅ 容器 `agr-qdrant` @ `127.0.0.1:6333`，应用生命周期外 | §5.1 |
| 持久化数据卷 | ✅ D9 已验证 named volume 重启恢复；本任务未重复重启演练 | D9 §4.2 |
| 单 Collection + user_id 过滤 | ✅ `agr_user_memory` 单 collection；每次 search 带 filter；双用户真实隔离 | §5.2 |
| 本地 Embedding | ✅ `intfloat/multilingual-e5-base` 本地 CPU 推理，无外部 embedding API | §5.2、§5.3 |
| 版本 + 维度锁定 | ✅ 768 维、模型 revision、mem0/qdrant 版本在代码/health/YAML 可追溯 | §6 |
| 数据边界（Mem0 vs SQLite 事件） | ✅ Mem0 history DB 在 `artifacts/mem0/`，未触碰 `InteractionEvent` | §2.3 |
| R8 向量空间隔离 | ✅ 独立 collection，未与商品 `e5-base-v2` 索引混用 | §2.2 |
| R11 跨用户泄漏 | ✅ 代码级 filter + payload 二次校验 + 真实双用户测试 | §5.2 |
| R12 两套 SQLite 混淆 | ✅ 未修改 TIGER/InteractionHistory；Mem0 history DB 独立 | §2.3 |

---

## 8. 未验证项与风险

| # | 未验证 / 风险 | 级别 | 说明 / 建议 |
|---|---|---|---|
| 1 | 真实 E2E 全链路未由本任务重复运行 | 中 | Lead 已报告跑通；本任务只验证 Adapter/Generator。以 Lead 的 E2E 报告 + trace 为最终验收。 |
| 2 | Mem0 `infer=True`（LLM 事实抽取 / ADD-UPDATE-DELETE） | 中 | Adapter 固定 `infer=False`（task-10 要求）；记忆抽取规则与更新冲突处理属于后续 Memory Update 规则引擎，本任务未实现。 |
| 3 | BM25 / spaCy 混合检索未启用 | 低 | 服务器 mem0 提示未安装 `fastembed`/spaCy；本任务为 dense-only 基线，语义检索正常。若启用需装 `mem0ai[extras]` 并重评。 |
| 4 | E5 余弦基线高，`threshold` 未校准（R9） | 中 | 交叉 query 仍返回本用户低相关记忆（0.80）；当前依赖 top-k + 相对排序。不能把 score 当概率或固定阈值判相关性，Phase 7 需校准。 |
| 5 | Memory Update 准入规则（只有有效反馈才写、模拟数据隔离） | 高（模块外） | Adapter 只提供底层写接口与 `is_synthetic` 标记；「何时允许写」由上层 Memory Update 规则引擎/反馈链路决定，本任务未实现。E2E 的 feedback write 是否满足准入需另行审查。 |
| 6 | Memory 并发写 / 多进程 | 低 | 单进程 E2E 已用；Mem0 + Qdrant 的并发 add 未压测。 |
| 7 | Mem0 `update` 路径未真实调用 | 低 | Embedder 映射单测覆盖 update→passage；Adapter 接口不含 update；需要时再补真实 update smoke。 |
| 8 | Qdrant 无鉴权/TLS、备份未自动化（R6/R7） | 中 | D9 已登记；本任务未改变部署形态。 |
| 9 | 语言一致性只在 2 个真实 query 上验证 | 低 | 英文/中文各 1 次真实 smoke + prompt 规则；多样本语言遵循率未系统评测。 |
| 10 | 全量 `mypy` 受 Lead 未提交 `e2e/service.py` 阻塞 | 低 | 6 个 error 位于 `src/recommender/e2e/service.py:536/538`；本任务文件 scoped mypy 通过。 |

---

## 9. 文件与 Git 边界

本次新增/修改（提交信息 `Phase 6: generator and Mem0 adapter`）：

```text
src/recommender/memory/adapter.py          (new)
src/recommender/memory/mem0_adapter.py     (new)
src/recommender/memory/__init__.py         (modified)
src/recommender/generation/deepseek_generator.py (new)
src/recommender/generation/__init__.py     (modified)
src/recommender/schemas/generation.py      (minimal extension: EvidenceRefs / GenerationMetadata)
configs/memory.yaml                        (modified: multilingual-e5-base + Q7 comments)
tests/unit/test_memory_adapter.py          (new, 17 tests)
tests/unit/test_generator.py               (new, 17 tests)
docs/phase6/GENERATOR_MEMORY.md            (new, this file)
```

**未修改 / 未提交**：`cli.py`、`src/recommender/e2e/`、`controller/`、`planner/`、`judgment/`、`config.py`、`pyproject.toml`（Lead 并行修改）、其他 configs、任何模型/索引/密钥/大文件。

---

## 10. 复现

```bash
# 本地（无 mem0 / 无真实 API）
python -m pytest tests/unit/test_memory_adapter.py tests/unit/test_generator.py -q
python -m ruff check src/recommender/memory src/recommender/generation \
  src/recommender/schemas/generation.py tests/unit/test_memory_adapter.py tests/unit/test_generator.py
python -m mypy src/recommender/memory src/recommender/generation src/recommender/schemas/generation.py

# 服务器真实 Mem0 + Qdrant（smoke 脚本为临时文件，不进入 Git）
cd /tmp/agr/g5
set -a && . ~/Agentic-Generative-Recommendation/.env && set +a   # 仅进程环境，不打印
source scripts/env_s4000.sh
export HF_HOME=/mnt/his_test/group11/agr/hf
export HF_HUB_OFFLINE=1
export MEM0_TELEMETRY=False
PYTHONPATH=/tmp/agr/g5/src /tmp/agr/venv/bin/python smoke/mem0_smoke.py

# 真实 DeepSeek Generator（英文 / 中文 query）
PYTHONPATH=/tmp/agr/g5/src /tmp/agr/venv/bin/python smoke/generator_smoke.py
```

*本报告数字均来自上述真实执行输出；证据 JSON 位于服务器 `/tmp`，未进入仓库。*
