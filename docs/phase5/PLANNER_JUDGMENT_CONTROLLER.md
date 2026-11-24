# Phase 5：Planner / Judgment / Adaptive Controller 实现说明

> 版本：V1.0（2026-11-24）  
> 作者：Phase 5 实现（teammate `phase5-deepseek`）  
> 范围：`src/recommender/api/deepseek.py`、`planner/`、`judgment/`、`controller/` 及对应测试  
> 未修改：`cli.py`、`configs/`、`README`、`docs/audit/`、`docs/decisions/`（由 Lead 负责集成）

---

## 1. 交付概览

| 模块 | 文件 | 内容 |
|---|---|---|
| DeepSeek Client | `src/recommender/api/deepseek.py` | httpx Client、超时、有限重试 + 指数退避 + 抖动、并发信号量、错误分类、request/trace id、token/延迟日志、脱敏、JSON mode、推理截断降级 |
| Planner / Replanner | `src/recommender/planner/{prompts.py,planner.py,__init__.py}` | 版本化 Prompt（`planner.initial.v1` / `planner.replan.v1`）、统一 `RecommendationPlan` 校验、有限修复、结构化失败、硬约束保留校验 |
| Judgment | `src/recommender/judgment/{prompts.py,judge.py,__init__.py}` | 批量逐商品判断（`judgment.v1`）、product_id 严格对齐、缺项/多项/重复报错、证据来源校验、只读硬约束上下文 |
| Controller | `src/recommender/controller/{controller.py,workflow.py,__init__.py}` | 纯 Python REPLAN/STOP/CLARIFY 决策、多维预算、重复 query 检测、纯 Python 状态机工作流（LangGraph 未安装，见 §7） |
| 测试 | `tests/unit/test_phase5_{planner,judgment,controller}.py` | 68 个 unit 测试，全部使用 `httpx.MockTransport`，默认不调用真实 API |
| 真实 API 证据 | `docs/phase5/deepseek_smoke_*.json` | 2 次真实运行记录（1 次暴露推理截断问题，1 次修复后成功） |

**测试与检查结果（本机，2026-11-24）**

```text
python -m pytest -q          -> 221 passed, 3 skipped（其中 Phase 5 新增 70 个 unit 测试）
python -m ruff check src tests -> All checks passed
python -m mypy               -> 本 Phase 新增/修改模块 0 error；
                                全仓库剩余 2 个 error 位于其他 teammate 并行开发的
                                src/recommender/retrieval/tiger/generator.py（见 §10）
```

---

## 2. DeepSeek Client（`api/deepseek.py`）

### 2.1 构造与注入

```python
DeepSeekClient(
    *,
    base_url: str | None = None,      # None -> load_settings().config.deepseek.base_url
    model: str | None = None,         # None -> config.deepseek.planner_model
    api_key: str | None = None,       # None -> settings.require_deepseek_api_key()
    timeout_s: float | None = None,
    max_retries: int | None = None,
    max_concurrency: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    use_json_mode: bool = True,
    reasoning_effort: str | None = None,
    allow_reasoning_fallback: bool = True,
    client: httpx.Client | None = None,   # 测试注入 MockTransport
    settings: Settings | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    random_source: Callable[[], float] = random.random,
)
```

- 所有敏感值可注入；key 只来自环境变量 / gitignored `.env`（`load_settings()`），
  **不写入任何文件、不进入日志/trace**。
- 不在 import 时创建 client 或发请求；`close()` / context manager 管理连接。
- 默认超时/重试/并发/temperature/max_tokens 来自 `configs/models.yaml` 的 `deepseek` 段
  （当前 60s / 2 / 4 / 0.2 / 4096）。

### 2.2 请求与错误策略

| 场景 | 行为 |
|---|---|
| URL | `{base_url}/chat/completions`（`base_url` 可含 `/v1`） |
| JSON mode | `response_format={"type":"json_object"}`；若 API 返回 400 且错误信息包含 `response_format`，记录 warning、关闭 JSON mode 并重试一次（兼容探测不计入用户可见 attempt） |
| 超时/网络/408/409/425/429/5xx | 有限重试：最多 `max_retries` 次（默认 2，总请求 ≤ 3）；指数退避 `0.5 * 2^(n-1)` + 10% 抖动，上限 8s；429 优先使用 `Retry-After`（同样受 8s 上限约束） |
| 401/403 | `DeepSeekAuthError`，不重试 |
| 其它 4xx | `DeepSeekRequestError`，不重试 |
| 200 但 envelope 非法/无 choices | `DeepSeekResponseError` |
| 200 但 `finish_reason=length` 且 `content` 为空 | `DeepSeekTruncatedError`（`deepseek_truncated`）；若调用方未显式设置 `reasoning_effort` 且启用 fallback，则自动用 `reasoning_effort="none"` 重试一次；否则显式失败 |
| 模型输出不是合法 JSON 对象 | `DeepSeekJSONError`（客户端只负责分类；修复由 Planner/Judge 的有限修复轮完成） |

所有错误继承 `utils.errors.ExternalAPIError`，带 `provider="deepseek"`、`status_code`、
`retryable`、`kind`、`request_id`、`trace_id`、`attempts`，可直接 `to_dict()` 入 trace。
日志只记录安全字段（model/attempt/latency/status/tokens/finish_reason/error_kind/脱敏摘要），
不记录 Authorization header 或 key。

### 2.3 并发与身份

- 每个 client 实例一个 `threading.BoundedSemaphore(max_concurrency)`，覆盖单次 HTTP 请求；
  退避 sleep 在信号量之外，避免占坑。
- `request_id` / `trace_id` 取自 `observability.context.get_request_context()`；无上下文时生成
  `req_*` / `tr_*`。日志由 `ContextFilter` 附加，`ChatCompletion` 返回值也携带，供 trace 关联。

---

## 3. Planner / Replanner

### 3.1 Prompt 版本

| 版本常量 | 值 | 位置 |
|---|---|---|
| `INITIAL_PROMPT_VERSION` | `planner.initial.v1` | `planner/prompts.py` |
| `REPLAN_PROMPT_VERSION` | `planner.replan.v1` | `planner/prompts.py` |

Prompt 内容为代码内常量（便于 review 与版本 diff），包含：

- 完整统一 `RecommendationPlan` JSON 结构（字段名固定、禁止额外字段）；
- 硬约束只能是当前请求中的确定性条件；Memory/历史偏好只能进软偏好；
- retrieval 只能启用 `system_capabilities.retrieval_modules` 中的路径；
- E5 query 数/长度上限（来自 `PlannerLimits`，默认 5 条 / 300 字符，对应 `configs/retrieval.yaml`）；
- Replanner 必须保留 `previous_plan.constraints.hard` 全部条目、至少生成一条新 query、
  `change_summary` 至少 1 条、`budget_state` 只读。

### 3.2 接口

```python
Planner(client, *, model=None, limits=None, max_attempts=2, temperature=None,
        max_tokens=None, trace=None)
  .plan_initial(PlannerInput) -> PlannerResult
  .plan_replan(ReplanInput) -> PlannerResult
  .generate(mode: PlanMode, input_data) -> PlannerResult   # 模块二 §6.1 统一接口
```

`PlannerInput`：`user_request`、`user_memory`、`session_history`、`system_capabilities`、
可选 `plan_id`。`ReplanInput`：上述 + `previous_plan`、`execution_feedback`、
`controller_feedback`、`budget_state`。

`PlannerResult`：`plan`、`prompt_version`、`model`、`attempts`、`latency_ms`、
`request_id`、`trace_id`、`input_tokens`、`output_tokens`、`repaired`、
`identity_overrides`。

### 3.3 校验与降级

1. 解析 JSON 对象（容忍 markdown fence / 前后缀，提取最外层 `{...}`）。
2. 系统字段归一化：`schema_version`、`plan_id`、`parent_plan_id`、`plan_version`、`mode`
   由编排器断言（模型只负责语义内容），模型写错时记录到 `identity_overrides` 并在日志中 warning。
3. `RecommendationPlan.model_validate`（`extra="forbid"`）。
4. `_validate_capabilities`：启用的召回路径必须在 System Capabilities 内；硬约束字段必须在
   `available_product_fields` 内（防止 `max_price` 这类评估器不支持的字段）。
5. `_validate_limits`：E5 query 数量/长度不超过运行上限。
6. Replan 额外：`validate_replan_preserves_hard_constraints(parent, replan)`（复用现有契约函数）、
   至少一条新 query、`change_summary` 非空。
7. 任一失败 → 将无效输出 + 截断后的校验错误回传，做 **1 次**修复（`max_attempts=2`）；
   仍失败 → 抛 `PlanGenerationError`。

`PlanGenerationError.to_dict()` 包含 `degraded=True`、`plan_generated=False`、`mode`、
`attempts`、`prompt_version`、`error_kind`、`last_error`、`request_id`、`trace_id`。
**不会返回伪造的计划，也没有 `plan` 属性。**

### 3.4 与文档的偏差（已记录，待 Lead 确认）

模块二 §8 写有“多次失败：降级到默认均衡召回计划”，但本 Phase 任务与 AGENTS.md 要求
“失败不得返回空候选/假成功”。本实现的处理：

- Initial Planner 失败 → 结构化错误，工作流 STOP `RETRIEVAL_DEGRADED`，不编造计划；
- Replanner 失败 → 保留上一轮有效计划（final-stage §9.3 的显式降级语义），但**不重复执行同一
  计划**，工作流 STOP `RETRIEVAL_DEGRADED` 并在 reasons 中注明。

这是有意的契约取舍；若负责人要求实现“默认均衡计划”降级，应新增明确的
`degraded_default_plan` 字段与审计标记，而不是复用正常计划结构。

---

## 4. Judgment

### 4.1 Prompt 版本

| 版本常量 | 值 |
|---|---|
| `JUDGMENT_PROMPT_VERSION` | `judgment.v1` |

Prompt 建立在 final-stage §3.5/§3.6 的四个维度上：Preference Fit、Attribute Evidence、
Match Assessment、Uncertainty。额外硬性规则：

- 每个输入商品恰好一条结果、`product_id` 原样复制、不得新增输入外商品；
- 只能依据 `title/price/description/attributes` 实际字段；缺失字段必须 `unknown`，禁止推断；
- `supported/contradicted` 的 `source` 必须引用真实存在的字段（`title` / `description` /
  `price` / `attributes.<key>`），`unknown` 必须 `source=null`；
- 不得输出或修改 `constraint_status`（Python Constraint Evaluator 只读上下文）；
- 三个 preference 集合互斥，元素来自 Plan 中出现的偏好字符串。

### 4.2 接口

```python
Judge(client, *, model=None, prompt_version="judgment.v1", batch_size=10,
      max_attempts=2, temperature=None, max_tokens=None, trace=None)
  .batch_count(product_count) -> int
  .judge(*, request_id, user_request, plan, candidates) -> JudgmentBatch
```

- 输入 `JudgmentCandidate` 序列必须非空且 `product_id` 唯一（否则 `ContractError`）。
- 按 `batch_size` 分块，每块一次 API 请求；块内逐商品独立校验。
- 返回 `JudgmentBatch`，记录顺序严格按输入候选顺序（即使模型乱序）。
- `JudgmentRecord` 携带 `model`、`prompt_version`、batch 级 `latency_ms`、`input_tokens`、
  `output_tokens`（同一批共享、逐条重复，求和只应按批）、`attempts`。

### 4.3 对齐与失败

| 情况 | 行为 |
|---|---|
| items 缺失/非数组/为空 | 校验失败 → 修复轮 |
| 单个 item schema 非法 | 校验失败 → 修复轮 |
| 缺少某商品结果 | `_AlignmentError(missing=...)` → 修复轮 |
| 多出输入外商品 | `_AlignmentError(extra=...)` → 修复轮 |
| 同一商品重复结果 | 归入 extra → 修复轮 |
| `supported` 证据引用不存在的 catalog 字段 | `ValueError` → 修复轮 |
| 修复轮仍失败 | `JudgmentBatchError`（`degraded=True`, `judgment_available=False`, `missing_products`, `extra_products`, `attempts`, `error_kind`, `last_error`） |
| API 超时/HTTP 错误（客户端重试耗尽） | `JudgmentBatchError(error_kind=timeout/auth/server/...)`，不伪造结果 |

工作流捕获 `JudgmentBatchError` 后交给 Controller 走 `api_failure` 降级（STOP
`RETRIEVAL_DEGRADED`/`NO_VALID_CANDIDATES`，`quality_satisfied=false`），不会把失败判断当作
`unknown`/`pass` 继续使用。

### 4.4 D6 Evidence Builder（Catalog 按需补证据，`judgment/evidence.py`）

决策 D6=B：候选池契约保持 `product_id/title/price/main_category/brand` 不变；Reranker 排序后，
只对进入 Judgment 的 Top-N `product_id` 批量查询 Catalog，补充**截断后的 description** 与
**和当前需求相关的 details**。新增 `JudgmentEvidenceBuilder`（默认
`description_max_chars=800`、`max_detail_keys=8`、`max_total_chars=24000`、
`max_total_tokens=None`；设置 token 上限时按 `chars_per_token=4` 折算，取更严格的字符预算）。

```python
JudgmentEvidenceBuilder(
    catalog=None,            # Mapping[str, ProductRecord]（已加载）与 catalog_lookup 二选一
    catalog_lookup=None,     # Callable[[Sequence[str]], Mapping[str, ProductRecord]]
    description_max_chars=800, max_detail_keys=8, max_total_chars=24000,
    max_total_tokens=None, chars_per_token=4,
)
  .build(product_ids, *, plan, catalog=None, catalog_lookup=None,
         constraint_status_by_id=None, extra_terms=()) -> EvidenceBuildResult
```

`EvidenceBuildResult` = `candidates: list[JudgmentCandidate]`（严格按 Reranker 顺序）+ `stats:
EvidenceStats`。`catalog` 与 `catalog_lookup` 互斥；查询函数每次 build 只调用一次，入参为去重后
的 Top-N id（保持顺序）。规则：

- **不推断**：只有真实存在的 Catalog 记录才构造 candidate；缺失商品记入 `missing_products`
  并跳过，不生成空 candidate，也不推断价格/规格/功效。
- **description**：超过 `description_max_chars` 时截断并以 `…` 标记，记入 `truncated_fields`；
  空/空白 description 保持 `None`。
- **details**：由 Plan 的硬约束字段、软偏好、`preferences.{priority,current,historical,negative}`、
  `ranking.priority_attributes` 与 intent 生成大小写/分隔符不敏感的 token 相关性打分（key 命中
  权重高于 value 命中），只保留得分 > 0 的键，最多 `max_detail_keys` 个，稳定排序（得分降序、
  键名升序）。无相关性信号的 details 一律不进入 payload（记为 `irrelevant`），不整包倾倒。
- **基础字段**：`title`、`price` 原样保留；`brand`/`main_category` 放入 `attributes`，缺失保持
  缺失；`features` 不进入 Judgment 视图（避免扩大 D6 范围）。
- **constraint_status**：只从 `constraint_status_by_id` 复制（可为 `None`），本模块不比较价格/
  品类，不把 `unknown`/`None` 升级为 `pass`（Q3）。
- **预算**：总字符预算 = `min(max_total_chars, max_total_tokens × chars_per_token)`；超限时按
  “核心身份字段 > 相关性筛选后的 details > description” 顺序丢弃可选文本，并逐项记录
  `dropped_fields` / `dropped_details`。核心字段（product_id/title/price/brand/main_category）
  永不丢弃，避免破坏“每商品恰好一条 Judgment”的对齐；若核心字段本身超预算，
  `estimated_chars` 会如实超过 `char_limit` 且 `budget_exhausted=true`，而不是静默减少候选。
- **统计**：`missing_products`、`duplicate_products`、`truncated_fields`、`dropped_fields`、
  `dropped_details`（含 reason）、`estimated_chars/tokens`、`char_limit/token_limit`、
  `budget_exhausted`。

Judge 集成（**增量 API，不改现有 `judge()`**）：

```python
Judge.judge_ranked(*, request_id, user_request, plan, ranked_product_ids,
                   catalog=None, catalog_lookup=None,
                   constraint_status_by_id=None, evidence_builder=None) -> RankedJudgment
```

`RankedJudgment` = `{evidence: EvidenceBuildResult, batch: JudgmentBatch}`；内部先
`builder.build` 再走原有 `judge()`，因此“一个商品一条独立结果”、缺失/额外结果校验、证据来源
校验全部沿用。Catalog 全部缺失时抛 `ContractError` 且不调用 API；部分缺失时 warning + trace
`judgment.evidence_built` 记录统计。现有 `RecommendationWorkflow` 仍使用
`candidates_from_pool` 注入点（本任务未改 `controller/`），Lead 集成时可把该注入点替换为
builder 或直接调用 `judge_ranked`。

---

## 5. Adaptive Controller（纯 Python）

### 5.1 质量汇总（`summarize_quality`）

| 字段 | 定义 |
|---|---|
| `valid_product_count` | 只取 Python Constraint Evaluator 的 `valid_product_count`（pass + not_configured），Judgment 不得覆盖 |
| `hard_constraints_satisfied` | 本阶段定义为“已评估商品中没有 fail/unknown，且至少 1 个 valid”；**不宣称已配置/已验证**，`not_configured_count` 仍由 `ConstraintSummary` 保留（Q3） |
| `mean_rerank_score` | 当前排序商品 rerank 分数均值 |
| `match_distribution` | direct/partial/no_match 计数 |
| `evidence_coverage` | 至少有一条 supported/contradicted 证据的商品占比；`None` 表示没有任何 Judgment 记录，**质量不能判定为达标** |
| `unknown_attribute_ratio` | unknown 证据条目 / 全部证据条目（无证据为 `None`） |
| `diversity` | **Phase 5 不计算**（Reranker 契约没有商品相似度）；`min_diversity` 保持 `null` |
| `quality_satisfied` | 由 `evaluate_quality` 按配置阈值判定；要求至少有一条 Judgment 记录；`calibrated=false` 时 stop reason 会注明未校准 |

### 5.2 决策顺序（`Controller.decide`）

1. `plan.clarification.required=True` → `CLARIFY`（`stop_reason=None`）。
2. `api_failure` 参数非空 → `STOP`：`valid=0` 用 `NO_VALID_CANDIDATES`，否则 `RETRIEVAL_DEGRADED`；`quality_satisfied=False`。
3. 本轮证据满足质量阈值 → `STOP / QUALITY_SATISFIED`（`quality_satisfied=True`）。
   质量判定要求至少一条 Judgment 记录（`evidence_coverage != None`），否则视为质量未满足；
   若此时预算也恰好为 0，会在 reasons 中注明“quality 先满足、无需再消耗预算”，
   **不会**把质量达标误报为预算停止。
4. 任一层预算耗尽 → `STOP / BUDGET_EXHAUSTED`，`quality_satisfied=False`、
   `budget_exhausted=True`（预算耗尽绝不标记质量达标）。
5. `valid=0`：还能 replan → `REPLAN`；否则 `STOP / NO_VALID_CANDIDATES`。
6. 质量不足：还能 replan → `REPLAN`；否则 `STOP / MAX_ROUNDS_REACHED`。

预算/轮数条件：`remaining_replan_rounds > 0 && current_round < loop_guard.max_rounds &&
current_round <= state.max_replan_rounds`。

**契约缺口（已记录）**：`StopReason` 没有 `API_FAILED`，API 失败复用
`RETRIEVAL_DEGRADED`/`NO_VALID_CANDIDATES`，并在 `reasons` 中写明 `API failure: ...`。
新增枚举值属于跨模块 schema 决策，留给 Lead 统一处理。

### 5.3 重复 Query 检测

- `seen_query_hashes` 存 SHA-256（normalize：空白折叠 + casefold）。
- Controller 提供 `query_hash` / `plan_query_hashes` / `duplicate_queries`；
  **实际拦截在 workflow 的 replan 节点**：Replanner 返回后，若新计划全部 E5 query 都在本轮之前
  执行过 → 不进入下一轮检索，`STOP / MAX_ROUNDS_REACHED`（reason: loop guard）。
- 部分重复会 warning 但放行（至少有一条新 query 可继续探索）。
- **契约缺口（已记录）**：`configs/controller.yaml` 的 `loop_guard.duplicate_query_similarity=0.95`
  无法用于 State 中仅存的 hash；本实现采用严格 hash 相等。若要支持近似重复，需要 State 保存
  query 文本或逐轮相似度摘要（跨模块契约变更）。

---

## 6. 工作流（`controller/workflow.py`）

`langgraph` 未安装（`pyproject` 的 `[workflow]` extra 未装），因此本 Phase 提供**等价的纯 Python
状态机**，节点/边与模块二 §7、final-stage §4.5 对应：

```text
initial_plan -> clarification_gate --(required)--> CLARIFY（挂起返回）
      |
      v
  retrieve -> rerank -> judge -> controller
      |                            |
      |                     REPLAN +--> replan -> 重复 query 检查
      |                                  |            |
      |                                  +--> retrieve ...
      |
      +--> STOP / CLARIFY -> 结束；STOP 时按证据选择 selected_products
```

工作流语义：

- 初始化：`current_plan is None` 时调用 Initial Planner；失败 → STOP `RETRIEVAL_DEGRADED`。
- clarification gate：Plan 要求澄清 → `CLARIFY`，不执行检索（Phase 5 只做挂起，不实现用户恢复）。
- 预算扣减（本工作流负责）：
  - Planner initial / replan：1 次 API；
  - Retrieval：`len(plan.retrieval.enabled_paths())` 次 retrieval；
  - Judgment：`judge.batch_count(n)` 次 API；
  - Replan：1 次 replan 轮次。
  余额不足时由 `Controller.budget_stop()` 生成 `BUDGET_EXHAUSTED`。
- 失败降级：
  - Retrieval/Reranker 抛 `RecommenderError` → STOP `RETRIEVAL_DEGRADED`；
  - 候选池非空但 Reranker 返回空列表 → STOP `RETRIEVAL_DEGRADED`（不跳过 Judgment 伪造质量）；
  - Judgment 失败 → Controller `api_failure` 分支；
  - Replanner 失败 → 保留父计划，STOP `RETRIEVAL_DEGRADED`；
  - 重复 query → STOP `MAX_ROUNDS_REACHED`。
- `selected_products`：仅当有 `JudgmentRecord`、`match_assessment != no_match` 且
  `uncertainty != high` 的已排序商品，最多 `required_count`（或注入 `select_top_k`）；
  不为了凑数补入未判断商品。Generator 属 Phase 6，此处不生成 `recommendation`。
- 依赖注入：`retrieve(plan) -> CandidatePool`、`rerank(plan, pool) -> Sequence[RankedProduct]`、
  可选 `candidates_from_pool`。因此纯单测不需要 E5 / Reranker / 真实 API。

### LangGraph 说明

未提交未经运行的 LangGraph 适配代码（R18 仍为 Open）。若后续安装 `langgraph`，只需把
`recommendation_workflow` 的节点函数包装为 graph nodes，条件边使用
`stop_reason`/`action` 即可，数据契约不变。文档在此明确“等价纯 Python 状态机”是 Phase 5
的实际实现，不是伪称已接入 LangGraph。

---

## 7. Schema 变更（均为小步扩展，已同步测试）

| 文件 | 变更 | 原因 |
|---|---|---|
| `schemas/judgment.py` | `JudgmentCandidate.constraint_status: ConstraintStatus \| None = None` | Q3：未配置硬约束时状态必须为 `None`（原 schema 必填，无法表达 not_configured，与 `FusedCandidate` 冲突） |
| `schemas/judgment.py` | `JudgmentRecord.attempts: int = 1 (>=1)` | 记录真实 API 重试/推理截断降级，便于审计 |
| `schemas/controller.py` | `RecommendationState.decision: ControllerDecision \| None = None` | 原 State 只有 action/stop_reason 等标量，丢失 reasons；保留完整决策对象以支持 trace，不新造平行字段 |

未更改：`RecommendationPlan`、`BudgetState`、`ControllerDecision`、`ConstraintSummary`、
`CandidatePool` 等既有契约。

---

## 8. 测试证据

文件与数量：

| 文件 | 测试数 | 覆盖 |
|---|---|---|
| `tests/unit/test_phase5_planner.py` | 28 | Client JSON mode/URL/auth header/token/latency、429/500/401/timeout、Retry-After、并发上限、日志脱敏、JSON fence、JSON mode 兼容回退、**推理截断回退**、Planner 成功/身份断言/非法 JSON 修复/两次失败结构化错误/schema 失败/capability/query 限制/API 失败、Replanner 成功/丢硬约束/重复 query/缺 change_summary、统一 dispatch、缺 key |
| `tests/unit/test_phase5_judgment.py` | 17 | 成功对齐（乱序）、只读 constraint_status、缺项/多项/重复、非法枚举、缺失字段伪装 supported、unknown 带 source、额外 constraint_status 字段、attributes 证据、JSON 修复、批量切分、重复/空输入、超时、not_configured 候选 |
| `tests/unit/test_phase5_controller.py` | 25 | 质量汇总、quality STOP、**无 Judgment 不得 quality STOP**、REPLAN、预算耗尽强制 STOP（quality false）、最后一轮质量不被误标预算、max rounds、无有效候选、API 失败降级、CLARIFY、重复 query 规范化、State decision 往返、工作流 quality STOP/REPLAN 循环/重复 query 停止/Planner 失败/Replanner 失败/Judgment 失败/预算不足/选取过滤/检索失败/**Rerank 空结果降级** |

全部 68 个测试使用 `httpx.MockTransport` 与 fake sleeper，**默认不访问真实 API**，无网络依赖。

### 8.1 D6 Evidence Builder 测试证据（2026-11-24）

`tests/unit/test_judgment_evidence.py`：**22 个单测全部通过**（`httpx.MockTransport`，无真实
API）。覆盖：

| 覆盖点 | 用例 |
|---|---|
| description 截断/保留/空值 | 超长截断并带 `…` 且记入 `truncated_fields`；短文本原样；空白 description 保持 `None` |
| details 相关性筛选 | 大小写不敏感 key 命中；value 命中；无信号时不倾倒 details；无关 details 记为 `irrelevant` |
| details 数量上限 | `max_detail_keys` 生效，超出记为 `max_detail_keys` |
| 基础字段 | title/price/brand/main_category 保留 |
| 缺失商品 | 记入 `missing_products`、跳过且不伪造；lookup 返回错 id 也按缺失处理 |
| 对齐 | 输出顺序 = Reranker 顺序；重复 id 去重并记录 |
| Catalog 查询 | `catalog_lookup` 每 build 只调用一次且入参保持顺序；无 catalog 源/空 id 抛 `ContractError` |
| constraint_status | 上游传入 `unknown` 原样保留；未传入为 `None`，不因价格便宜自动 `pass`（Q3） |
| 字符预算 | 超限截断 description，核心字段不丢，`estimated_chars <= char_limit` |
| 核心字段超预算 | 保留全部 product_id 对齐并置 `budget_exhausted=true`、记录 dropped 字段 |
| Token 预算 | `max_total_tokens=40, chars_per_token=4` → 有效字符预算 160 |
| Judge 集成 | `judge_ranked` 走 builder + stub DeepSeek：乱序结果按输入对齐、详情过滤进入 prompt、缺失商品不送审、全缺失不调用 API |

命令与结果：

```bash
python -m pytest tests/unit/test_judgment_evidence.py -q   # 22 passed
python -m pytest tests/unit/test_phase5_judgment.py -q     # 17 passed（既有 Judge 行为未变）
```

运行命令：

```bash
python -m pytest -q
python -m ruff check src tests
python -m mypy
```

---

## 9. 真实 DeepSeek API Smoke（2026-11-24）

运行环境：本机 Windows，项目根 `.env`（gitignored，权限 600）提供 `DEEPSEEK_API_KEY`；
base_url `https://api.deepseek.com`，模型 `deepseek-flash`，temperature 0.2，planner
max_tokens 2048，timeout 60s。key 未写入任何文件、未出现在日志/证据 JSON 中。

证据文件：

- `docs/phase5/deepseek_smoke_20261124T033242Z.json`：**首次运行，Judgment 失败**；
- `docs/phase5/deepseek_smoke_20261124T033749Z.json`：**修复后成功**（本次结论以此为准）。

### 9.1 最终成功结果（`...T033749Z.json`）

| 项 | 值 |
|---|---|
| Planner 状态 | success |
| 模型 | `deepseek-flash` |
| planner attempts | 1（JSON mode 一次成功） |
| planner 延迟 | **2735.9 ms** |
| planner tokens | input **1308** / output **672** |
| Plan 硬约束 | `price <= 30 USD`（保留 current_request 来源） |
| Plan E5 queries | 3 条（lightweight/fragrance-free/under $30 方向） |
| clarification | `required=false` |
| Judgment 状态 | success（2 个示例商品） |
| Judgment 每商品 attempts | **2**（第 1 次推理耗尽 max_tokens，自动 `reasoning_effort="none"` 重试成功） |
| Judgment tokens（batch 级） | input **1476** / output **230** |
| Judgment 结果 | `smoke_p1`: direct_match / low；`smoke_p2`: no_match / low（价格超限且明确强香型） |

说明：Judgment 的候选是 `smoke_p1` / `smoke_p2` 两个**示例商品**（非真实 Amazon 目录），
仅用于验证结构化输出与对齐，不能作为真实效果指标。

### 9.2 首次失败与根因（R17 重要发现）

`...T033242Z.json` 中 Planner 成功（1610.5 ms，1309/375 tokens），但 Judgment 报
`DeepSeek response has no textual message content`（`error_kind=response`）。

随后用原始 HTTP 探针（不落盘、不打印 key）确认：

| 探针 | 结果 |
|---|---|
| Judgment prompt，2 商品，`max_tokens=2048` | 200，`finish_reason=length`，`content` 空，`reasoning_content` 9704 字符，completion_tokens 2048 |
| 同上 `max_tokens=4096` | 200，`finish_reason=length`，`content` 空，`reasoning_content` 18952 字符，completion_tokens 4096 |
| 1 商品 `max_tokens=4096` | 200，`finish_reason=length`，`content` 空，reasoning 19284 字符 |
| 1 商品 `max_tokens=8192` | 200，`finish_reason=stop`，`content` 488 字符，completion_tokens 7428 |
| 2 商品 `max_tokens=8192` | 200，`finish_reason=stop`，`content` 738 字符，completion_tokens 5694 |
| 2 商品 + `reasoning_effort="none"`，`max_tokens=2048` | 200，`finish_reason=stop`，`content` 826 字符，reasoning 0，completion_tokens **236** |
| 2 商品 + `reasoning_effort="minimal"` | 200，`finish_reason=stop`，content 775，reasoning 7748，completion_tokens 1826 |
| 2 商品 + `thinking={"type":"disabled"}` | 200，`finish_reason=stop`，content 810，reasoning 0，completion_tokens 229 |

结论：

1. 当前 `deepseek-flash` 是**推理模型**；Judgment 这类多商品结构化任务在默认推理模式下会消耗
   数千 reasoning tokens，`configs/models.yaml` 的 `max_tokens=4096` 不足时 `content` 为空。
2. **`reasoning_effort="none"` 是真实生效的参数**：关闭推理后 2 商品 Judgment 仅需约 236
   completion tokens，且 JSON 结构完整（826 字符）。
3. 本实现已加入“空 content + finish_reason=length”分类（`deepseek_truncated`）与有界自动回退
   （`reasoning_effort="none"`），真实 API 上已验证可从失败恢复（attempts=2）。
4. **建议 Lead 在配置层决定**：为 Planner/Judgment 增加 `reasoning_effort` 配置项，并把结构化
   提取类调用默认设为 `none`（本任务不允许改 `configs/` 与 `config.py`，故只做了构造函数注入与
   自动回退，未写死到业务代码）。R17 状态应保持 `In Progress`，直至配置落地并完成
   “真实 429/5xx/配额”验证。

---

## 10. 未验证项与跨模块问题

1. **全仓库 mypy**：本 Phase 文件全部通过；剩余 2 个 error 在
   `src/recommender/retrieval/tiger/generator.py`（其他 teammate 并行开发中），本任务未修改该文件。
   Lead 合并前需确认该文件类型检查转绿。
2. **真实 API 的 429/5xx/网络超时**：仅用 MockTransport 验证；真实环境只验证了 200 成功路径与
   推理截断失败路径。
3. **真实 10 商品批量 Judgment**：只验证了 2 商品 smoke；批量成本/延迟需在 Phase 7 评估。
   若启用 `reasoning_effort=none`，成本预计接近 probe 的 236 completion tokens/2 商品。
4. **LangGraph**：未安装、未运行；当前为等价纯 Python 状态机。R18 仍需在安装后做
   REPLAN/STOP/预算耗尽/重复 query 的 graph 级集成测试。
5. **真实端到端链路**：未在本任务中串接真实 E5 + FAISS + Qwen3-Reranker + Judgment + Controller
   （由 Lead 在集成时完成）；本任务的 workflow 使用注入的 retrieve/rerank 依赖，单测已覆盖分支。
6. **Controller 阈值与预算**：`configs/controller.yaml` 的 `required_count/min_valid_candidates/
   max_unknown_attribute_ratio/预算值` 均未用验证集校准（`calibrated=false`）；实现只消费配置，
   未宣称已验证有效。
7. **多样性**：`QualityMetrics.diversity` 在 Phase 5 恒为 `None`；不能把 `min_diversity` 设为非空，
   否则 quality 永远无法满足。
8. **Catalog 元数据缺口**：`fusion.pool._metadata` 只输出 title/price/main_category/brand，
   Judgment 默认视图缺少 description/details；工作流保留 `candidates_from_pool` 注入点，
   Lead 集成时可用真实 catalog 补齐（本任务未改 fusion，避免跨 scope）。
9. **Planner 默认“降级计划”**：见 §3.4，文档与任务要求存在解释差异，已按任务要求实现为显式失败。

---

## 11. 关键复现命令（供 Lead 集成）

```bash
# 单元测试（无真实 API）
python -m pytest tests/unit/test_phase5_planner.py tests/unit/test_phase5_judgment.py \
                 tests/unit/test_phase5_controller.py -q

# 全量
python -m pytest -q
python -m ruff check src tests
python -m mypy

# 真实 smoke（需要项目根 .env 提供 DEEPSEEK_API_KEY；结果写到 docs/phase5/）
# 运行方式：一次性脚本（本任务未把临时脚本提交入库），
# 证据见 docs/phase5/deepseek_smoke_20261124T033749Z.json
```

集成建议顺序：`load_settings()` → `DeepSeekClient(settings=settings, max_tokens=...)` →
`Planner` / `Judge` → `Controller(config=settings.config.controller)` →
`RecommendationWorkflow(WorkflowDependencies(...), WorkflowOptions(...))`，
retrieve/rerank 由 Phase 2/3 服务注入；`BudgetState` 用 `Controller.initial_budget()` 初始化。
