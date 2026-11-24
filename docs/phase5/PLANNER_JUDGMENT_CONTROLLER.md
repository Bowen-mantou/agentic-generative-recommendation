# Phase 5：Planner / Judgment / Adaptive Controller 实现说明

> 版本：V1.1（2026-11-24，Phase 5 硬化 D3–D5/D7–D8；D6 见 §4.4/§8.1）  
> 作者：Phase 5 实现（teammate `phase5-deepseek`）；硬化与 D6 增补（`phase5-hardening` / `judgment-evidence`）  
> 范围：`src/recommender/api/deepseek.py`、`planner/`、`judgment/`、`controller/`、`schemas/{common,controller}.py`、`config.py`、`configs/controller.yaml` 及对应测试  
> 未修改：`cli.py`、`configs/`（仅 `controller.yaml`）、`configs/tiger/`、`docs/audit/`、`docs/decisions/`、`retrieval/tiger/`

---

## 1. 交付概览

| 模块 | 文件 | 内容 |
|---|---|---|
| DeepSeek Client | `src/recommender/api/deepseek.py` | httpx Client、超时、有限重试 + 指数退避 + 抖动、并发信号量、错误分类、request/trace id、token/延迟日志、脱敏、JSON mode、推理截断降级 |
| Planner / Replanner | `src/recommender/planner/{prompts.py,planner.py,__init__.py}` | 版本化 Prompt（`planner.initial.v1` / `planner.replan.v1`）、统一 `RecommendationPlan` 校验、有限修复、结构化失败（D7 默认）、显式确定性降级计划、硬约束保留校验 |
| Judgment | `src/recommender/judgment/{prompts.py,judge.py,judge_evidence.py,__init__.py}` | 批量逐商品判断（`judgment.v1`）、product_id 严格对齐、缺项/多项/重复报错、证据来源校验、只读硬约束上下文；D6 Evidence Builder（§4.4） |
| Controller | `src/recommender/controller/{controller.py,workflow.py,langgraph_adapter.py,__init__.py}` | 纯 Python REPLAN/STOP/CLARIFY 决策、多维预算、D4 `api_failed` 失败组件、D5 严格 query hash、D8 `min_judgments`、共享 step 函数 + LangGraph 正式运行时（D3） |
| 测试 | `tests/unit/test_phase5_{planner,judgment,controller}.py`、`tests/unit/test_judgment_evidence.py`、`tests/integration/test_langgraph_workflow.py` | 89 个 Phase 5 单测 + 22 个 D6 单测 + 9 个真实 LangGraph 集成测试，默认不调用真实 API |
| 真实 API 证据 | `docs/phase5/deepseek_smoke_*.json` | 2 次真实运行记录（1 次暴露推理截断问题，1 次修复后成功） |

**测试与检查结果（本机 + S4000，2026-11-24 硬化后）**

```text
本地 Windows（无 langgraph，run() 走纯 Python 回退）
python -m pytest -q             -> 260 passed, 14 skipped
python -m ruff check src tests  -> All checks passed
python -m mypy                  -> Success: no issues found in 66 source files

S4000 /tmp/agr/venv（langgraph 1.2.24，run() 走真实 graph）
python -m pytest tests/unit                          -> 260 passed, 2 skipped
python -m pytest tests/integration/test_langgraph_workflow.py -> 9 passed
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
`identity_overrides`，以及 D7 新增的 `degraded: bool = False`、
`plan_source: str | None = None`（仅显式降级计划为 `True`/`"system_default"`）。
构造器新增可选 `fallback: PlannerFallbackConfig | None`，默认关闭降级。

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

**D7 硬化（2026-11-24 决策 D7=C）**：以上“抛结构化错误”是默认行为
（`configs/controller.yaml` 的 `planner.fallback.enabled: false`）。只有显式开启
`planner.fallback.enabled: true` 时，Planner 在重试耗尽后返回**确定性 system_default
计划**：`PlannerResult.degraded=True`、`plan_source="system_default"`，
`plan.assumptions` 含 `Planner API unavailable` / `Using deterministic balanced retrieval`
/ `degraded: plan_source=system_default`，并写 `planner.plan_degraded` trace 事件。
`RecommendationPlan` 没有 `degraded`/`plan_source` 字段（本任务未改 `schemas/plan.py`），
因此降级标记落在 `assumptions` + `PlannerResult` + trace 三处，不伪装成 Planner API 成功。

### 3.4 与模块二 §8 的偏差与 D7 决策（已确认）

模块二 §8 写有“多次失败：降级到默认均衡召回计划”。D7=C 已明确：

- **默认**（`enabled: false`）：Initial Planner 失败 → 结构化错误；工作流 STOP
  `API_FAILED`（`failure.failed_component=planner`），**不编造计划**；
- Replanner 失败 → 保留上一轮有效计划，但**不重复执行同一计划**；工作流 STOP
  `API_FAILED`（`failed_component=planner`）并在 reasons 中注明；
- **显式开启**（`enabled: true`）：返回确定性 balanced 计划，带上述 `degraded` /
  `plan_source` / `assumptions` 标记；它不是“Planner API 成功”，不得在评测中按正常计划统计；
- 确定性降级计划无法构建时（例如系统没有任何召回路径）抛
  `PlanFallbackError`（`code=plan_fallback_failed`，`__cause__` 为原
  `PlanGenerationError`），不会悄悄返回空计划。

降级计划保留父计划硬约束/偏好（Replan）或明确声明“未提取硬约束”（Initial），
因此不会静默放宽用户约束；Initial 降级计划的 `constraints.hard` 为空是显式记录在
`assumptions` 中的已知降级，而不是验证通过的约束。

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

工作流捕获 `JudgmentBatchError` 后交给 Controller 走 D4 的 API 失败降级：STOP
`API_FAILED`，并在 `ControllerDecision.failure` 中记录
`failed_component=judgment`、`error_type=error_kind`、`retryable`、`attempts`、`message`，
`quality_satisfied=false`。它不会把失败判断当作 `unknown`/`pass` 继续使用，也不会把 Judgment
故障误报成 `retrieval_degraded` 或 `no_valid_candidates`。

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
| `valid_judgment_count` | **D8 新增**：`product_id` 与当前 `ranked_products` 对齐的去重 Judgment 数；与候选集不对齐的记录不算有效证据 |
| `evidence_coverage` | 至少有一条 supported/contradicted 证据的商品占比；`None` 表示没有任何 Judgment 记录，**质量不能判定为达标** |
| `unknown_attribute_ratio` | unknown 证据条目 / 全部证据条目（无证据为 `None`） |
| `diversity` | **Phase 5 不计算**（Reranker 契约没有商品相似度）；`min_diversity` 保持 `null` |
| `quality_satisfied` | 由 `evaluate_quality` 按配置阈值判定；要求至少一条 Judgment 记录且 `valid_judgment_count >= min_judgments`；`calibrated=false` 时 stop reason 会注明未校准 |

### 5.2 决策顺序（`Controller.decide`）

1. `plan.clarification.required=True` → `CLARIFY`（`stop_reason=None`）。
2. `failure` 参数非空（D4）→ `STOP / API_FAILED`，`ControllerDecision.failure` 记录
   `failed_component`（planner/judgment/generator）与 `error_type`/`retryable`/`attempts`/
   `message`；`quality_satisfied=False`。**不再复用**
   `RETRIEVAL_DEGRADED`/`NO_VALID_CANDIDATES`，无论当时是否有有效候选。
3. 本轮证据满足质量阈值 → `STOP / QUALITY_SATISFIED`（`quality_satisfied=True`）。
   质量判定要求至少一条 Judgment 记录（`evidence_coverage != None`）且
   `valid_judgment_count >= min_judgments`（D8），否则视为质量未满足；
   若此时预算也恰好为 0，会在 reasons 中注明“quality 先满足、无需再消耗预算”，
   **不会**把质量达标误报为预算停止。
4. 任一层预算耗尽 → `STOP / BUDGET_EXHAUSTED`，`quality_satisfied=False`、
   `budget_exhausted=True`（预算耗尽绝不标记质量达标）。
5. `valid=0`：还能 replan → `REPLAN`；否则 `STOP / NO_VALID_CANDIDATES`。
6. 质量不足：还能 replan → `REPLAN`；否则 `STOP / MAX_ROUNDS_REACHED`。

预算/轮数条件：`remaining_replan_rounds > 0 && current_round < loop_guard.max_rounds &&
current_round <= state.max_replan_rounds`。

D8 说明：`controller.quality.min_judgments` 初始值 3（工程起点，未校准）；
2 条有效 Judgment 在 `min_judgments=3` 下只能 REPLAN/停止，不能 quality STOP；
Top-K 中每个被选中商品都必须在 `judgment_results` 中有对应记录（`_select_products` 跳过
无 Judgment 的已排序商品），因此不会用未判断商品凑满 Top-K。
“有效 Judgment”= 结构化解析成功 + Schema 校验通过（`JudgmentRecord`）+ `product_id` 与输入
候选对齐；失败批次的 `JudgmentBatchError` 不产生 `JudgmentRecord`，不计入数量。
配置位置为 `controller.quality.min_judgments`（决策草案写作 `controller.min_judgments`；
本实现把它归入 quality 阈值组，便于 `evaluate_quality` 直接消费，语义与门槛完全相同）。

### 5.3 重复 Query 检测

- `seen_query_hashes` 存 SHA-256（D5 严格归一化：`unicodedata.normalize("NFKC")` → 空白折叠
  → `casefold`；全角/大小写/制表符/表意空格差异归一为同一 hash）。
- 不删除停用词、不重排词序：那会改变需求语义而不是归一化文本。
- Controller 提供 `query_hash` / `plan_query_hashes` / `duplicate_queries`；
  **实际拦截在 workflow 的 replan 节点**：Replanner 返回后，若新计划全部 E5 query 都在本轮之前
  执行过 → 不进入下一轮检索，`STOP / MAX_ROUNDS_REACHED`（reason: loop guard）。
- 部分重复会 warning 但放行（至少有一条新 query 可继续探索）。
- **D5 已删除** `configs/controller.yaml` 的 `loop_guard.duplicate_query_similarity`：
  State 只保存 hash，相似度阈值无法执行；`LoopGuardConfig` 同步移除该字段。若要支持近似重复，
  需要 State 保存 query 文本或逐轮相似度摘要（跨模块契约变更），本阶段不做。

---

## 6. 工作流（`controller/workflow.py`）

Phase 5 硬化后工作流被拆成**共享 step 函数**，由两个运行时执行同一份业务逻辑：

| step | 职责 |
|---|---|
| `step_prepare` | clamp loop guard、写 `workflow.started` |
| `step_plan` | Initial Planner（失败 → `API_FAILED`/planner 或 D7 降级计划），记录 query hash |
| `step_clarification` | clarification gate → `CLARIFY` |
| `step_retrieve` | 预算检查/扣减 + 召回 → CandidatePool/ConstraintSummary |
| `step_rerank` | 精排；候选非空但排序为空 → `RETRIEVAL_DEGRADED` |
| `step_judge` | Judgment；`JudgmentBatchError` → `API_FAILED`/judgment |
| `step_controller` | 汇总质量指标并决策 |
| `step_replan` | Replanner（失败 → `API_FAILED`/planner）+ 重复 query 拦截 |
| `finalize` | 写入 decision、选择 `selected_products`、写 `workflow.finished` |

- **正式入口**：`RecommendationWorkflow.run(state)`。安装 `langgraph` 时走真实 graph；
  未安装时记录 WARNING 后回退到 `run_python`（明确日志，不静默）。
- `RecommendationWorkflow.run_langgraph(state)`：强制正式运行时；缺少依赖时抛
  `LangGraphUnavailableError`，不会假装执行。
- `RecommendationWorkflow.run_python(state)`：纯 Python 状态机（业务核心 + 离线回退）。
- 节点/边与模块二 §7、final-stage §4.5 对应：

```text
prepare -> planner -> clarification_gate --(required)--> finalize(CLARIFY)
                          |
                          v
   retrieve -> rerank -> judge -> controller
       ^                            |
       |                     REPLAN +--> replanner -> 重复 query 检查
       |                                  |            |
       |                                  +--> retrieve ...
       |
       +--> STOP -> finalize -> 结束；STOP 时按证据选择 selected_products
```

工作流语义：

- 初始化：`current_plan is None` 时调用 Initial Planner；失败 → STOP
  `API_FAILED`（D4，`failed_component=planner`），不编造计划。
- clarification gate：Plan 要求澄清 → `CLARIFY`，不执行检索（Phase 5 只做挂起，不实现用户恢复）。
- 预算扣减（本工作流负责）：
  - Planner initial / replan：1 次 API；
  - Retrieval：`len(plan.retrieval.enabled_paths())` 次 retrieval；
  - Judgment：`judge.batch_count(n)` 次 API；
  - Replan：1 次 replan 轮次。
  余额不足时由 `Controller.budget_stop()` 生成 `BUDGET_EXHAUSTED`。
- 失败降级（D4）：
  - Initial Planner / Replanner 失败 → STOP `API_FAILED`（`failed_component=planner`）；
    Replanner 失败时保留父计划但不重复执行；
  - Judgment 失败 → STOP `API_FAILED`（`failed_component=judgment`）；
  - Retrieval/Reranker 抛 `RecommenderError` → STOP `RETRIEVAL_DEGRADED`（不是模型 API 故障）；
  - 候选池非空但 Reranker 返回空列表 → STOP `RETRIEVAL_DEGRADED`（不跳过 Judgment 伪造质量）；
  - Generator 属 Phase 6；`FailedComponent.GENERATOR` 已预留，当前工作流不产生该失败；
  - 重复 query → STOP `MAX_ROUNDS_REACHED`。
- 重试成功不提前 STOP：API client 内部有界重试成功时正常返回（测试
  `test_planner_retry_success_is_not_marked_degraded` / `test_client_retries_429_then_succeeds`），
  只有重试耗尽才进入 `API_FAILED`。
- `selected_products`：仅当有 `JudgmentRecord`、`match_assessment != no_match` 且
  `uncertainty != high` 的已排序商品，最多 `required_count`（或注入 `select_top_k`）；
  不为了凑数补入未判断商品。Generator 属 Phase 6，此处不生成 `recommendation`。
- 依赖注入：`retrieve(plan) -> CandidatePool`、`rerank(plan, pool) -> Sequence[RankedProduct]`、
  可选 `candidates_from_pool`。因此纯单测不需要 E5 / Reranker / 真实 API。

### LangGraph 说明（D3=A）

`pyproject.toml` 的 `[workflow]` extra 在 S4000 服务器 `/tmp/agr/venv` 中真实安装
（`langgraph` 1.2.24、`langchain-core` 1.7.7，清华 tuna 镜像），`controller/langgraph_adapter.py`
把上表 step 函数逐个包装为 graph node，条件边按 `decision`/`action` 路由：

```text
START -> prepare -> planner -> clarification_gate -> retrieve -> rerank -> judge
                                                                              |
   finalize <- (STOP/CLARIFY) <- controller <- (judgment ok) ------------------+
        ^                          |
        +-- (REPLAN) -> replanner -+-> retrieve ...
```

- LangGraph 只负责编排、状态传递与条件转移；约束/预算/质量/降级判定仍是 Python。
- `run()` 是正式入口，安装依赖时**必然**经 LangGraph；`run_python()` 仅作业务核心与显式回退。
- 测试：`tests/integration/test_langgraph_workflow.py` 用 stub 依赖执行**真实** graph
  （quality STOP / REPLAN 循环 / Judgment 失败 / Planner 失败 / CLARIFY / 预算耗尽 /
  重复 query / 与纯 Python 状态完全一致），S4000 上 9 项全部通过；
  同一套 260 个单测在 S4000（langgraph 已装，`run()` 走 graph）与本地（无 langgraph，
  走纯 Python 回退）均通过，证明两条运行时语义一致。
- R18 由 Open 转 In Progress：graph 级集成测试已在真实 langgraph 上运行，但尚未接入
  真实 DeepSeek + E5 + Reranker 的服务器端端到端链路。

---

## 7. Schema 变更（均为小步扩展，已同步测试）

| 文件 | 变更 | 原因 |
|---|---|---|
| `schemas/judgment.py` | `JudgmentCandidate.constraint_status: ConstraintStatus \| None = None` | Q3：未配置硬约束时状态必须为 `None`（原 schema 必填，无法表达 not_configured，与 `FusedCandidate` 冲突） |
| `schemas/judgment.py` | `JudgmentRecord.attempts: int = 1 (>=1)` | 记录真实 API 重试/推理截断降级，便于审计 |
| `schemas/controller.py` | `RecommendationState.decision: ControllerDecision \| None = None` | 原 State 只有 action/stop_reason 等标量，丢失 reasons；保留完整决策对象以支持 trace，不新造平行字段 |
| `schemas/common.py` | `StopReason.API_FAILED = "api_failed"`（D4） | API 重试耗尽必须与检索退化/无候选/预算停止区分 |
| `schemas/common.py` | `FailedComponent`（planner/judgment/generator，D4） | 记录失败组件，限定三种模型调用 |
| `schemas/controller.py` | `FailureInfo`（D4，全部字段可选除 `failed_component`） | `api_failed` 的可序列化失败记录 |
| `schemas/controller.py` | `ControllerDecision.failure: FailureInfo \| None = None`（D4） | 仅 `API_FAILED` 携带；validator 强制成对出现，其他决策序列化不变 |
| `schemas/controller.py` | `QualityMetrics.valid_judgment_count: int = 0`（D8） | 质量 STOP 需要有效 Judgment 数量 |
| `config.py` | `QualityThresholds.min_judgments: int = 3`（D8） | 配置化质量门槛 |
| `config.py` | `PlannerFallbackConfig` / `PlannerConfig`（D7） | `planner.fallback.enabled=false` 默认结构化失败 |
| `config.py` | `LoopGuardConfig.duplicate_query_similarity` 移除（D5） | 无 query 原文时无法执行；严格 hash 替代 |

未更改：`RecommendationPlan`（因此 D7 降级标记写 `assumptions`）、`BudgetState`、
`ConstraintSummary`、`CandidatePool` 等既有契约。`ControllerDecision` 仅新增可选
`failure` 字段，旧 payload 反序列化保持兼容。

---

## 8. 测试证据

文件与数量：

| 文件 | 测试数 | 覆盖 |
|---|---|---|
| `tests/unit/test_phase5_planner.py` | 36 | Client JSON mode/URL/auth header/token/latency、429/500/401/timeout、Retry-After、并发上限、日志脱敏、JSON fence、JSON mode 兼容回退、**推理截断回退**、Planner 成功/身份断言/非法 JSON 修复/两次失败结构化错误/schema 失败/capability/query 限制/API 失败、Replanner 成功/丢硬约束/重复 query/缺 change_summary、统一 dispatch、缺 key、**D7 默认失败/显式确定性降级（initial+replan）/无召回路径 PlanFallbackError/API 错误也触发降级/默认关闭**、**重试成功不降级**、**D5 全角+大小写+制表符变体仍判重复 query** |
| `tests/unit/test_phase5_judgment.py` | 17 | 成功对齐（乱序）、只读 constraint_status、缺项/多项/重复、非法枚举、缺失字段伪装 supported、unknown 带 source、额外 constraint_status 字段、attributes 证据、JSON 修复、批量切分、重复/空输入、超时、not_configured 候选 |
| `tests/unit/test_phase5_controller.py` | 34（+2 条件跳过） | 质量汇总、quality STOP、**无 Judgment 不得 quality STOP**、REPLAN、预算耗尽强制 STOP（quality false）、最后一轮质量不被误标预算、max rounds、无有效候选、CLARIFY、State decision 往返、工作流 quality STOP/REPLAN 循环/重复 query 停止/Planner 失败/Replanner 失败/Judgment 失败/预算不足/选取过滤/检索失败/**Rerank 空结果降级**、**D4 api_failed+failed_component（planner/judgment/generator 往返、无候选仍 api_failed）**、**D5 NFKC 严格 hash（全角/大小写/多空白/表意空格，且不排序/不去停用词）**、**D8 valid_judgment_count 对齐过滤 + 2 条不达标/3 条达标 + 未判断商品不进 Top-K**、**D3 run/run_python/run_langgraph 派发与回退（无 langgraph 时 run_langgraph 报错、有 langgraph 时两运行时 State 完全一致）** |
| `tests/integration/test_langgraph_workflow.py` | 9 | **真实 langgraph** 编译图实例、quality STOP、REPLAN 两轮、Judgment 失败→api_failed/judgment、Planner 失败→api_failed/planner、CLARIFY 不出检索、预算耗尽、重复 query loop guard、与纯 Python State 完全一致 |

全部 **89 个 Phase 5 单测**（本地无 langgraph 时 2 个运行时分支跳过，实际执行 87 个）使用 `httpx.MockTransport` / stub 依赖，**默认不访问真实 API，
无网络依赖**；本地无 `langgraph` 时 2 个“真实运行时”分支单测与 9 个集成测试 skip，服务器上
全部实际执行。D6 Evidence Builder 的 22 个单测见 §8.1。

### 8.2 Phase 5 硬化（D3–D5, D7–D8）测试证据（2026-11-24）

命令与结果：

```bash
# 本地 Windows / Python 3.13.11（无 langgraph，run() 走纯 Python 回退）
python -m pytest -o addopts="" -q        # 260 passed, 14 skipped
python -m ruff check src tests           # All checks passed!
python -m mypy                           # Success: no issues found in 66 source files

# S4000 192.0.2.10 / Python 3.10.12 / /tmp/agr/venv（langgraph 1.2.24）
# run() 必须走 LangGraph（use_langgraph=True 默认）
PYTHONPATH=src /tmp/agr/venv/bin/python -m pytest -o addopts="" -q tests/unit
                                         # 260 passed, 2 skipped
PYTHONPATH=src /tmp/agr/venv/bin/python -m pytest -o addopts="" -q \
    tests/integration/test_langgraph_workflow.py
                                         # 9 passed
```

决策 → 关键用例：

| 决策 | 关键用例（文件::用例） |
|---|---|
| D4=api_failed+failed_component | `test_phase5_controller.py::test_controller_api_failure_stops_with_api_failed_and_component`、`::test_controller_api_failure_without_valid_candidates_is_still_api_failed`、`::test_controller_api_failed_decision_roundtrips_through_state`、`::test_workflow_initial_planner_failure_stops_api_failed_planner`、`::test_workflow_replanner_failure_keeps_previous_plan_and_stops_api_failed`、`::test_workflow_judgment_failure_stops_api_failed_judgment`；`test_langgraph_workflow.py` 中 judgment/planner 失败两例 |
| D5=严格 hash | `test_phase5_controller.py::test_normalize_query_is_nfkc_casefold_and_whitespace_strict`、`::test_duplicate_query_hashing_sees_full_width_variants_as_duplicates`、`::test_duplicate_query_detection_normalizes_case_and_whitespace`；`test_phase5_planner.py::test_replanner_rejects_full_width_case_variant_of_previous_query` |
| D7=默认结构化失败/显式降级 | `test_phase5_planner.py::test_planner_fallback_is_disabled_by_default`、`::test_planner_fallback_enabled_returns_deterministic_marked_plan`、`::test_planner_fallback_uses_deterministic_plan_id_when_input_has_none`、`::test_planner_fallback_applies_to_api_errors_too`、`::test_planner_fallback_replan_preserves_parent_constraints_and_changes_query`、`::test_planner_fallback_without_retrieval_modules_raises_plan_fallback_error`、`::test_planner_retry_success_is_not_marked_degraded` |
| D8=min_judgments=3 | `test_phase5_controller.py::test_valid_judgment_count_excludes_unaligned_products`、`::test_quality_stop_requires_min_judgments_two_is_not_enough`、`::test_quality_stop_with_three_min_judgments_satisfies`、`::test_workflow_selection_requires_a_judgment_for_each_selected_product` |
| D3=LangGraph 正式运行时 | `test_phase5_controller.py::test_run_langgraph_fails_loudly_when_runtime_missing`、`::test_run_falls_back_to_python_and_records_it_without_langgraph`、`::test_python_and_langgraph_runtimes_produce_identical_states`、`::test_run_uses_the_langgraph_runtime_when_installed`；`test_langgraph_workflow.py` 全部 9 例 |

服务器证据（非伪造）：`/tmp/agr/venv` 中 `langgraph==1.2.24`、`langchain-core==1.7.7`
（`importlib.metadata.version` 实测）；集成测试断言 graph 是真实的
`langgraph.graph.state.CompiledStateGraph` 实例。项目副本位于 `/tmp/agr/project`
（测试用临时同步目录，不是产物存储）。

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

1. **全仓库 mypy**：2026-11-24 本任务收尾时 `python -m mypy` 在本工作区通过
   （66 个源文件，含 LangGraph adapter；新增 `langgraph` import 使用局部
   `# type: ignore[import-not-found]`，本地未安装该包时可检查）。其他 teammate 并行修改的
   `retrieval/tiger/*` 文件不在本任务范围，合并前仍应重跑。
2. **真实 API 的 429/5xx/网络超时**：仅用 MockTransport 验证；真实环境只验证了 200 成功路径与
   推理截断失败路径。
3. **真实 10 商品批量 Judgment**：只验证了 2 商品 smoke；批量成本/延迟需在 Phase 7 评估。
   若启用 `reasoning_effort=none`，成本预计接近 probe 的 236 completion tokens/2 商品。
4. **LangGraph**：已在 S4000 `/tmp/agr/venv` 真实安装（`langgraph==1.2.24`）并跑通 9 个
   graph 级集成测试；`run()` 在服务器上走真实 graph，260 个单测通过。仍未验证：真实
   DeepSeek + E5 + Reranker 依赖下的服务器端端到端 graph 运行（由 Lead 集成时完成），
   以及 LangGraph checkpointer/持久化恢复（Phase 5 未启用 checkpointer，CLARIFY 仍是挂起返回）。
5. **真实端到端链路**：未在本任务中串接真实 E5 + FAISS + Qwen3-Reranker + Judgment + Controller
   （由 Lead 在集成时完成）；本任务的 workflow 使用注入的 retrieve/rerank 依赖，单测已覆盖分支。
6. **Controller 阈值与预算**：`configs/controller.yaml` 的 `required_count/min_valid_candidates/
   min_judgments/max_unknown_attribute_ratio/预算值` 均未用验证集校准（`calibrated=false`）；
   实现只消费配置，未宣称已验证有效。
7. **多样性**：`QualityMetrics.diversity` 在 Phase 5 恒为 `None`；不能把 `min_diversity` 设为非空，
   否则 quality 永远无法满足。
8. **Catalog 元数据缺口**：`fusion.pool._metadata` 只输出 title/price/main_category/brand，
   Judgment 默认视图缺少 description/details；工作流保留 `candidates_from_pool` 注入点，
   Lead 集成时可用真实 catalog 补齐（D6 已提供 `JudgmentEvidenceBuilder`/`judge_ranked`，
   本任务未改 `controller/` 的注入点）。
9. **D7 降级计划的结构化标记**：`RecommendationPlan` 没有 `degraded`/`plan_source` 字段
   （本任务范围不含 `schemas/plan.py`），因此标记落在 `PlannerResult.degraded`、
   `plan.assumptions` 与 `planner.plan_degraded` trace 三处；若后续要在 Plan 契约内表达，
   需 ADR 修改 `schemas/plan.py` 并同步所有读取方。
10. **D5 语义边界**：严格 hash 只识别“归一化后完全相同”的 query；改写、同义词、词序变化
    都不算重复（有意为之，避免改变检索语义）。近似重复需要保存 query 文本，本阶段未做。
11. **D8 有效 Judgment 定义**：`valid_judgment_count` 只统计已通过 `JudgmentRecord` Schema 且
    `product_id` 与当前 `ranked_products` 对齐的记录；未覆盖“Judgment 内容错误但结构合法”的
    语义正确性问题（Phase 7 人工抽检）。

---

## 11. 关键复现命令（供 Lead 集成）

```bash
# 单元测试（无真实 API）
python -m pytest tests/unit/test_phase5_planner.py tests/unit/test_phase5_judgment.py \
                 tests/unit/test_phase5_controller.py -q

# D3：LangGraph 正式运行时（本地未安装时集成测试 skip；服务器真实执行）
python -m pip install -e ".[workflow]"        # 或 pip install langgraph langchain-core
python -m pytest tests/integration/test_langgraph_workflow.py -q

# 全量
python -m pytest -q
python -m ruff check src tests
python -m mypy

# 真实 smoke（需要项目根 .env 提供 DEEPSEEK_API_KEY；结果写到 docs/phase5/）
# 运行方式：一次性脚本（本任务未把临时脚本提交入库），
# 证据见 docs/phase5/deepseek_smoke_20261124T033749Z.json
```

集成建议顺序：`load_settings()` → `DeepSeekClient(settings=settings, max_tokens=...)` →
`Planner(..., fallback=settings.config.planner.fallback)` / `Judge` →
`Controller(config=settings.config.controller)` →
`RecommendationWorkflow(WorkflowDependencies(...), WorkflowOptions(...))`，
正式入口为 `workflow.run(state)`（安装 LangGraph 后必走 graph）；retrieve/rerank 由 Phase 2/3
服务注入；`BudgetState` 用 `Controller.initial_budget()` 初始化。

---

## 12. Phase 5 硬化（D3–D5, D7–D8，2026-11-24）

本节是 D3/D4/D5/D7/D8 的实施汇总；前文对应小节已同步更新，未删除任何既有契约。

### 12.1 决策 → 实现对照

| 决策 | 结论 | 实现位置 | 关键行为 |
|---|---|---|---|
| D3 | A：LangGraph 正式运行 | `controller/langgraph_adapter.py`（新）、`controller/workflow.py`、`controller/__init__.py` | `run()` 安装 langgraph 时必走真实 graph；`run_langgraph()` 缺依赖抛 `LangGraphUnavailableError`；`run_python()` 为业务核心与显式回退；graph 节点 = 既有 step 函数，条件边按 decision/action |
| D4 | C：新增 `api_failed` + `failed_component` | `schemas/common.py`、`schemas/controller.py`、`controller/controller.py`、`controller/workflow.py` | Planner/Judgment 重试耗尽 → `stop_reason=api_failed` + `ControllerDecision.failure`；Retrieval/Reranker 失败仍 `retrieval_degraded`；重试成功不 STOP；Generator 组件已预留 |
| D5 | B：严格 Hash | `controller/controller.py`、`planner/planner.py`、`config.py`、`configs/controller.yaml` | `normalize_query` = NFKC + 空白折叠 + casefold；SHA-256 相等才判重复；删除 `duplicate_query_similarity`；不排序/不去停用词 |
| D7 | C：可配置，默认结构化失败 | `config.py`、`configs/controller.yaml`、`planner/planner.py`、`workflow.py` | 默认 `PlanGenerationError`；显式 `enabled=true` 返回确定性 balanced 计划，`degraded=True`/`plan_source=system_default`/assumptions/trace 标记；无法构建时 `PlanFallbackError` |
| D8 | C：`min_judgments=3` | `config.py`、`configs/controller.yaml`、`schemas/controller.py`、`controller/controller.py`、`workflow.py` | `QualityMetrics.valid_judgment_count`（与 ranked 对齐、去重）；`valid_judgment_count < min_judgments` 时 quality_satisfied 恒 false；2 条不达标、3 条达标；未判断商品不进 Top-K |

### 12.2 契约与配置默认值

- `StopReason.API_FAILED = "api_failed"`；`FailedComponent` 只允许
  `planner`/`judgment`/`generator`；`FailureInfo` 的 `failed_component` 必填，其余可选。
- `ControllerDecision.failure` 仅与 `api_failed` 成对出现（validator 双向强制）；
  其他决策的 JSON 不新增字段，旧 payload 可反序列化。
- `QualityMetrics.valid_judgment_count` 默认 0（向后兼容直接构造 metrics 的旧测试/代码）。
- `controller.quality.min_judgments: 3`（未校准，工程起点，`calibrated: false`）。
- `planner.fallback.enabled: false`、`planner.fallback.strategy: balanced`（唯一支持值）。
- `controller.loop_guard.duplicate_query_similarity` 已移除（配置加载 `extra=forbid`，
  残留该键会显式报错而不是被忽略）。

### 12.3 运行时与编排

- `RecommendationWorkflow.run(state)`：正式入口。`use_langgraph=True` 默认；
  有 langgraph 时把 state 交给编译后的 `CompiledStateGraph`，无 langgraph 时 WARNING
  回退纯 Python（日志字段 `fallback=python_state_machine`）。
- `RecommendationWorkflow.run_langgraph(state)`：强制正式运行时，缺少依赖直接
  `LangGraphUnavailableError`（不静默）。
- `RecommendationWorkflow.run_python(state)`：纯 Python 状态机，与 graph 共享同一组
  `step_*` 函数，语义由测试
  `test_python_and_langgraph_runtimes_produce_identical_states` 与 S4000 上 260 个单测共同锁定。
- LangGraph 的 recursion_limit 由 `loop_guard.max_rounds` 推导（`max(50, 12*(max_rounds+2))`），
  防止配置外的无限循环；业务层预算/轮数仍是第一道保护。

### 12.4 改动范围与验证状态

- 仅修改任务范围：`src/recommender/controller/`、`src/recommender/planner/`、
  `src/recommender/schemas/common.py`、`src/recommender/schemas/controller.py`、
  `src/recommender/config.py`、`configs/controller.yaml`、`tests/unit/test_phase5_controller.py`、
  `tests/unit/test_phase5_planner.py`、`tests/integration/test_langgraph_workflow.py`、
  `docs/phase5/`。未修改 `cli.py`、`docs/audit`、`docs/decisions`、`retrieval/tiger`、
  `configs/tiger`、`judgment/`（D6 由另一成员完成）。
- 服务器证据：`/tmp/agr/venv`（langgraph 1.2.24）上
  `tests/unit` 260 passed / 2 skipped，`tests/integration/test_langgraph_workflow.py` 9 passed；
  本地（无 langgraph）260 passed / 14 skipped；`ruff check` 与 `mypy` 通过。
- 未验证项见 §10（真实端到端 graph 链路、checkpointer、阈值校准、Plan schema 内的 degraded 字段、
  Judgment 语义正确性抽检）。
