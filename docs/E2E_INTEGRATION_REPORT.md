# E2E Integration Report — `agr recommend`

- 日期：2026-11-25 01:15（Asia/Shanghai）
- 基线：24 小时方案 V1.0（`docs/decisions/...` 外部文档） + commit `a8717b7`/`4f6efbb`
- 状态：**E5-only 主线的真实 E2E G0–G6 已跑通**；TIGER 仍为 D11 No-Go，不代表双路召回完成
- 证据目录：`docs/e2e/`（summary + trace，已脱敏）；完整 trace 在 NFS `runs/e2e/`

## 1. 结论

统一命令：

```bash
agr recommend --query "fragrance-free moisturizer for sensitive skin" \
  --user-id demo_user_002 \
  --catalog <catalog_dir> --bundle <e5_bundle>
```

真实执行了：

```text
Mem0 Read → DeepSeek Planner → E5/FAISS → Fusion → Hard Constraints
→ Qwen3-Reranker → D6 Evidence Builder → DeepSeek Judgment
→ LangGraph Controller → STOP / REPLAN → DeepSeek Generator
```

- 3 次完整真实运行（同一请求）：**57.9 s / 61.9 s / 66.5 s**；
- 至少一次自然 **quality_satisfied STOP**：5 个商品，10 条有效 Judgment，117 eligible（最新一次）；
- 至少一次自然 **REPLAN**：`$1` 预算约束触发第 1 轮 REPLAN，真实 Replanner 生成新 Query，
  第二轮后 `budget_exhausted` STOP，无商品被强行推荐；
- Memory：新用户空记忆可继续；显式 feedback 写入后同一用户读到 1 条，另一用户 0 条；
- Generator：真实 `deepseek-flash`，输出商品全部来自 Controller 选择，reasons 引用证据；
- Trace：每次请求有 `trace.jsonl` + `summary.json`，记录请求/轨迹 ID、候选、Reranker 分数、
  Judgment、Controller 决策、StopReason、Generator 结果、耗时和 token（可获得处）。

## 2. Gate 结果

| Gate | 结果 | 证据 |
|---|---|---|
| G0 统一 CLI | ✅ | `agr recommend` / `agr feedback`；退出码 0；trace 路径可追踪 |
| G1 真实 Planner + Memory Read | ✅ | Memory Read 在 Planner 前；空记忆/1 条记忆均可；DeepSeek `plan_initial` 通过 Schema |
| G2 真实 Retrieval + Reranker | ✅ | E5/FAISS 真实 78/117 eligible；Fusion/约束；Qwen3 50 候选真实评分 |
| G3 Evidence + Judgment | ✅ | D6 builder 10 候选；10 条真实 DeepSeek Judgment；缺失字段 UNKNOWN |
| G4 LangGraph Controller | ✅ | 真实 LangGraph 运行；quality STOP；自然 REPLAN + 二次检索 + budget STOP |
| G5 Generator + Memory Adapter | ✅ | 真实 Generator 5 条；商品 ID ⊆ selected；feedback add/read/隔离通过 |
| G6 E2E 证据 | ✅ | 3 次完整运行 + REPLAN + feedback trace；token/延迟在 trace/log |

## 3. 真实运行记录

| 运行 | 用户 | 记忆命中 | eligible | 结果 | 耗时 |
|---|---|---|---:|---|---:|
| `run_stop_initial` | `demo_user_002`（新用户） | 0 | 78 | quality_satisfied，5 推荐 | 61.6 s |
| `run_stop_previous` | `demo_user_002`（feedback 后） | 1 | 117 | quality_satisfied，5 推荐 | 57.9 s |
| `run_stop_latest` | `demo_user_002`（feedback 后） | 1 | 117 | quality_satisfied，5 推荐 | 66.5 s |
| `run_replan` | `demo_user_003` | 0 | 0 eligible / 149 unknown / 38 fail | REPLAN 后 budget_exhausted，0 推荐 | 25.3 s |

说明：

- `run_stop_initial` 是语言修复前的运行，Generator 返回中文 reasons（已记录为缺陷）；
- 语言修复后（commit `4f6efbb` 内 Generator prompt）英文请求返回英文 reasons；
- `run_replan` 中 Generator 因无候选返回 `fallback_template`，**没有伪造商品**；
  该 fallback 不计入 G5 真实 Generator 通过。

### 3.1 最新 quality STOP 摘要

- Memory：Mem0 1 hit（`demo_user_002` 的显式 positive feedback）；
- Planner：DeepSeek `deepseek-flash`，2 条 E5 Query，0 硬约束；
- Retrieval：E5 融合 117 eligible；Qwen3-Reranker 对 50 个候选评分；
- Judgment：10 个候选、10 条有效记录（`match`/`uncertainty` 已写入 summary）；
- Controller：`stop / quality_satisfied`，`selected_products` 5 个；
- Generator：`deepseek-flash`，5 条推荐，商品 ID 全部来自 selected；reasons 为英文；
- 总耗时 66.5 s，其中 Memory 8.9 s、workflow 46.5 s、Generator 1.9 s（详细见 summary）。

### 3.2 REPLAN 摘要

- Planner 第一轮识别 `price <= 1` 硬约束；
- 187 个候选经约束后 0 eligible、149 unknown、38 fail；
- Controller 返回 `replan`，Replanner 生成 4 条新 E5 Query；
- 第二轮仍 0 eligible，预算耗尽 `budget_exhausted` STOP；
- 无随机/伪造商品，Generator fallback 输出 0 条。

## 4. Memory 验证

- `agr feedback --user-id demo_user_002 --product-id B07N5B8WWR --feedback positive`
  写入成功，`read_back_same_user=1`、`other_user_hits=0`、`user_isolation_ok=true`；
- Qdrant collection `agr_user_memory`，768 维 Cosine，multilingual-e5-base；
- Mem0 `infer=False`（显式反馈不做 LLM 抽取），`is_synthetic` 落 payload；
- 真实 E2E 的 Memory Read 在 Planner 之前执行，空记忆是合法状态；
- 测试用户 `demo_user_002` 的反馈仍保留在 collection 中（E2E 证据），不是生产数据。

## 5. 测试与回归

| 项 | 结果 |
|---|---|
| 单元/集成测试 | 本地 **308 passed / 15 skipped**（E2E 真实测试按 env 跳过） |
| 真实 E2E 测试 | 服务器 `AGR_RUN_E2E=1 pytest tests/integration/test_e2e_real.py` → **1 passed**（35.9 s） |
| ruff / mypy | 全绿（80 source files） |
| CI | GitHub Actions Python 3.10/3.11（提交后运行） |
| 真实 API | Planner/Judgment/Generator 均 `deepseek-flash`，JSON mode，reasoning_effort=none |

## 6. 已知边界与未完成

| 项 | 状态 |
|---|---|
| TIGER | D11 No-Go；本轮 E5-only，**不宣称双路召回完成** |
| Q12 偏差 | 需课程/老师单独确认；本轮只完成 E5-only 工程主线 |
| Controller 阈值 | `calibrated=false`；`min_judgments=3` 等是工程初值 |
| 全量 Reranker / Cluster Bootstrap | 未做；2,224 条最终评测集只有 Recall 阶段 |
| Mem0 `infer=True` 抽取 | 未验证；Adapter 固定 infer=False |
| BM25/混合检索 | fastembed 未装，dense-only |
| Qdrant 鉴权 | 仅回环、无鉴权/TLS |
| API 成本硬上限 | 单请求 Controller 预算 20 次 API；未设置 provider 级日预算 |
| 并发/多进程 | 未压测 |
| 演示用户记忆 | `demo_user_002` 测试反馈保留，未清理 |

## 7. 证据文件

- `docs/e2e/run_stop_latest_summary.json` / `run_stop_latest_trace.jsonl`
- `docs/e2e/run_stop_previous_*`、`run_stop_initial_*`
- `docs/e2e/run_replan_summary.json` / `run_replan_trace.jsonl`
- `docs/e2e/feedback_writeback.json`
- 完整 NFS 目录：`/mnt/agr-data/runs/e2e/<request_id>/`

## 8. 结论

- **G0–G6 的 E5-only 真实 E2E 已通过**，可以演示“一个真实请求从记忆到生成”的完整链路；
- 它不是双路召回验收，也不是 TIGER 成功；TIGER 的 D11 No-Go 与 E5-only 偏差必须在答辩/汇报中明确；
- Phase 7 的 2,224 条全量 Reranker、Cluster Bootstrap 和 Controller 校准仍是后续工作。
