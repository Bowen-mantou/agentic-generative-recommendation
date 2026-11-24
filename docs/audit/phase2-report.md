# Phase 2 执行记录：候选融合与硬约束

- 日期：2026-11-24
- 状态：**Phase 2 完成**
- 依据：AGENTS.md §2.4/§2.5、模块三 §5、决策 Q3（约束 Schema）、Q12（第一阶段 = Phase 0–6）
- 证据：[pool_demo_v2.json](evidence/pool_demo_v2.json) · [risk-status.md](risk-status.md)

## 1. 本次完成

| 能力 | 实现 |
|---|---|
| 统一候选协议输入 | 复用 `RecallResult`（`product_id` 主键、来源、排名、原始分数） |
| 跨路径 RRF | `fusion/fuse.py`：`fuse_paths()`，只按排名融合，不累加异构原始分数 |
| 去重与来源保留 | 每个商品保留 `retrieval_sources`、`source_ranks`、`raw_scores` |
| 路径失败处理 | 单路失败 → 标记 `degraded_paths` 并继续；全部失败 → `RetrievalError`（不伪装空候选） |
| 唯一硬约束 Evaluator | `constraints/evaluator.py`：price/category/brand/rating/title/details 属性，三态 + 原因 |
| 候选池 | `fusion/pool.py`：eligible / unverified / rejected 三分区 + `ConstraintSummary` |
| Q3 Schema 落地 | 无硬约束时 `constraint_checks=[]`、整体状态 `None`、`not_configured_count` 单独统计 |
| CLI | `agr pool --bundle --catalog --query ... --constraint field:op:value[:unit]` |

## 2. 接口变化

| 契约 | 变化 |
|---|---|
| `ConstraintEvaluation.status` | 改为 `ConstraintStatus \| None`；无 check 时为 `None`，不得伪装 `pass`（Q3） |
| `ConstraintSummary` | 新增 `not_configured_count`；`valid_product_count = pass + not_configured` |
| `CandidatePool` | `candidates` 允许 `constraint_status=None`（仅当无约束）；分区与 summary 一致性强制校验 |
| 新增 | `FusionOutcome(candidates, degraded_paths, dropped_by_limit, used_paths)`（服务层 dataclass） |
| 新增 CLI | `agr pool` |

## 3. 验证结果

### 3.1 测试

| 环境 | 命令 | 结果 |
|---|---|---|
| 本地 | `python -m pytest` | **136 passed, 2 skipped**（2 个真实 bundle 集成测试在无 faiss/索引的机器上跳过） |
| 服务器 | `python -m pytest` | **138 passed**（含真实 E5 bundle + 候选池集成测试） |
| 本地 | `ruff check` / `mypy` | 通过 / Success（50 source files） |

覆盖点：两级 RRF 顺序与同分策略、去重、来源/排名/分数保留、路径降级与全失败、候选上限；
约束的 pass/fail/unknown（缺字段、货币不符、非数值比较、in/not_in/contains、details 属性）；
无约束空状态；候选池三分区与汇总一致性。

### 3.2 真实数据演示（All_Beauty，E5 单路 + 两条硬约束）

命令：

```bash
agr pool --bundle /mnt/agr-data/artifacts/faiss/e5-base-v2_all_beauty_v2 \
  --catalog /mnt/agr-data/artifacts/catalog/all_beauty_v2 \
  --query "lightweight daily moisturizer for sensitive skin" \
  --query "fragrance free face cream under 30 dollars" \
  --constraint "price:lte:30:USD" --constraint "main_category:eq:All Beauty" --top-k 6
```

| 指标 | 实测值 |
|---|---|
| 融合候选 | 99（2 Query × 20 去重后） |
| eligible（全部 pass） | **15** |
| unverified（价格/字段缺失 → unknown） | **75** |
| rejected（违反约束） | **9** |
| 有效候选数 | 15（`pass + not_configured`，Q3） |
| 端到端延迟 | 23.5 s（含模型加载冷启动；检索批次 19.1 s 含首次推理预热） |
| Top eligible 示例 | `B01L0JWQTW`、`B0009ET4SG`、`B0016N55BE`、`B013THQAAO`（价格与品类均 pass） |

**与决策 Q5 的一致性**：price 覆盖率约 15.7%，因此 99 个候选中 75 个只能得到 `unknown`；
这证明 evaluator 没有把缺失价格当作满足预算，符合 Q3/Q5。

## 4. 明确限制

1. **真实双路融合尚未运行**：TIGER 端到端权重不存在（R1），Q1 等待负责人确认；
   本阶段的双路融合只在单元测试中用合成的 TIGER/E5 排名验证协议与降级逻辑。
   Phase 2 的“两路结果可以合并”在协议层成立，真实 TIGER 合并需等 Phase 4。
2. 候选池尚未接 Reranker（Phase 3）。
3. `unknown` 比例高是数据事实（price/description 缺失），不是实现缺陷；评测需报告
   可验证/未知比例（R8）。

## 5. 下一步

1. Phase 3：按 Q2 的已验证配方（Transformers 原生、yes=9693/no=2152、CPU）实现
   `reranking/`，对候选池精排并记录分数/排名/延迟。
2. Phase 5/6 可在 E5 单路候选池上继续（Planner/Judgment/Controller/Generator/Mem0），
   TIGER 接入后回归双路测试；若 Q1 最终不批准自建，按 Q12 提交偏差说明。
