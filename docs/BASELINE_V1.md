# E2E Baseline Freeze V1

- 冻结时间：2026-11-25 07:05（Asia/Shanghai）
- 用途：固定已经跑通的 E5-only E2E 主线，后续优化不得静默破坏；P1 Reranker 实验以此为对照基线。
- Git commit：`a1c987f`（tag `e2e-baseline-v1`）
- GitHub Actions：run `45446919385`，Python 3.10/3.11 **success**
- Python：3.10（服务器 `/tmp/agr/venv`）；本地测试 3.10 / CI 3.10+3.11

## 1. 冻结的代码与入口

```text
agr recommend  (真实 E5-only E2E)
agr feedback   (显式反馈写入 Mem0)
```

入口文件：

- `src/recommender/e2e/service.py`
- `src/recommender/cli.py`
- `src/recommender/memory/{adapter,mem0_adapter}.py`
- `src/recommender/generation/deepseek_generator.py`

## 2. 配置哈希（SHA-256 前 16 位）

| 文件 | 哈希 |
|---|---|
| `configs/base.yaml` | `17b99e60866c8b21` |
| `configs/controller.yaml` | `931aa0e57268fcd4` |
| `configs/evaluation.yaml` | `f56f5e59c1388fc4` |
| `configs/memory.yaml` | `2f443bc1d231f66d` |
| `configs/models.yaml` | `79619716e5e8f3a8` |
| `configs/retrieval.yaml` | `bd96087fc14347e1` |

## 3. Prompt / 模型版本

| 组件 | 版本 / 模型 |
|---|---|
| Planner initial | `planner.initial.v1` |
| Planner replan | `planner.replan.v1` |
| Judgment | `judgment.v1` |
| Generator | `recommendation_generator_v1` |
| LLM | `deepseek-flash`，temperature 0.2，JSON mode，`reasoning_effort=none` |
| E5 商品召回 | `intfloat/e5-base-v2`，768 维 |
| Memory Embedding | `intfloat/multilingual-e5-base`，revision `d1287505…` |
| Reranker | `Qwen/Qwen3-Reranker-0.6B`，Transformers 原生 CPU |
| LangGraph | 1.2.x（服务器 `/tmp/agr/venv`） |

## 4. 数据资产

| 项 | 版本/路径 |
|---|---|
| Catalog | `all_beauty_v2`（112,590 商品） |
| E5 bundle | `e5-base-v2_all_beauty_v2` |
| 评测数据集 | `v1.0` frozen，2,224 条最终样本 |
| Manifest | `docs/phase7/v1.0/evaluation_manifest_final.json`，SHA-256 `5696a859b6dea4a3…` |
| Target 泄漏 | 0（2,551 条 Query 扫描） |

## 5. 冻结的 E2E Trace（docs/e2e/）

| 场景 | request_id | trace_id | 结果 |
|---|---|---|---|
| stop latest | `req_e84061a4ed904d3dbc0582f53efa2d57` | `tr_1ad4ef4209904be0ab0dacb29aa3b7fc` | quality_satisfied，5 推荐，66.5 s |
| stop previous | `req_4b45db55ae0442fb85ec4d1597a00853` | `tr_19621ca3c1e148d4b8f8d39dfc164f4a` | quality_satisfied，5 推荐，57.9 s |
| stop initial | `req_2c9b41667a62441eb56a9658ecc4eaba` | `tr_0c413ae7749b4365932222d58329f6f5` | quality_satisfied，5 推荐，61.6 s |
| replan | `req_e88fd89546f14a608ff46b5b0919a5e8` | `tr_af9ec4fc00584a8e856da8f25db4076d` | REPLAN → budget_exhausted，0 推荐，25.3 s |
| feedback | — | — | add + readback 1，isolation OK，14.1 s |

完整 trace 在 NFS：`/mnt/agr-data/runs/e2e/<request_id>/`；仓库内为脱敏 summary + trace。

## 6. 冻结的测试证据

- 本地：`308 passed / 15 skipped`；
- 真实 E2E：服务器 `AGR_RUN_E2E=1 pytest tests/integration/test_e2e_real.py` → `1 passed`（35.9 s）；
- ruff：通过；mypy：80 source files 无错误；
- CI：Python 3.10/3.11 success（run `45446919385`）。

## 7. 不允许的回归

- 不得删除或跳过现有测试来换取绿色 CI；
- 不得把 `fallback_template` 的 Generator 输出算作真实 Generator 通过；
- 不得在未启用 Memory 时声称 Memory Gate 通过；
- 不得重新引入 TIGER 为默认召回或宣称双路召回完成；
- 不得改动冻结评测集（2,224 条）而不新增版本。

## 8. 复现

```bash
git checkout e2e-baseline-v1
agr recommend --query "fragrance-free moisturizer for sensitive skin" \
  --user-id demo_user_002 --catalog ... --bundle ...
AGR_RUN_E2E=1 pytest tests/integration/test_e2e_real.py -q
```

> 该基线是 **E5-only** 工程主线，不是 TIGER + E5 双路召回验收。
