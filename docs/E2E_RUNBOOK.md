# E2E Runbook — `agr recommend`

本 runbook 对应 24 小时方案的真实 E5-only 主线。命令在项目根目录执行；数据/索引/模型在 NFS，不提交 Git。

## 1. 前置条件

| 项 | 要求 |
|---|---|
| Python 环境 | 项目 venv，安装 `.[retrieval,workflow,memory]`（服务器 `/tmp/agr/venv` 已装；重启用 `scripts/bootstrap_s4000.sh` + 手动补 workflow/memory） |
| DeepSeek | 环境变量 `DEEPSEEK_API_KEY`（或 `.env`，gitignored） |
| E5 bundle | `artifacts/faiss/e5-base-v2_all_beauty_v2/`（manifest/index/embeddings/product_ids） |
| Catalog | `artifacts/catalog/all_beauty_v2/`（products.jsonl、texts.jsonl） |
| Qdrant | `agr-qdrant` 容器，127.0.0.1:6333，collection `agr_user_memory` 可写 |
| HF cache | `HF_HOME=/mnt/agr-data/hf`（真实路径按部署填），包含 e5-base-v2 / multilingual-e5-base / Qwen3-Reranker |
| LangGraph | 项目 venv 已安装（`langgraph 1.2.x`） |

服务器示例：

```bash
cd ~/Agentic-Generative-Recommendation
source scripts/env_s4000.sh
export HF_HOME=/mnt/his_test/group11/agr/hf HF_ENDPOINT=https://hf-mirror.com
```

## 2. 健康检查

```bash
agr config validate
curl -s http://127.0.0.1:6333/collections
python -c "import langgraph, mem0, qdrant_client; print('deps ok')"
```

## 3. 运行推荐

```bash
agr recommend \
  --query "fragrance-free moisturizer for sensitive skin" \
  --user-id demo_user_002 \
  --catalog /mnt/agr-data/artifacts/catalog/all_beauty_v2 \
  --bundle /mnt/agr-data/artifacts/faiss/e5-base-v2_all_beauty_v2 \
  --trace-dir /mnt/agr-data/runs/e2e \
  --top-k 5 --judgment-top-n 10 --rerank-top-n 50
```

- 默认启用 Memory；加 `--no-memory` 可显式关闭并记录 `backend=disabled`；
- 输出是 summary JSON；同时生成 `runs/e2e/<request_id>/trace.jsonl` 和 `summary.json`；
- 退出码 0 表示执行完成，不代表质量一定达标；看 `controller.stop_reason`。

### 预期结果

- `memory.health.available=true`（Memory 已配置时）；
- `retrieval.eligible > 0`；
- `controller.action=stop`，常见 `stop_reason=quality_satisfied` 或 `budget_exhausted`；
- `recommendation.recommendations` 的商品全部来自 `controller.selected_products`；
- `generator_model=deepseek-flash`（不是 `fallback_template`）。

### 无候选/REPLAN 示例

```bash
agr recommend \
  --query "fragrance-free moisturizer under \$1" \
  --user-id demo_user_003 \
  --catalog ... --bundle ... --trace-dir ...
```

预期：Controller 第一轮 `replan`，真实 Replanner 生成新 Query，第二轮可能
`budget_exhausted` STOP；不得伪造商品。

## 4. 写入显式反馈

```bash
agr feedback \
  --user-id demo_user_002 \
  --product-id B07N5B8WWR \
  --feedback positive \
  --note "Liked the fragrance-free formula"
```

输出应包含：

- `written.memory_id`；
- `read_back_same_user >= 1`；
- `other_user_hits = 0`；
- `user_isolation_ok = true`。

模拟反馈（仅测试用户）：

```bash
agr feedback --user-id e2e_simulated_user --product-id B07N5B8WWR \
  --feedback negative --simulated --note "test only"
```

`is_simulated=true` 会写入 metadata，不能混入真实用户记忆。

## 5. 查看 trace

```bash
jq . runs/e2e/<request_id>/summary.json
jq -c '{event, payload}' runs/e2e/<request_id>/trace.jsonl | head
```

关键事件：

- `e2e.started` / `e2e.memory_read`；
- `workflow.started` / `workflow.retrieval_completed` / `workflow.rerank_completed`；
- `workflow.judgment_completed` / `workflow.finished`；
- `e2e.evidence_built` / `e2e.generator_completed` / `e2e.completed`。

## 6. 常见故障

| 现象 | 处理 |
|---|---|
| `DEEPSEEK_API_KEY` 缺失 | 配置环境变量/`.env`；不得用 Mock 冒充 E2E |
| Memory health `available=false` | 检查 Qdrant 容器、mem0ai/qdrant-client、`HF_HOME`；E2E 仍可 `--no-memory` 但 G1 不完整 |
| `no eligible candidates` | 检查约束是否过严、E5 bundle/catalog 版本；允许 REPLAN，不填随机商品 |
| Reranker OOM/慢 | 降低 `--rerank-top-n`（默认 50），记录预算变化；不得静默跳过 |
| Generator `fallback_template` | 查看 API/JSON 错误；fallback 不算 G5 真实通过 |
| LangGraph 缺失 | 安装 `.[workflow]`；正式 E2E 必须走 LangGraph |
| Qdrant 不可用 | Memory Gate 不通过；修复后重跑，不用空列表假装成功 |

## 7. 测试

```bash
# 单元/集成（不调用真实 API/模型）
pytest tests/unit -q

# 真实 E2E（需要数据、模型、Qdrant、API key）
AGR_RUN_E2E=1 \
AGR_E2E_CATALOG=/mnt/agr-data/artifacts/catalog/all_beauty_v2 \
AGR_E2E_BUNDLE=/mnt/agr-data/artifacts/faiss/e5-base-v2_all_beauty_v2 \
HF_HOME=/mnt/agr-data/hf \
pytest tests/integration/test_e2e_real.py -q
```

## 8. 清理测试数据

- E2E 使用 `demo_user_002`（已有 1 条 positive feedback）和 `demo_user_003`；
- 如需清理 Qdrant collection，先确认没有其他用户数据；不得删除真实用户 collection；
- Trace/summary 在 NFS，可保留作为验收证据；不要把大 trace 提交 Git。
