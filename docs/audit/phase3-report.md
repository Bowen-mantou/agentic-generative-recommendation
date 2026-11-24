# Phase 3 执行记录：Qwen3-Reranker-0.6B

- 日期：2026-11-24
- 状态：**Phase 3 完成**
- 依据：决策 Q2（Transformers 原生加载、CPU 基线）、AGENTS.md §3（Reranker 只排序）
- 证据：[reranker-smoke.md](reranker-smoke.md)（实现正确性）· [evidence/rerank_demo_v2.json](evidence/rerank_demo_v2.json)（真实候选池精排）

## 1. 本次完成

| 能力 | 实现 |
|---|---|
| Transformers 原生打分 | `reranking/qwen.py`：`Qwen3ForCausalLM` + tokenizer，官方 prefix/suffix、yes=9693 / no=2152、fp32、`P(yes)` 打分 |
| 批量与截断 | `batch_size=4`、`max_length=1024`（配置）；按 batch 前向，记录总延迟 |
| 候选筛选 | 只精排 eligible（`pass` 或无约束）；`unknown` / `fail` 不进入精排 |
| 文档构造 | 复用 catalog `rerank_text`（`product_text.rerank.v1`）；缺少文本显式跳过并告警，不补写属性 |
| 精排服务 | `reranking/service.py`：稳定排序（分数降序、同分按 `product_id`）、保留 fusion 排名/来源、`top_k` |
| CLI | `agr rerank --bundle --catalog --query ... --constraint ... --top-k` |
| 契约 | 复用 `RerankResult`；`score_interpretation` 明确“相关性分数、非购买概率” |

## 2. 接口变化

- 新增 `PairScorer` 协议（`model_id`、`batch_size`、`score_pairs`），服务可注入 stub 测试；
- 新增 `RerankService.rerank(query, candidates, top_k=None) -> RerankResult`；
- 配置：`reranker.max_length: 1024`、`batch_size: 4`、`device: cpu`（依据冒烟实测）。

## 3. 验证结果

### 3.1 测试

| 环境 | 命令 | 结果 |
|---|---|---|
| 本地 | `python -m pytest` | **142 passed, 3 skipped**（3 个需真实模型/索引的集成测试跳过） |
| 服务器（默认） | `python -m pytest` | **144 passed, 1 skipped**（跳过需显式开启的真实 Reranker 测试） |
| 服务器（真实模型） | `AGR_RUN_RERANKER=1 python -m pytest tests/integration/test_reranker_real.py` | **1 passed in 9.98 s** |
| 本地 | `ruff check` / `mypy` | 通过 / Success（52 source files） |

单测覆盖：排序与排名、`top_k`、fusion 排名/来源保留、缺失文本跳过、空候选/空文本报错、分数数量校验、eligible 过滤。

### 3.2 真实候选池精排（All_Beauty）

命令：

```bash
agr rerank --bundle .../faiss/e5-base-v2_all_beauty_v2 \
  --catalog .../catalog/all_beauty_v2 \
  --query "lightweight daily moisturizer for sensitive skin" \
  --query "fragrance free face cream under 30 dollars" \
  --constraint "price:lte:30:USD" --top-k 5
```

| 指标 | 实测 |
|---|---|
| 候选池 | 99 evaluated → **15 eligible** / 76 unknown / 8 fail |
| 精排候选 | 15（全部有 catalog 文本，missing_text=0） |
| 模型 / 批量 / 截断 | `Qwen/Qwen3-Reranker-0.6B` / 4 / 1024 |
| 精排延迟 | **10.29 s**（15 条，CPU，batch=4；约 0.69 s/条） |
| CLI 端到端 | 32.5 s（含 E5 检索、模型加载、精排） |

Top-5（真实商品，分数为 P(yes)，只用于同一 Query 内排序）：

| rerank rank | product_id | score | fusion rank | 商品 |
|---|---|---|---|---|
| 1 | `B013THQAAO` | 0.996291 | 4 | Remedy Phytoplex Botanical Nourishing Skin Cream Moisturizer |
| 2 | `B00UCKQ796` | 0.992262 | 15 | Ultra Balm Daily Moisturizer For Dry Itchy Skin |
| 3 | `B08KH886NY` | 0.987371 | 11 | canBE naturally i can be radiant DAILY MOISTURIZER |
| 4 | `B087QVX3JL` | 0.925871 | 9 | C'est Moi Tinted Moisturizing Lotion SPF 30 |
| 5 | `B0009ET4SG` | 0.922555 | 2 | Beauty without Cruelty Maximum Moisture Cream |

观察：精排确实改变了融合顺序（原 fusion #15 → rerank #2；fusion #2 → rerank #5），
且 Top-5 全部为 `price` 已验证通过的保湿类商品。

## 4. 明确限制

1. 分数饱和（相关 ≥0.92，负例接近 0），**不能当概率或跨 Query 阈值**；阈值校准需要标注数据（R9）。
2. 尚未接入 `bge-reranker-v2-m3` 对照（Phase 7 对照实验）。
3. 精排依赖 catalog 文本；价格/描述缺失会影响文本质量（R8）。
4. 当前只对 E5 单路候选池精排；TIGER 接入后需回归双路候选池。
5. 延迟随候选数与文本长度增长；20 候选 512 token 约 6 s（冒烟报告），本场景 15 候选 1024 token 约 10.3 s。

## 5. 下一步

- Phase 5：DeepSeek Planner / Judgment、LangGraph Controller（Q12 要求）。
- Phase 6：Generator、Mem0、反馈更新。
- Phase 4：等待 Q1 解释确认（外部权重已证伪，见 [tiger-feasibility-assessment.md](tiger-feasibility-assessment.md)）。
