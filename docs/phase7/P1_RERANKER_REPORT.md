# P1 Reranker 完整评测报告

- 日期：2026-11-25
- 数据：frozen v1.0，2,224 条最终样本
- 基线：`e2e-baseline-v1`
- 实验：E5-only（Top-100） vs E5 + Qwen3-Reranker（E5 Top-10 重排）
- 结果文件：[reranker_p1_top10.json](v1.0/reranker_p1_top10.json)（原始：[raw](v1.0/reranker_p1_top10_raw.json)）
- 状态：**已完成；Reranker 未显示统计显著增益**

## 1. 实验设置

| 项 | 值 |
|---|---|
| 样本 | 2,224（v1.0 final，人工 PASS 子集） |
| 查询 | 冻结 synthetic Query（target-isolated） |
| E5 | `intfloat/e5-base-v2`，FAISS Top-100 |
| Reranker | `Qwen/Qwen3-Reranker-0.6B`，CPU，batch 64，max_length 512 |
| Rerank 候选 | E5 Top-10 |
| 指标 | Recall@10、NDCG@10、MRR；用户 cluster bootstrap (1,000) |
| 运行 | 本机 CPU；约 48 分钟（2,224 × 10 候选） |
| 延迟 | E5 8.3 ms/样本；Reranker 1,306 ms/样本（top-10） |

> 说明：Reranker 只看 E5 Top-10，因此 **Recall@50/100 不可比**；JSON 中这两个字段置 null，
> 只报告 E5 full Top-100 的这一指标。

## 2. 结果

| 方法 | Recall@10 | Recall@50 | Recall@100 | NDCG@10 | MRR |
|---|---:|---:|---:|---:|---:|
| E5-only（Top-100） | **1.98%** | **3.33%** | **3.60%** | 0.0121 | 0.0104 |
| E5 + Reranker（Top-10 重排） | 1.98% | — | — | **0.0127** | 0.0104 |

用户级 cluster bootstrap（Reranker − E5，1,301 个用户簇）：

| 指标 | Delta | 95% CI | Bootstrap P(delta>0) |
|---|---:|---|---:|
| NDCG@10 | +0.0006 | [-0.0008, +0.0019] | 0.783 |
| MRR | -0.0000 | [-0.0019, +0.0018] | 0.467 |
| Recall@10 | 0.0000 | [0, 0] | 0 |

## 3. 解释

1. **Recall@10 不变是结构性的**：Reranker 只重排 E5 Top-10，无法把 Top-10 之外的
   商品带入；目标如果在 E5 Top-10 内，重排不会改变是否命中；如果在外面，也无法召回。
2. **NDCG@10 略升但不显著**：+0.0006，95% CI 跨 0；bootstrap 78.3% 重采样为正，
   不能声称 Reranker 带来可靠排序增益。
3. **MRR 无变化**：目标商品的平均排名位置基本未变。
4. **成本显著上升**：Reranker 1,306 ms/样本，比 E5 8.3 ms 高约 157 倍；
   在 CPU 上不是“免费”的改进。
5. **任务口径限制**：离线标签是“下一次评论/评分商品”，不是“Query 相关性”；
   交叉编码器可能提升语义相关性，但未必提升下一商品命中。评测结论只适用于当前数据/目标。

## 4. 结论

- **P1 的当前配置（Top-10 重排）不支持在默认 E2E 中加入 Reranker**：无显著 Recall/NDCG/MRR 增益，
  延迟成本高；
- 保留 Reranker 为可选阶段（Phase 3 / 诊断），不改变 E5-only 默认路径；
- 若继续 P1，下一步应做 **Top-50 / Top-100 候选预算**与 **GPU/并行**，否则与协议 §10 的
  “统一候选预算”不可比；本报告的 Top-10 是受 CPU 预算限制的工程版。

## 5. 未验证 / 风险

- Reranker 全量 Top-100 未跑（CPU 估算 >10h）；
- 未做相关性人工标注，只使用下一商品标签；
- 未做 query-level 多样性/位置偏差分析；
- 延迟是单进程 CPU，batch 64；未做多进程/GPU。

## 6. 复现

```bash
python scripts/evaluate_reranker_p1.py \
  --pilot data/evaluation/v1.0 \
  --bundle artifacts/faiss/e5-base-v2_all_beauty_v2 \
  --catalog artifacts/catalog/all_beauty_v2 \
  --top-n 10 --batch-size 64 --max-length 512 \
  --out data/evaluation/v1.0/reports/reranker_p1_top10.json
```

> 运行成本：2,224 样本 × 10 候选；建议在服务器后台运行并保留 checkpoint。
