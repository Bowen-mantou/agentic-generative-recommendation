# Evaluation Dataset Pilot Report (v0.1)

- 日期：2026-11-24
- 依据：[Evaluation Dataset Construction Protocol V1.0](EVALUATION_DATASET_CONSTRUCTION_PROTOCOL_V1.0.md)（本项目实施规范）
- 状态：**Pilot 完成（协议阶段 A–D）；E 未冻结；F 仅完成召回阶段三路对照**
- 数据目录（服务器 NFS）：`/mnt/his_test/group11/agr/evaluation/v0.1-pilot`
- 代码：`src/recommender/evaluation/`、`scripts/retrieval_comparison_pilot.py`

> 离线标签是下一次评论/评分商品，不代表真实购买转化。Query 由 DeepSeek PFlash
> 根据 Python 规则画像生成并独立审核，属于 synthetic query；人工审核尚未完成。

## 1. 结论摘要

| 项 | 结果 |
|---|---|
| 时间截点恢复（现有 2,685 条 TIGER test） | **2,685 / 2,685（100%）**，无失败；历史与 Semantic ID 逐条一致 |
| 重复 Target / 新商品 | 23 / 2,662（主评测保留，`novel_item` 子集另报） |
| 历史长度 | min 2 / p50 3 / p90 10 / max 10 / mean 4.59 |
| Profile（100 条 Pilot） | 71 条有品类兴趣、97 条有品牌兴趣、90 条有属性兴趣、40 条有负向信号 |
| Query 生成 | 97 generated + 3 insufficient_profile（无强制生成） |
| 独立审核 | 82 PASS / 14 REVIEW / 1 FAIL / 3 ERROR |
| Python Schema + Leakage | 97 条通过；Target ID/Title 精确扫描 **0 泄漏** |
| 人工复核队列 | **20 条待人工**（14 REVIEW + 1 FAIL + 3 ERROR + 2 条 PASS 抽样） |
| 冻结状态 | **未冻结**（`final_frozen_eligible=0`；人工结论尚未录入） |
| 三路召回（82 条 PASS，Recall-only） | TIGER 0% / E5 2.44%@10、4.88%@50 / Hybrid 2.44%@10、4.88%@50 |
| D11 关系 | 与 Phase 4 No-Go 一致：TIGER 在本 Pilot 未给 E5 带来增量（rescue=0） |

## 2. 阶段 A：时间截点审计

从现有 TIGER test 样本恢复 `user_id`、`cutoff_time`、`history_item_ids`、`history_event_ids`、
`target_product_id`、target 事件与时间，并逐条验证：

- 历史事件 `timestamp < target_timestamp`，`cutoff = target_timestamp - 1`；
- Target 事件 `timestamp > cutoff`；
- 用 `SemanticIdTable` 将历史商品映射回 Semantic ID，与样本中的 `history_ids` **完全一致**；
- 历史/目标商品都在冻结目录 `all_beauty_v2` 中。

结果：**2,685 条全部可恢复，失败 0**。`sample_id = sha256(user_id|target_timestamp|target_product_id)[:16]`。
Target 标签单独写入 `labels/target_labels.jsonl`，样本文件不含 Target。

## 3. 阶段 B：Profile Pilot（100 条）

Profile V1 只使用截点前的商品 Title/Brand/Category/Details/Rating/Timestamp，不读取评论正文、
Target 或未来事件。事实由 Python 规则生成，包含 `fact_id`、`source_event_ids`、支持计数与固定
规则置信度。

| 画像字段 | 有该字段的样本数 | 事实总数 |
|---|---|---|
| category_interests | 71 | 165 |
| brand_interests | 97 | 282 |
| attribute_interests | 90 | 337 |
| negative_signals | 40 | 142 |

`unknown_preferences` 分布：scent 62、skin type 54、hair type 44、budget 42（缺失即 unknown，不由模型补造）。

## 4. 阶段 C：Query 生成（PFlash）

- 模型：`deepseek-flash`，temperature 0，JSON mode，`reasoning_effort=none`；
- Prompt：`query_generator_v1`（hash `4ce3807e6425c69d`）；
- 输入仅 Profile 事实 + `sample_id/cutoff`，**代码层不传 Target**；
- 每样本一个 Query，使用 `used_fact_ids` 回溯证据；
- 结果：**97 generated、3 insufficient_profile**（无强制生成）；
- Query 长度中位数 10 词（3–16）；
- 生成阶段 token：prompt 190,415 / completion 20,212（含审核；见质量报告）。

## 5. 阶段 D：独立审核与 Python 校验

- Auditor 使用独立 Prompt `query_auditor_v1`（hash `45983f014da1ec6e`），只接收 Profile 与候选 Query，
  不接收 Generator 的 `used_fact_ids` 或 Target；
- 状态：82 PASS / 14 REVIEW / 1 FAIL / 3 ERROR（ERROR 为 insufficient_profile 未送审）；
- Python 校验：97 条通过；Target ID 与完整 Title 精确扫描 **0 泄漏**；
- 6 条 unsupported claim 被 Auditor 记录；
- REVIEW 的典型原因：Query 过度泛化（"beauty product"）、把不相关品类/属性拼接、与低置信负向信号冲突；
- 1 条 FAIL 经一次重生成后仍失败（`sample_140ca2d82ac60ed5`），进入人工队列；
- 人工队列 20 条（含 2 条 PASS 抽样，约 3%），`human_audit.csv` 的人工列**全部为空**，未伪造审核结论。

## 6. 阶段 F（部分）：Recall 阶段三路对照

在 **82 条 PASS** 上使用同一样本、同一 Target：

| 方法 | Recall@10 | Recall@50 | Recall@100 | NDCG@10 | MRR | 候选覆盖率 |
|---|---|---|---|---|---|---|
| TIGER-only | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100% |
| E5-only | **2.44%** | **4.88%** | 4.88% | 1.99% | 1.95% | 100% |
| Hybrid RRF | 2.44% | 4.88% | 4.88% | 1.38% | 1.09% | 100% |

- TIGER 独有命中 Target：0；E5 未命中 78；Rescue Rate = **0.0**；
- Hybrid − E5：Recall@10 = 0、Recall@50 = 0；平均候选重叠 0.0047；
- 延迟（含模型加载/批处理摊销，非严格单请求延迟）：TIGER 15.9 ms/样本，E5 185.8 ms/样本；
- **未运行 Reranker**，因此不是完整协议 §10 的三路 + 统一 Reranker 结果；不能据此做最终统计结论。

## 7. 覆盖率和未冻结原因

- 原始 test：2,685；恢复：2,685；
- Pilot 抽样：100（按历史长度分层、固定 seed=42）；
- Profile：100；Query：97 generated；审核 PASS：82；校验通过：97；
- 人工复核：20 条待处理；**最终冻结可用样本：0**；
- 未冻结的原因：人工未审核（REVIEW/FAIL/ERROR 与 PASS 抽样未录入结论）；Reranker 阶段未运行；
- 因此本 Pilot 仅用于验证数据生产链路和暴露问题，不作为最终测试集或最终模型结论。

## 8. 与协议验收的差距

| 协议要求 | 当前状态 |
|---|---|
| 唯一 sample_id、时间/商品映射 | ✅ 2,685 条全部通过 |
| Profile 事实均有来源、缺失不补造 | ✅ 规则化，100 条 Pilot 检查 |
| Query 关联 sample_id、synthetic、used_fact_ids | ✅ |
| Generator 不接触 Target、Auditor 独立 | ✅ 代码隔离 + 0 泄漏扫描 |
| REVIEW 人工处理 | ❌ 20 条待人工 |
| 全量生成 + 审核 + 冻结 | ❌ 仅 100 条 Pilot；全量 2,685 未生成 |
| 三路 + 统一 Reranker + 统计区间 | ⚠️ 仅 Recall 阶段；无 Reranker；样本 82 不做显著性声明 |
| Manifest/版本/哈希 | ✅ 已生成；人工审核标记为未完成 |

## 9. 成本与资源

| 项 | 值 |
|---|---|
| Pilot 运行 | 约 3 分 31 秒（100 条 Query + 审核；不含模型/目录加载） |
| API token（Pilot 100） | prompt 190,415 / completion 20,212 |
| Query/Audit 模型 | `deepseek-flash`，temperature 0，JSON mode |
| 数据大小 | samples/labels/profiles/queries/audits/reports 均在 NFS；仓库仅保留报告与 manifest |
| 目录版本 | `all_beauty_v2`，E5 `intfloat/e5-base-v2` |

## 10. 下一步（需人工或更大预算）

1. **人工审核 20 条队列**，把结论写回 `human_audit.csv`；PASS 抽检若发现系统性问题，修正 Prompt/Profile 后重新生成新版本；
2. 全量 2,685 条 Query 生成 + 审核（成本约 Pilot 的 27 倍 token，尚未执行）；
3. 统一 Reranker 后重跑三路 + Reranker 指标（协议 §10.2）；
4. 按用户 cluster bootstrap 报告置信区间；若区间跨零标记 inconclusive；
5. 冻结 `v1.0` 数据集与 manifest，再用于 Phase 7 最终报告。

## 11. 复现

```bash
# 阶段 A：时间审计
agr eval audit \
  --samples artifacts/semantic_ids/samples_a1/test.jsonl \
  --db data/processed/interaction_history.db \
  --semantic-ids artifacts/semantic_ids/rqvae_pilot_a1 \
  --catalog artifacts/catalog/all_beauty_v2 \
  --out data/evaluation/audit_v0.1

# 阶段 B–D：100 条 Pilot（需要 DEEPSEEK_API_KEY）
agr eval pilot \
  --samples artifacts/semantic_ids/samples_a1/test.jsonl \
  --db data/processed/interaction_history.db \
  --semantic-ids artifacts/semantic_ids/rqvae_pilot_a1 \
  --catalog artifacts/catalog/all_beauty_v2 \
  --out data/evaluation/v0.1-pilot --limit 100

# 阶段 F（仅 Recall）：TIGER / E5 / Hybrid
python scripts/retrieval_comparison_pilot.py \
  --pilot data/evaluation/v0.1-pilot \
  --semantic-ids artifacts/semantic_ids/rqvae_pilot_a1 \
  --generator-dir artifacts/semantic_ids/generator_pilot_a1 \
  --bundle artifacts/faiss/e5-base-v2_all_beauty_v2 \
  --db data/processed/interaction_history.db \
  --out data/evaluation/v0.1-pilot/reports/retrieval_comparison_pilot.json
```

`data/` 与模型/索引不提交 Git；报告、manifest 与代码提交到仓库。
