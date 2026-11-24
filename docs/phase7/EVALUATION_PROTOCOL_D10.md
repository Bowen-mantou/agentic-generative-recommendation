# Phase 7 评测协议（D10：人工 Query 为主、Persona 合成为辅）

- 日期：2026-11-24
- 依据：D10 = C+A；`docs/decisions/Phase4-7_D1-D11_DECISION_DRAFT_V1.0.md` §12
- 状态：**协议与模板已建立；人工标注尚未进行**（不能由 AI 代替人工标注）

## 1. 三类独立评测协议

| 协议 | 用途 | 标签来源 | 数据文件 |
|---|---|---|---|
| 人工自然语言 Query | E5 文本召回、Fusion、Reranker、Judgment、Planner 计划质量 | 人工标注（相关性等级） | `data/samples/eval/manual_queries_template.csv`（待标注） |
| Persona 合成 Query | 个性化与多轮对话场景 | 历史行为 + 模拟生成（`is_synthetic=true`） | 待 Phase 6 模拟对话生成后写入 |
| TIGER Next-item | 序列预测下一商品 | 时间切分后的真实交互代理标签 | `artifacts/semantic_ids/samples_*/{train,validation,test}.jsonl` |

三类指标**分开报告**，不得混合；Amazon Reviews 是评论/评分代理数据，离线 next-item 结果不能
解释为真实购买转化率。

## 2. 人工 Query 集规范

- 规模：首批 100–300 条；
- 每条字段：`query_id, language, query, hard_constraints, soft_preferences, relevant_product_ids, relevance_grades, rationale, annotator, annotated_at, split_time_cutoff`；
- 相关性等级：`3 = 完全满足`、`2 = 部分满足`、`1 = 弱相关但不满足硬约束`、`0 = 不相关`；
- 硬约束必须来自 Query 文本，不允许标注者补充用户未表达的要求；
- 标注前隐藏目标商品/未来评论信息；标注者不得查看测试期行为；
- 相关性判断需要引用商品目录字段（title/description/details），不得凭印象。

模板：`data/samples/eval/manual_queries_template.csv`（仅表头，尚无标注数据）。

## 3. Persona 合成 Query 规范

1. 只使用预测时间点**之前**的交互（`timestamp < cutoff`）；
2. 由 Planner/模拟器生成的 Query 与对话必须带 `is_synthetic=true`；
3. 不得读取未来评论、目标商品、测试期 Memory；
4. 合成 Query 与人工 Query 指标分开报告；不得用合成数据替代人工评测结论；
5. 生成 Prompt、模型版本、随机种子写入实验记录。

## 4. 防泄漏检查清单

- [ ] 人工 Query 标注时未见目标商品与测试期行为；
- [ ] 商品索引只使用允许访问的商品目录（不含测试期新增商品）；
- [ ] Persona 输入仅使用截点前事件；
- [ ] 模拟数据标记 `is_synthetic`，未混入 TIGER 训练序列；
- [ ] 评测脚本固定数据快照、配置、随机种子与 git commit。

## 5. 指标

| 协议 | 指标 |
|---|---|
| 人工 Query | Recall@10/50、NDCG@10、MRR、Reranker NDCG@10、Judgment 一致性 |
| Persona 合成 | 任务完成率、约束满足率、多轮一致性（不计入真实指标） |
| Next-item | Recall@10/50、NDCG@10、MRR、生成合法率、解码成功率 |
