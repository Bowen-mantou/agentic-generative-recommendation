# ADR-0005: First-stage data scope is All_Beauty

- **状态**：已采纳（可由负责人扩大）
- **日期**：2026-11-24
- **上下文**：AGENTS 只规定数据来源为 Amazon Reviews 2023，没有规定品类范围。
  设计文档示例使用 Beauty/skincare 商品，因此单品类 `All_Beauty` 与文档示例语义一致，
  同时数据量可控（metadata 213 MB、reviews 327 MB、112,590 条商品记录）。
  服务器 `/home` 空间紧张，多品类会显著放大原始数据、embedding 与索引。

## 决策

第一阶段使用 Amazon Reviews 2023 的 **`All_Beauty`** 品类：

| 项 | 值 |
|---|---|
| 源数据集 | `McAuley-Lab/Amazon-Reviews-2023` |
| metadata | `raw/meta_categories/meta_All_Beauty.jsonl`，212,990,142 B，SHA-256 `1cfbdf6d…` |
| reviews（Phase 4/6 用） | `raw/review_categories/All_Beauty.jsonl`，326,611,506 B |
| 商品主键 | `parent_asin → product_id`（一对一映射，保留来源字段） |

## 理由

- 与六份文档的 Beauty 示例同域，Planner/Query/评测样例不需要跨域改写；
- 112k 商品足以支撑 10 万级 FAISS 精确检索与 Recall@50/100 评测；
- 单品类可把 TIGER 用户序列与 RQ-VAE 训练控制在可承受规模。

## 后果与风险

- 结论不能外推到全品类或多语言场景；品类扩展需要重建 catalog 与索引；
- 长尾/冷启动分析在单品类内进行，样本量有限（Phase 7 需说明）；
- reviews 尚未下载/建表（Phase 4/6 前置）；数据量已实测，可随时补下。
