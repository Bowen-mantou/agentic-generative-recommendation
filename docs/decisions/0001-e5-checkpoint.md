# ADR-0001: E5 checkpoint for the semantic retrieval baseline

- **状态**：已采纳（可配置，待实验确认）
- **日期**：2026-11-24
- **上下文**：AGENTS.md §8.4 把 “E5 具体 checkpoint、前缀格式和向量维度” 列为未验证事项；
  模块三只写 “E5” 而未指定权重。Phase 1 必须选定一个真实可下载的 checkpoint 才能建立
  索引并验收。

## 决策

默认使用 `intfloat/e5-base-v2`：

| 项 | 值 |
|---|---|
| 维度 | 768 |
| 前缀 | passage: `passage: `；query: `query: ` |
| 归一化 | L2，FAISS `IndexFlatIP` = 余弦相似度 |
| 语言 | 英文（Amazon Reviews 2023 语料为英文） |
| 配置位置 | `configs/models.yaml: e5.*` |

模型选择、维度、前缀、`max_length` 都写入 `artifacts/*/manifest.json`，索引与模型版本
绑定；换模型必须重建索引（`agr index build`），不允许混用。

## 理由

- 768 维在 10 万级商品上是精确检索可承受的规模（索引约 340 MB）；
- E5-base-v2 是检索任务的标准基线，社区成熟、权重可从镜像下载；
- 语言与数据一致，避免多语言模型带来的无谓开销。

## 备选

- `intfloat/e5-large-v2`（1024 维）：质量可能更高，CPU 编码与索引成本约 1.7×；
- `intfloat/multilingual-e5-base`：仅在需要多语言商品/查询时启用；
- 任何替换都通过配置进行，并在实验报告中记录 manifest 哈希。

## 影响

- 该选择是**基线配置**，不是论文结论；Phase 7 的对照实验可以替换。
- 若 Phase 7 证明 E5-base 不足，升级只改配置 + 重建索引，不改代码。
