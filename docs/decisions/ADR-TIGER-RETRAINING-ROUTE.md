# ADR-TIGER-RETRAINING-ROUTE：B→A（重解释）+ A3 Pilot

- 状态：已确认（2026-11-24，项目负责人）
- 依据：`docs/decisions/Phase4_TIGER实施方案与Phase5-7集成计划_V1.0.md` §1.1
- 关联：Q1 / Q11 / Q12

## 决策

1. **Q1 采用 B→A（重解释）**：复用外部 TIGER 代码/结构/训练流程，但使用本项目
   All_Beauty 商品目录、E5 商品向量与真实 reviews 序列，**重新训练 RQ-VAE 与自回归生成器**；
   不直接使用外部预训练权重，不使用 Mock / 随机 Semantic ID / 规则映射。
2. **训练策略 A3**：先执行 A1 Pilot（3 级 × 256、序列长度 ≥3），达到约定验收条件后
   才进入全量训练；若 A1 暴露明确编码问题，再做 A2（4 级 × 1024、≥2）受控对照。
3. **Q11 不变**：第一阶段 CPU 基线；Pilot 必须先实测训练时间/峰值内存/checkpoint 大小。
4. **Q12 不变**：Phase 0–6 完整验收要求真实 TIGER + E5 双路；若 Pilot 失败或超预算，
   提交偏差说明并由负责人重新批准。

## 后果

- Phase 4 的验收证据包括：RQ-VAE 训练与编码、Semantic ID 映射覆盖、真实时间序列、
  真实生成与解码、时间切分 Recall/NDCG/MRR、资源记录与可复现配置。
- Phase 5/6 可先行用 E5 候选池开发（Milestone 1），不阻塞于 TIGER。
- 所有 Pilot 配置、checkpoint、manifest 与报告写入 `configs/tiger/`、`artifacts/`（NFS）
  与 `docs/phase4/`。
