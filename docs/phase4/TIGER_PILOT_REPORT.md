# Phase 4 TIGER Pilot 报告（A1：3×256，序列 ≥3）

- 日期：2026-11-24
- 依据：[Phase4 实施计划](../decisions/Phase4_TIGER_IMPLEMENTATION_AND_PHASE5-7_PLAN_V1.0.md) §2.4–2.6、ADR-TIGER-RETRAINING-ROUTE
- 状态：**Pilot 已执行；RQ-VAE 编码通过，生成式召回质量未达门槛；未启动全量训练**
- 产物：`/mnt/agr-data/artifacts/semantic_ids/{rqvae_pilot_a1,generator_pilot_a1,samples_a1}`
- 后续受控实验（B1/B2）与 D11 最终判定：[D1_B1_B2_GATE_REPORT.md](D1_B1_B2_GATE_REPORT.md)

## 1. 结论摘要

| 项 | 结果 |
|---|---|
| 外部实现审计 | Cornac v3.2.0（Apache-2.0，tag `bb3b42d`）在 CPU 上安装/导入/768 维自定义 embedding/序列/时间切分全部可用；无 TIGER 权重（[实现计划](TIGER_IMPLEMENTATION_PLAN.md)） |
| RQ-VAE 编码 | **通过**：码本利用率 100% / 98.8% / 97.7%，dead code 0 / 3 / 6，val recon MSE 1.87e-4，Unique ID 100,702/112,590（89.4%） |
| Semantic ID 映射 | 17.5% 商品与其他商品共享 ID（最大桶 25）；解码为一对多，需消歧 |
| 用户序列 | 8,201 用户（≥3 事件）→ **19,222 样本**：train 13,710 / val 2,827 / test 2,685（按全局时间分位 80%/90%） |
| 生成器 | 470,272 参数，CPU 训练 110.7 s；best val epoch 7（val 4.924），最终过拟合（val 5.499） |
| 测试集效果（best ckpt） | **Recall@10 0.15%、Recall@50 0.19%、NDCG@10 0.09%、MRR 0.08%**；合法 ID 93%，平均 10 个 ID，17.8 ms/样本 |
| 参考基线（同一测试集） | 热度 Top-50：Recall@10 **0.34%** / Recall@50 0.78%；E5 用户画像（500 样本）：Recall@10 **1.6%** / Recall@50 2.6% |
| 评审结论 | **A1 生成的召回低于朴素热度基线，远低于 E5 用户画像；不满足进入全量训练的条件** |

> 任务表述：这是“基于历史交互的下一商品预测”离线评测，不是购买转化率；所有事件来自
> Amazon Reviews 代理数据，无模拟数据参与。

## 2. Step 1：数据与外部代码审计

- 商品向量：112,590×768 float32，finite、无零行、单位范数、与 `product_ids.jsonl` 行序一致。
- Reviews：694,252 事件 / 631,986 用户 / 112,565 商品；0 条模拟事件。
- 外部代码：Cornac 实测 RQ-VAE 112,590×768 **约 1.7 s/epoch**；默认 seq2seq 4.66M 参数、5,000 样本约 39.6 s/epoch；SID pickle 31.6 MB；**无 TIGER 预训练权重**。
- 结论：底座沿用本项目自研 A1（可复现、无新运行时依赖）；Cornac 保留为独立校验/回退。

## 3. Step 2：RQ-VAE A1

### 3.1 第一次失败与修复（真实记录）

首轮（20 epochs、随机初始化码本）出现**严重码本塌缩**：level-1 仅使用 4/256 码，
dead code 252；Unique ID 仅 4,457（碰撞率 99.4%，最大桶 586）——命中计划 §9.3 停止条件。

修复（标准 RQ-VAE 做法）：

1. 先做 5 epoch 自编码器 warm-up（不量化）；
2. 对每一级残差用 k-means（20,000 样本、25 迭代）初始化码本；
3. 主训练 100 epochs（batch 1024、lr 1e-3）。

### 3.2 修复后指标（A1 最终）

| 指标 | 值 |
|---|---|
| 训练时长 / 峰值 RSS | **77.0 s** / 875 MB（CPU，48 线程） |
| 重建 MSE（train / val，最后 epoch） | 1.86e-4 / 1.87e-4 |
| 码本利用率（L1/L2/L3） | **100% / 98.83% / 97.66%** |
| Dead codes（L1/L2/L3） | 0 / 3 / 6 |
| Perplexity（L1/L2/L3） | 233.96 / 242.96 / 241.34 |
| Unique Semantic ID | 100,702 / 112,590（**89.4% 唯一**） |
| 碰撞率（商品占比） / 最大桶 | **17.5%** / 25 |
| prefix-1 / prefix-2 唯一数 | 256 / 37,887 |
| checkpoint / codes | 15 MB / 1.35 MB（112,590×3 int32） |

## 4. Step 3：序列与生成器

### 4.1 时间切分样本

| 项 | 值 |
|---|---|
| 用户（≥3 事件） | 8,201 |
| 样本（history ≥2） | **19,222** |
| train / validation / test | 13,710 / 2,827 / 2,685 |
| 时间边界 | train_ts ≤ 1,626,820,406；val_ts ≤ 1,648,574,591；test 更晚（Unix 秒） |
| 跳过 | short history 16,402；缺失 SID 0 |

### 4.2 生成器

| 项 | 值 |
|---|---|
| 结构 | decoder-only Transformer，2 层，d=128，4 heads，FFN 256，词表 768 码 + BOS/PAD |
| 参数 / 训练时长 / RSS | 470,272 / 110.7 s / 971 MB |
| 训练配置 | 40 epochs，batch 256，AdamW lr 1e-3，beam 10 |
| best val | epoch 7，val 4.9237（最终 epoch val 5.4994 → 过拟合） |
| 生成 | 每样本 10 个 Semantic ID；合法 ID 比例 93.0%；17.8 ms/样本 |

### 4.3 测试集结果与基线对比

| 方法 | Recall@10 | Recall@50 | NDCG@10 | MRR |
|---|---|---|---|---|
| TIGER A1（best ckpt） | 0.0015 | 0.0019 | 0.0009 | 0.0008 |
| 热度 Top-50（全用户相同） | 0.0034 | 0.0078 | 0.0013 | 0.0008 |
| E5 用户画像（500 样本，均值向量→FAISS-50） | **0.0160** | **0.0260** | 0.0083 | 0.0062 |

> 基线用于判断“模型是否优于平凡方法”；E5 画像只用了 500 个测试样本以控制时间，
> 不是正式实验结论，Phase 7 会统一协议。

## 5. Pilot 评审（对应计划 §2.5 Step 4）

| 问题 | 回答 |
|---|---|
| 1. 外部代码能否稳定运行？ | 可以（Cornac 只读验证通过；本项目实现不依赖它） |
| 2. RQ-VAE 能否生成稳定 Semantic ID？ | 能；利用率健康、映射覆盖 100% 商品 |
| 3. 码本塌缩/Dead code？ | 首轮严重塌缩，修复后 dead code 0/3/6，不构成阻塞 |
| 4. 生成器是否学到下一商品预测？ | **弱**：Recall@10 0.15%，低于热度基线 0.34% |
| 5. 生成结果能否解码？ | 能；93% 生成 ID 可映射到真实商品 |
| 6. 设备/预算能否扩大训练？ | CPU 可承担（RQ-VAE 77 s、生成器 111 s）；瓶颈是**数据规模**与泛化，不是算力 |
| 7. A1 是否足以进入全量训练？ | **否**；A2（更大码本）不解决样本不足，且会扩大候选空间 |

## 6. 停止条件与待决策（Q1/Q5/Q12）

按计划 §9，以下条件被触发：**Phase 4 预计无法在项目期限内达到可用质量**（数据规模限制）。
已提交的证据：训练/验证曲线、测试指标、两条参考基线、资源实测。

| 选项 | 内容 | 影响 |
|---|---|---|
| A2 受控对照 | 4 级 × 1024、序列门槛 ≥2（43,044 用户），仅改变码本 + 样本范围中的一项做对照 | 需额外 1–2 天；仍可能受样本限制 |
| 数据范围调整 | 扩展品类或引入更多序列（需修改 Q5） | 与已确认“仅 All_Beauty”冲突，需负责人批准 |
| 任务表述调整 | 接受 TIGER 离线质量不足，保留“真实生成链路”证据但不作为主力召回 | 需重新定义 Phase 4/7 验收 |
| 偏差方案 | 暂以 E5 单路完成 Phase 5/6 联调，提交 Q12 偏差说明 | 不得宣称完成双路召回验收 |

在负责人确认前：**不启动全量训练**、不修改 Q5/Q11 边界、不把 E5 单路结果写成双路验收。

## 7. 资源与可复现

| 项 | 值 |
|---|---|
| 设备 | CPU 48 线程（Q11 基线），128 逻辑核宿主 |
| RQ-VAE | 77 s、875 MB、checkpoint 15 MB |
| Generator | 110.7 s、971 MB、checkpoint 3.7 MB（best + final） |
| 数据快照 | interactions DB（694,252 事件）；bundle `e5-base-v2_all_beauty_v2` |
| 代码版本 | commit 见 `manifest.json` / `generator_manifest.json`（`git_commit`） |
| 配置 | `configs/tiger/rqvae_pilot_a1.yaml`、`configs/tiger/generator_pilot.yaml` |
| 随机种子 | 42（RQ-VAE 与生成器） |

## 8. 未验证项

- A2（4×1024 / ≥2 序列）未运行；A1 的逐级消融（1/2/3 层 ID）未做。
- 生成器的 beam/温度/长度策略只用了固定配置，未调参；early-stopping 已按 best-val 评估。
- 未做 TIGER-only 与真实用户 query 的对照（需要 Phase 7 统一协议）。
- SID 碰撞消歧只用了“热度 + 生成分”，未做 embedding 重建向量重排。
