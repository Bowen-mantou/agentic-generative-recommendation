# Phase 4 TIGER 实现计划：外部实现审计与适配（V1.0）

- **版本**：V1.0
- **日期**：2026-11-24（Asia/Shanghai）
- **作者**：tiger-audit（TIGER 工程审计员）
- **状态**：审计完成；底座选择与全目录 SID 方案待 Lead / 项目负责人确认
- **关联决策**：Q1（B→A 重解释 + A3 Pilot）、Q10（存储）、Q11（CPU 基线）、Q12（验收边界）；
  `docs/decisions/ADR-TIGER-RETRAINING-ROUTE.md`
- **上游要求**：`docs/decisions/Phase4_TIGER%E5%AE%9E%E6%96%BD%E6%96%B9%E6%A1%88%E4%B8%8EPhase5-7%E9%9B%86%E6%88%90%E8%AE%A1%E5%88%92_V1.0 (1).md`
  §2.2（外部代码复用范围）、§2.5 Step 1（数据与代码审计）、§8（文件/接口/轨迹）
- **证据快照**：2026-11-24 11:22–11:45 CST。文中的 stars、最后提交、release、PyPI 元数据均为当日抓取；
  服务器实测命令与输出见附录 A。
- **约束**：只读审计；未修改项目代码、配置与数据；服务器测试只使用 `/tmp` 临时 venv 与脚本；
  **未执行正式训练**，所有 toy 结果只证明“能跑通”，不是模型效果。

---

## 0. 结论摘要（先看这一节）

**审计结论等级：代码可复用，权重不可复用；本项目必须用自己的 All_Beauty 商品目录与 E5 向量重训。**

| # | 结论 | 证据强度 |
|---|---|---|
| 1 | **Cornac v3.2.0 是本审计实际安装运行、并逐项核对 §2.2 检查表的唯一外部实现**：Apache-2.0、维护活跃（最后提交 2026-11-23）、Python≥3.10、CPU 可运行、支持自定义 768 维商品 embedding、支持自定义 next-item 序列与**全局时间切分**、内置 RQ-VAE/rkmeans tokenizer + T5-style seq2seq + 约束 beam + exact 全目录打分 + Semantic-ID 指标。 | 高（源码 + 服务器实测） |
| 2 | Cornac 的 TIGER 模块是 **2026 年新增代码**（v3.0.0，PR #711；v3.2.0 最新），README 明确写 “There is no official code release”，属于社区高质量移植；v3.2.0 与 master 的 `cornac/models/tiger/*` 文件 **blob SHA 完全一致（内容相同）**，可安全锁定 v3.2.0。 | 高（Release/Git tree 对比） |
| 3 | **Cornac 不是开箱即用的生产底座**：SID 的商品域 = 数据中出现的商品（train+val+test 全局 item map），本项目 112,590 个商品中有 25 个从未出现在 reviews 中，因此**全目录 SID 覆盖需要适配/打补丁**；checkpoint 是整模型 pickle，没有 manifest/版本字段；TIGER 无专属单元测试；安装会额外引入 scipy/matplotlib/powerlaw。 | 高（源码 + 实测） |
| 4 | **工程底座建议：沿用仓库内已出现的 A1 自研实现（A3 路线），Cornac 作为独立校验基线与回退底座**。理由：自研实现已覆盖 RQ-VAE/SID/序列/生成器/checkpoint 全链路并产出 pilot 物；Cornac 替换它需要额外适配与回归，收益主要是“独立第二实现”。**若负责人坚持 Q1 字面的“外部代码优先”，则 Cornac + §5 适配层是唯一可行方案（估 6–10 人时，未实测）**，不建议两套长期并行。 | 中高（审计判断，非实验结论） |
| 5 | 其余外部实现只作参考：`torch-rechub` 数据接口最接近本项目（`inter.json` + `semantic_ids.json`），`Botta` 有可复现的 RQ-VAE 权重但绑定 P5 数据，`XiaoLongtaoo` 依赖过旧且 `generate_code.py` 硬编码 `cuda:0`，`genrec` 配置最重且无权重。**没有任何外部 TIGER 生成器权重可直接用于 All_Beauty。** | 高（仓库/依赖/README 核对） |

**一句话**：**Cornac 是最合格的外部代码底座，但本项目更合理的是“自研实现 + Cornac 交叉校验”**；
两条路线共享同一套数据契约（§5），都必须在 112,590 个商品、真实时间序列上重训 RQ-VAE 与生成器。

---

## 1. 审计范围、方法与运行环境

### 1.1 检查项（对齐 Phase4 方案 §2.2）

| 检查项 | 本次做法 | 结果位置 |
|---|---|---|
| License | GitHub API `license` + PyPI METADATA + 仓库 `LICENSE` | §2.1 |
| 代码可运行性 | 服务器 `/tmp` 临时 venv 安装 + import + toy 端到端 | §2.7 / 附录 A |
| 模型结构 | 定位 RQ-VAE、Semantic ID 编码/解码、seq2seq、约束 beam | §2.3 / §2.4 |
| 训练入口 | 示例脚本、`cornac.Experiment`、`TIGER.fit`、数据格式 | §2.5 / §2.6 |
| 向量接口 | `FeatureModality(features, ids)` + 768 维实测 | §2.5 / §2.7 |
| 商品映射 | `iid_map`、`sid_to_item`、`item_ids`、save/load 后恢复 | §2.6 |
| 序列接口 | `NextItemEvaluation` 的 SIT/USIT、`from_timestamps` 时间切分 | §2.5 |
| 推理接口 | `score` / `generate_beam` / `score_all_items` / SID 指标 | §2.4 / §2.6 |
| 数据核验 | 本项目 E5 向量、reviews/DB、SID 碰撞（只读） | §4 |

### 1.2 证据采集方式

- GitHub：`api.github.com/repos/...`、`/tags`、`/releases`、`/git/trees?recursive=1`、`/commits`。
- 源码读取：`cdn.jsdelivr.net/gh/<owner>/<repo>@<ref>/<path>`（本机 `raw.githubusercontent.com` 不稳定）。
- PyPI：`https://pypi.org/pypi/cornac/3.2.0/json`。
- 服务器：`python /c/new/intern/plan/_work/s4000_ssh.py "<cmd>"`（凭据在仓库外，本文不记录密码）。
- 数据核验：只读 `numpy.load(embeddings.npy, mmap_mode='r')` 与 SQLite `mode=ro`。

### 1.3 服务器环境（实测，2026-11-24）

| 项目 | 值 | 说明 |
|---|---|---|
| 宿主 | MTT S4000，`agr-worker` | 共享机，非独占 |
| Python | 3.10.12 | 宿主 `python3`；`ensurepip` 缺失 |
| CPU / 内存 | 128 核 / 1007 GB（可用 908 GB） | `nproc` / `free -g` |
| 负载 | `loadavg 60.40 / 39.72 / 20.40` | **计时为共享负载下的量级参考** |
| `/tmp` | 根分区 246G，已用 113G，可用 120G | 临时 venv 放这里 |
| `/home` | 191G，可用 8.9G | 不能放大文件 |
| 宿主 torch / numpy / transformers | 2.14.1+cpu / 2.2.6 / 5.19.0 | 审计 venv 借用系统 site-packages |
| 项目 venv `/tmp/agr/venv` | torch 2.6.0+cpu、numpy 2.2.6、faiss 1.16.0、transformers 4.57.6 | 项目既有环境（只读检查） |
| GPU | 未使用 | Q11 第一阶段 CPU 基线 |

### 1.4 本审计未做的事

- 未在 All_Beauty 全量数据上训练 Cornac TIGER；未跑官方 Amazon 2014 复现实验。
- 未下载任何 HuggingFace 权重（`huggingface.co` 直连超时，沿用既有 `hf-mirror.com` 结论，未新增验证）。
- 未安装/运行 Botta、XiaoLongtaoo、torch-rechub、genrec（只做仓库与依赖静态对比）。
- toy 实验使用随机合成向量/商品 ID，**只证明接口可运行，不代表召回效果**。

---

## 2. Cornac 完整审计

仓库：[PreferredAI/cornac](https://github.com/PreferredAI/cornac)

### 2.1 仓库元数据（2026-11-24 抓取）

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/PreferredAI/cornac |
| 描述 | A Comparative Framework for Multimodal Recommender Systems |
| Stars / Forks / Open issues | 1,065 / 181 / 24 |
| License | **Apache-2.0**（GitHub API `spdx_id=Apache-2.0`；PyPI METADATA 同） |
| 创建时间 / 最后 push | 2018-07-17 / **2026-11-23T00:45:20Z** |
| 最新 release | **v3.2.0**（2026-11-13T18:02:18Z） |
| release tag commit | **`bb3b42d425f5d44079c001ffd0c197a1a7c2a6f9`**（2026-11-13T17:55:44Z，“chore: bump version to 3.2.0 (#743)”） |
| master HEAD | **`8923b8b0074b38d20a28c38df31a2bd83249ca03`**（2026-11-23T00:45:19Z） |
| TIGER 引入版本 | **v3.0.0**（2026-08-27，PR #711 “Add TIGER model for Semantic ID recommendation”） |
| PyPI | `cornac==3.2.0` 可安装；`requires-python >=3.10`；cp310 manylinux/macOS/Windows wheels |
| 维护状态 | 活跃（2026-11 仍在合并 PR）；未归档 |

> 关键一致性检查：`cornac/models/tiger/{README.md,__init__.py,grid_config.py,paischer_config.py,recom_tiger.py,tiger.py,requirements.txt}`
> 在 **v3.2.0 与 master 的 blob SHA 完全一致**；仅 `cornac/eval_methods/next_item_evaluation.py`、
> `cornac/models/recommender.py` 在 tag 之后有改动。因此 **锁定 v3.2.0 = 锁定本次实测代码**。

### 2.2 代码结构（v3.2.0）

| 路径 | 作用 | 关键符号 |
|---|---|---|
| `cornac/models/tiger/tiger.py` | 神经网络模块：RQ-VAE tokenizer + T5-style seq2seq | `RQVAE`、`TIGERSeq2Seq`、`_kmeans` |
| `cornac/models/tiger/recom_tiger.py` | 模型封装：tokenize、SID 构建、训练、打分、解码 | `TIGER`（继承 `NextItemRecommender, SemanticIDMixin`） |
| `cornac/models/tiger/grid_config.py` | GRID 复现配方（rkmeans + 短预算） | `GRID_CONFIG` |
| `cornac/models/tiger/paischer_config.py` | 论文对齐配方（RQ-VAE + 更长训练） | `PAISCHER_CONFIG`、`PAISCHER_SPORTS_CONFIG`、`PAISCHER_TOYS_CONFIG` |
| `cornac/models/tiger/README.md` | 论文对齐说明、Amazon 2014 结果表、与 GRID/PAISCHER 的差异 | — |
| `cornac/models/tiger/requirements.txt` | TIGER 额外依赖 | `torch>=1.12.0`、`transformers>=4.30.0` |
| `examples/tiger_example.py` | 端到端示例（Beauty 2014 + Sentence-T5） | `TIGER(**PAISCHER_CONFIG, …)` |
| `cornac/eval_methods/next_item_evaluation.py` | next-item 评测、会话数据、时间切分、SID 指标 | `NextItemEvaluation.from_splits/from_timestamps/leave_last_out` |
| `cornac/data/modality.py` | 商品内容向量 | `FeatureModality(features, ids)` |
| `cornac/models/recommender.py` | 基类：`iid_map`、`item_ids`、`total_items`、save/load | `Recommender.save/load`、`SemanticIDMixin` |
| `cornac/datasets/amazon_review.py` | 内置 Amazon 2014 loader | `load_feedback`、`load_text`（**仅 2014**） |
| `tests/cornac/models/test_semantic_id_adapters.py` 等 | SID adapter / 指标测试 | **未发现以 tiger 命名的专属测试文件** |

### 2.3 RQ-VAE 与 Semantic ID

- `RQVAE(input_dim, hidden_dims=(512,256,128), latent_dim=32, num_levels=3, codebook_size=256, beta=0.25)`：
  encoder MLP（768→512→256→128→32）+ 3 层残差码本（每层 256×32）+ 对称 decoder；Straight-Through
  estimator，loss = reconstruction MSE + codebook loss + β·commitment（β=0.25）。
- 训练细节（`TIGER._fit_rqvae`）：`AdamW`，默认 200 epochs、batch 1024；**K-means 初始化码本**
  （`kmeans_init_codebooks`）与**每 epoch 重启 dead codes**（`restart_dead_codes`）。
- 两种 tokenizer：`tokenizer="rqvae"`（论文路线，有梯度）或 `tokenizer="rkmeans"`（GRID 路线，逐层残差
  k-means，无梯度）。README（Amazon 2014 Beauty，R@5）报告：`paischer+rqvae 0.0419`（论文 0.0454）、
  `grid+rkmeans 0.0340`；rkmeans 与 rqvae 差距约 4% 以内，训练成本显著更低。
- `TIGER._build_semantic_ids(codes)`：在 `num_levels` 个码本之外**自动追加一层碰撞消歧位**
  （`dedup[i] = 同前缀商品出现序号`），得到完整 SID `sid_table (N, L+1)`；
  同时构建 `sid_to_item`（完整 SID → item index）与前缀 Trie `prefix_children`，用于约束解码。
- SID 长度示例：3×256 + 消歧层；在真实 E5 向量上（见 §4.2）消歧层可达 **86**，即 SID 为 4 个 token。

### 2.4 自回归生成模块

`TIGERSeq2Seq(level_sizes, d_model=128, d_ff=1024, num_heads=6, d_kv=64, num_enc_layers=4, num_dec_layers=4, dropout=0.1)`：

- 用 **T5Stack**（`transformers.models.t5.modeling_t5.T5Stack`）分别构建 encoder/decoder，**共享一张
  token embedding**（token 0 = padding；各层 token 用累积 offset 区分），每一层一个输出 head，decoder 用
  可学习 BOS 起始。默认架构约 **4.66M 参数（实测 4,657,536，level_sizes=[256,256,256,3]；
  [256,256,256,2] 时为 4,657,280）**；README 的 “~13M” 与实测不一致，以实测为准。
- 训练：teacher forcing，逐层 cross-entropy 求和；`max_len=20`（默认历史商品数），因此 encoder 输入为
  `max_len × (num_levels+1)` 个 token；支持 `model_selection="best"`（按验证集 NDCG/Recall 保存最优）。
- 解码：
  - `generate_beam`：**约束 beam search**，每层只允许 `prefix_children[level][prefix]` 中的 token；
  - `score_all_items`：对**全部商品**的 SID 做 teacher-forced log-likelihood（exact scoring），用于验证
    beam ≈ exact；
  - `TIGER.score(user_idx, history_items)` 返回长度 `total_items` 的分数数组。
- 生成器不含 user-ID token（README 说明：论文 2000 桶 user hash，GRID 消融显示去掉更优）。

### 2.5 数据接口（本项目最关心的部分）

**A. next-item 序列**

`NextItemEvaluation` 接受 `fmt`：

- `SIT`：`(session, item, timestamp)`
- `USIT`：`(user, session, item, timestamp)` ← 推荐
- `SITJson` / `USITJson`：追加 JSON 字段

构造方式：

| 类方法 | 用途 | 备注 |
|---|---|---|
| `from_splits(train_data, test_data, val_data=None, fmt="SIT", item_feature=...)` | 显式切分 | 会话可跨 split 时需自己保证不截断 |
| `from_timestamps(data, test_timestamp, val_timestamp=None, fmt="USIT", item_feature=...)` | **全局时间切分** | 会话原子：按会话最后事件时间归入 split；本项目应采用 |
| `leave_last_out(data, ...)` | 每个用户留最后一条 | 论文可比口径，但**不符合本项目时间切分要求** |

时间切分语义（源码注释）：`train: last_ts < val_timestamp`；`val: val_timestamp <= last_ts < test_timestamp`；
`test: last_ts >= test_timestamp`；`>=` 归入更晚 split。会话内相对顺序保留。

**B. 商品内容向量**

```python
FeatureModality(features=embeddings, ids=item_ids)
# features: (N, 768) float32; ids: 与行对齐的原始 item id（本项目 product_id）
```

- TIGER `_get_item_features()` 要求 `features.shape[0] >= total_items`，取 `features[:total_items]`；
  `total_items = len(self.iid_map)`，即 **train+val+test 的全局商品集合**。
- `FeatureModality.build(id_map)` 会按全局 id map 重排特征行（`_swap_feature`，内部 `np.copy` 整矩阵）。
- **实测**：768 维、字符串商品 ID（`p000`/`B000`）可直接使用（§2.7 / 附录 A）。

**C. 商品域（重要限制）**

- SID 只为 `iid_map` 中出现的商品建立；`FeatureModality` 里多出的目录商品会被忽略。
- Toy 实测：给 80 个商品特征，但数据只出现 71 个 → `sid_table` 只有 **71** 行、`total_items=71`。
- 本项目：catalog 112,590 个 `product_id`，DB 中出现过 **112,565** 个（差 **25** 个，0.022%）。
  也就是说 Cornac 默认会让 TIGER 永远无法生成这 25 个商品；若要求“全目录 SID”，必须 patch（§5.1）。

**D. 内置 Amazon loader**

- `cornac/datasets/amazon_review.py` 文档字符串写 “2014/2018/2023”，但 `_validate()` **实际只允许 `2014`**：
  `if version != "2014": raise ValueError(... only '2014' ... available`)。
- 本项目**不要**依赖该 loader；直接用自建 `USIT` 数据（绕过网络与 2014 字段差异）。

### 2.6 训练入口、checkpoint 与评测

- 入口：`cornac.Experiment(eval_method=NextItemEvaluation…, models=[TIGER(...)], metrics=[...]).run()`；
  或手动 `model.fit(eval_method.train_set, eval_method.val_set)`（本次实测采用后者做 toy）。
- 评测：`semantic_id_metrics=True` 时，除普通 item `Recall/NDCG/MRR` 外，还输出
  **`SID-Recall@K`、`SID-NDCG@K`、`CCE-Recall@K`、`CCE-NDCG@K`**（碰撞桶按均匀顺序折算），
  非法/重复预测会先被压缩；beam 大小不足时给出 warning。
- checkpoint（继承 `Recommender`）：
  - `model.save(save_dir=...)` → `<save_dir>/<name>/<timestamp>.pkl` + `.meta`（metadata JSON）；
  - `TIGER.load(path)` 静态方法，pickle 反序列化；
  - `fit()` 结束会把 `model` 与 `rqvae` 移回 CPU，便于跨 GPU/CPU 机器 pickle；
  - **实测**：默认架构 + 112,590 个随机 SID（消歧层 3）pickle 大小 **31.6 MB**；真实消歧层 86 时预计仍在
    **< 40 MB** 量级；toy 模型 0.3 MB；
  - **实测** save → load 后 `iid_map`、`sid_table`、`sid_to_item`、`item_ids` 全部保留，SID 可反查到
    `product_id`（附录 A.6）。
  - 未提供训练配置/manifest 版本字段；需要本项目自己补（§5 #6）。

### 2.7 依赖与运行性实测（服务器，CPU）

**安装**

```bash
# 1) 临时 venv（宿主无 ensurepip；--without-pip 借用系统 pip）
python3 -m venv --without-pip --system-site-packages /tmp/tiger_audit_venv
/tmp/tiger_audit_venv/bin/python -V          # Python 3.10.12
/tmp/tiger_audit_venv/bin/python -m pip -V   # pip 26.2.1（user-site）

# 2) 安装（清华镜像）
/tmp/tiger_audit_venv/bin/python -m pip install --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple cornac==3.2.0
```

实际安装结果（摘录）：

```text
Successfully installed cornac-3.2.0 contourpy-1.3.2 cycler-0.12.1 fonttools-4.65.0
  kiwisolver-1.5.1 matplotlib-3.10.9 pillow-12.3.0 powerlaw-2.0.0
  pyparsing-3.3.3 scipy-1.15.3
# 借用系统已有：numpy 2.2.6、torch 2.14.1+cpu、tqdm 4.70.1
```

- **重要**：`cornac==3.2.0` 的核心依赖是 `numpy>2.0.0`、`scipy`、`tqdm`、`powerlaw`（会带 matplotlib），
  **不包含 torch/transformers**；TIGER 的 `requirements.txt` 才声明 `torch>=1.12.0`、`transformers>=4.30.0`。
  安装后 `/tmp/tiger_audit_venv` 占 **438 MB**（后来装 transformers 4.57.6 后）。
- 与项目环境兼容性：项目 venv 已有 `numpy 2.2.6`（满足 `numpy>2`）、`torch 2.6.0+cpu`、`faiss 1.16.0`、
  `transformers 4.57.6`；**numpy 无冲突**。代价是新增 scipy/matplotlib/powerlaw 依赖树。

**import 与 toy 端到端**（合成数据，仅验证可运行性）

```python
import cornac, torch, transformers
from cornac.models import TIGER
from cornac.models.tiger import GRID_CONFIG, PAISCHER_CONFIG
from cornac.models.tiger.tiger import RQVAE, TIGERSeq2Seq
```

输出：

```text
cornac 3.2.0 | torch 2.14.1+cpu | transformers 5.19.0
TIGER import OK; GRID keys: ['batch_size','learning_rate','lr_schedule','model_selection',
  'n_epochs','tokenizer','val_eval_every','val_k','val_metric','val_sample','weight_decay']
RQVAE params: 1149472
Seq2Seq params: 4657280          # level_sizes=[256,256,256,2]
FeatureModality OK dim= 768
[transformers] Model config: eos_token_id must be None or an integer within the vocabulary
  (between 0 and 0), got 1.   # 仅 warning，不影响运行（T5Config vocab_size=1 的内部实现细节）
```

toy 端到端（80 商品 × 768 维，30 用户，每用户 6 事件，`rkmeans 3×16`，1 epoch）：

```text
train sets: num_items=62 num_users=21 num_sessions=21
Semantic IDs assigned: 71 items, 8 collisions, dedup level size 3
sid_table shape (71, 4) | level_sizes [16,16,16,3] | total_items 71 | pad_idx 71
fit seconds 2.88
score shape (71,) | top5 [36,29,14,9,59]
ckpt 2026-11-24_11-29-09-508547.pkl size_MB 0.3
load OK: TIGER sid_table (71,4) scoring beam
score reproducible: True
```

> 注意：80 个商品特征 → SID 只有 71 行，原因是数据中只出现 71 个商品（§2.5C 的实证）。

**transformers 版本兼容**：项目 `pyproject.toml` 要求 `transformers>=4.44,<5`。把审计 venv 降到
**4.57.6** 后重跑同一 toy：`4.57.6 / torch 2.14.1+cpu / cornac 3.2.0`，fit 2.15s，save/load 与分数复现均通过，
且 `eos_token_id` warning 消失。结论：**4.57.6 与 5.19.0 均可运行；项目应锁 4.x（与 pyproject 一致）**。

**时间切分 + RQ-VAE + exact scoring**（另一组合成 USIT 数据）：

```text
time-split: train items 24 users 6 | test items 24
rqvae+seq2seq fit seconds 2.1
sid_table (24, 4) levels [16,16,16,2] collisions 2
exact scoring sid_table (24, 3) max score idx 23
```

说明 `NextItemEvaluation.from_timestamps(data, test_timestamp=…, fmt="USIT", item_feature=…)` 可用；
`scoring="exact"`（全商品 teacher-forced 打分）可用。

**CPU 训练成本微基准**（合成数据，112,590×768，64 线程，共享宿主 loadavg≈60）：

| 阶段 | 实测 | 外推（量级） |
|---|---|---|
| RQ-VAE K-means 初始化码本 | **5.4 s** | 一次性 |
| RQ-VAE 1 epoch（batch 1024） | **1.7 s** | 20 epochs ≈ 0.6 min；100 epochs ≈ 3 min |
| 峰值 RSS（含 346MB 特征矩阵） | **1,190 MB** | — |
| seq2seq 1 epoch（5,000 samples，batch 256，80 tokens/样本） | **39.6 s** | 19,222 samples ≈ 152 s/epoch；20 epochs ≈ 51 min；40 epochs ≈ 101 min |
| seq2seq 峰值 RSS（含数据） | **5,444 MB** | 宿主机内存充足 |

> 上述均为**量级参考**：共享机负载高、未做多次重复；真实样本数与序列长度需以 §4.3 的正式统计为准。

### 2.8 兼容性结论矩阵（Cornac vs 本项目）

| 需求 | Cornac v3.2.0 | 差距 | 结论 |
|---|---|---|---|
| 768 维 E5 向量输入 | `FeatureModality(features, ids)`，实测 768 OK | 无 | ✅ 直接可用 |
| `product_id`（字符串）作为 item id | `FeatureModality(ids=...)` + Cornac id map；实测 `B000` 可用 | 无 | ✅ 直接可用 |
| 全目录 112,590 商品 SID | SID 只覆盖数据中出现的商品（112,565） | 25 个商品缺 SID；需 patch | ⚠️ 需适配 |
| 自定义 next-item 序列 | `USIT` + `NextItemEvaluation` | 需把 `InteractionEvent` 转成 USIT 行 | ⚠️ 需 adapter |
| 时间切分（防泄漏） | `from_timestamps` 全局时间切点 | 需定义 session 粒度 | ✅ 支持 |
| RQ-VAE 重训 | 内置，支持 rqvae/rkmeans | 无外部权重，需自训 | ✅ 可训 |
| 生成器 | T5-style seq2seq + 约束 beam + exact | 无预训练权重 | ✅ 可训 |
| SID → `product_id` | `sid_to_item` → `item_ids`，save/load 保留 | 无独立映射产物，需自己落盘 | ⚠️ 需 adapter |
| checkpoint 版本管理 | pickle + `.meta` | 无 manifest（git/向量/码本哈希） | ⚠️ 需自研 |
| 纯 CPU 运行 | 实测通过 | 计时受共享负载影响 | ✅ 可用 |
| License | Apache-2.0 | 修改/再分发需保留声明 | ✅ 合规 |

---

## 3. 其他外部实现简要对比（2026-11-24 抓取）

| 实现 | 代码/权重 | 含生成器 | License | 依赖 | TIGER 相关位置 | 结论 |
|---|---|---|---|---|---|---|
| [Cornac](https://github.com/PreferredAI/cornac) | 完整 TIGER 实现；无权重 | **有**（T5-style seq2seq + 约束 beam，从零训练） | **Apache-2.0** | Python≥3.10；`torch>=1.12`、`transformers>=4.30`；核心 `numpy>2, scipy, powerlaw` | `cornac/models/tiger/*` | 唯一完整通过审计的外部底座 |
| [EdoardoBotta/RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender) | RQ-VAE tokenizer + T5 decoder 全流程；**含 RQ-VAE 权重**（Beauty/Sports/ML32M），**无 decoder 权重** | **有**（T5 decoder，无权重） | **MIT** | Python≥3.10；`torch>=2.5.1`、`transformers>=4.47,<5`、`sentence-transformers>=3.3.1`、`torch-geometric`、`accelerate`、`gin-config`、`wandb` | `modules/rqvae.py`、`train_rqvae.py`、`train_decoder.py`、`configs/decoder_amazon.gin`、`modules/tokenizer/semids.py` | 工程最干净、检查点可复现；但数据绑 **P5 Amazon（Google Drive `P5_data.zip`）/MovieLens**，换 All_Beauty 需改数据层；tokenizer 权重不可复用 |
| [XiaoLongtaoo/TIGER](https://github.com/XiaoLongtaoo/TIGER) | 论文非官方 PyTorch 复现；有 Beauty RQ-VAE ckpt；**无 T5 权重**（`model/ckpt/None`） | **有**（T5 seq2seq，无权重） | **MIT** | **Python 3.7.16、torch 1.10.1、transformers 4.30.2、numpy 1.21.6** | `rqvae/`、`model/main.py`、`rqvae/generate_code.py` | 依赖过旧、与项目 Python 3.10/torch 2.x 冲突；`generate_code.py` 硬编码 `cuda:0`；只能作 recipe 参考 |
| [datawhalechina/torch-rechub](https://github.com/datawhalechina/torch-rechub) | TIGER + RQ-VAE 实现；**无权重**（T5 从零训练） | **有**（HF T5 从零） | **MIT** | Python≥3.10；`torch>=2.6,<3`；`transformers>=4.57.3,<5`（`generative` extra） | `torch_rechub/models/generative/tiger.py`、`rqvae.py`、`docs/zh/blog/tiger_reproduction.md` | **数据接口最接近本项目**：`inter.json`（user→item id 序列）+ `semantic_ids.json`（item→SID 字符串），Trie 约束 beam；可作 A2/备选/交叉校验，但需自备 RQ-VAE 与 E5 |
| [phonism/genrec](https://github.com/phonism/genrec) | TIGER 论文对齐 recipe；**无权重** | **有**（自研 T5 模块，从零） | **MIT** | Python≥3.9；`torch>=2.0`、`transformers>=4.40`、`gin-config`、`sentence-transformers`、`wandb` | `genrec/models/tiger.py`、`config/tiger/amazon/*.gin` | 模型 zoo，配置与数据管线最重；Amazon 2014 口径，社区较小；适合“第二意见”基线 |
| [snap-research/GRID](https://github.com/snap-research/GRID) | 生成式推荐框架（TIGER 与 RQ-KMeans/RVQ/RQ-VAE） | 有（TIGER 模块，无权重） | **Snap 非商业研究许可** | `torch==2.6.0+cu124`、`lightning`、`hydra` 等，锁 CUDA | — | 许可证禁止商业用途、依赖锁 CUDA；仅备查（沿用既有核查结论，本审计未重新运行） |

补充事实：

- **没有任何 TIGER 端到端公开权重**；Cornac README 明确 “There is no official code release”。唯一可下载的
  tokenizer 权重（Botta/XiaoLongtaoo/HF `edobotta/rqvae-amazon-beauty`）都绑定别的商品集合与 embedding 空间，
  维度同为 768 ≠ 空间相同，**不可复用**（沿用 `docs/audit/tiger-compatibility-report.md` 的结论）。
- Botta 的 `pyproject.toml` 版本为 1.0.1，但 tag 为 `v1_0_2`（HEAD 957d32bda439，2026-08-31），版本号与 tag
  不完全一致；使用时应以 commit 为准。
- torch-rechub 的 `tiger.py` 已针对 `transformers>=5` 的 `model_parallel` 属性缺失做兼容；但它的
  `generative` extra 仍要求 `transformers>=4.57.3,<5`。

---

## 4. 本项目数据接口核验（只读）

> 本节是 Phase4 §2.5 Step 1 的“商品向量 / Reviews / 序列统计”交付证据；所有读取均只读，未修改数据。

### 4.1 All_Beauty E5 商品向量

数据：`/mnt/agr-data/artifacts/faiss/e5-base-v2_all_beauty_v2/embeddings.npy`（只读）。

| 检查 | 结果 |
|---|---|
| 形状 / dtype | **(112,590, 768) float32**（345,876,608 B，含 npy header） |
| 数值 | `np.isfinite(X).all() == True`；零向量行 **0** |
| 归一化 | 每行 L2 范数 **≈1.0000**（打印值 mean/min/max 均为 1.0000），与 FAISS 内积索引一致 |
| 行对齐 | `product_ids.jsonl` 112,590 条、unique 112,590；与 `catalog/all_beauty_v2/products.jsonl` **集合相等、顺序一致** |
| catalog | 112,590 个 `product_id`；manifest 记录 source=`parent_asin`、`checksum_sha256` |

结论：**E5 向量可直接作为 Cornac `FeatureModality(features=X, ids=product_ids)` 的输入**（768 维、行对齐、
已归一化）；不需要重算。

### 4.2 SID 碰撞与消歧层（对 A1 3×256 的关键估计）

用与 Cornac `rkmeans` 相同的残差 k-means（3 层 × 256，`_kmeans`，10 iter）在真实 E5 向量上做**零训练估计**：

| 指标 | 结果 |
|---|---|
| 每层码本使用 | 256/256（dead code 0） |
| 唯一 3 层 SID 数 | **81,709 / 112,590（72.6%）** |
| 碰撞冗余数 Σ(bucket−1) | **30,881（27.4%）**（不等于“碰撞商品数”；桶>2 时后者更大） |
| 最大碰撞桶 | **86**（需要 86 个消歧位，即 SID 第 4 层 vocab=86） |
| 耗时 | 114.2 s（含 346MB 读取 + 3 层 k-means + 统计） |

与并行 pilot 的 RQ-VAE 结果对照（`artifacts/semantic_ids/rqvae_pilot_a1/semantic_id_manifest.json`，只读）：

| 指标 | RQ-VAE 3×256（pilot） |
|---|---|
| 唯一 SID | 100,702 / 112,590（89.4%） |
| 碰撞商品 | 19,708（**17.5%**） |
| 最大碰撞桶 | 25（消歧层 vocab=25） |
| 码本使用率 / dead codes | 100% / 0；98.83% / 3；97.66% / 6 |
| 码本 perplexity | 234.0 / 243.0 / 241.3 |

**结论**：

1. A1（3×256）在真实 E5 上**必然发生大量的前缀碰撞**（rkmeans：仅 72.6% 的 SID 唯一；RQ-VAE：89.4% 唯一，
   17.5% 商品落在碰撞桶中）。
2. 必须采用 **TIGER 式第 4 层消歧后缀**（Cornac 自动追加，pilot 的 `SemanticIdTable` 目前**没有**追加）或
   由负责人书面接受“桶内多商品 + popularity tie-break”的歧义解码；否则“Semantic ID → 唯一 product_id”
   这一验收项无法成立。
3. 若采用消歧层，生成器词表要从 `3×256` 扩到 `3×256 + 25`（RQ-VAE）或 `+86`（rkmeans），**生成器必须按新
   SID 重新训练**；这不是后处理能解决的。

### 4.3 Reviews 与 InteractionEvent 序列

数据（2026-11-24 服务器只读检查）：

| 项 | 值 |
|---|---|
| raw reviews | `/mnt/agr-data/data/raw/review_All_Beauty.jsonl`，**326,611,506 B**，**701,528 行** |
| raw metadata | `meta_All_Beauty.jsonl`，**212,990,142 B**，112,590 条（manifest `record_count=112590`） |
| reviews 关键字段（首条实测） | `rating/title/text/images/asin/parent_asin` + 用户/时间字段（`user_id`、`timestamp` 为毫秒，接入层已统一为秒） |
| SQLite | `/mnt/agr-data/data/processed/interaction_history.db`，180 MB，表 `interaction_events` |
| 事件数 / 用户数 / 商品数 | **694,252 / 631,986 / 112,565** |
| 时间范围 | 973,052,658 – 1,694,219,976（Unix 秒，2000-10 至 2023-09） |
| 事件类型 | `review` 693,653；`rating` 599；`is_synthetic=false`（字段约束见 schema） |
| 用户序列长度 | mean 1.10；median 1；P90 1；max 165 |
| 用户数（≥3 / ≥2 事件） | **8,201 / 43,044** |
| 序列样本（pilot `samples_a1/sequence_manifest.json`，只读引用） | 8,201 用户 → **19,222 样本**；train 13,710 / val 2,827 / test 2,685；skipped short 16,402；`max_history_items=10`；切点 `train_ts=1,626,820,406`、`val_ts=1,648,574,591` |

**解释**：All_Beauty 用户绝大多数只有 1 条 review（mean 1.10），因此 TIGER 可用序列只有 8,201 个用户
（≥3 事件）；这与 Phase4 方案中的“8,201 条长度至少为 3 的序列”一致。**这是本项目 TIGER 样本量的硬上限**，
正式训练/评估必须以这 8,201 个用户的时间切分为准，不能把 631,986 个用户都算作训练序列。

### 4.4 数据接口结论

- 商品侧：E5 112,590×768 已对齐，可直接进 RQ-VAE；`product_id` 即 item id。
- 用户侧：`InteractionEvent`（SQLite）→ 按 `user_id` 时间排序 → `USIT`（session 建议取 `user_id`，或按
  时间间隔切会话）→ `from_timestamps` 全局时间切分。
- 商品域：catalog 112,590 vs 有交互 112,565（25 个冷商品）；Cornac 默认 SID 只覆盖后者。
- 泄漏边界：SID tokenizer 用**商品内容向量**（全目录）训练，不涉及用户行为；测试期用户的未来交互不得进入
  历史序列。RQ-VAE 在测试商品内容上训练属于 TIGER/GRS 的常规 transductive 设定，**必须在实验记录中声明**。

---

## 5. 适配清单（哪些必须自研）

> 下表同时适用于两条路线：**C = Cornac 底座**，**A1 = 现有自研实现**。每项给出“复用/自研”边界。

| # | 能力 | Cornac 现状 | 本项目需求 | 适配动作 | 验收 |
|---|---|---|---|---|---|
| 1 | **E5 768 维输入** | `FeatureModality(features=X, ids=product_ids)` 直接可用；要求行数 ≥ 全局商品数 | 112,590×768、已归一化、行序与 catalog 对齐 | 无代码改动；在 `configs/tiger/` 记录 E5 checkpoint + `embeddings_sha256` + `product_ids_sha256` | 抽样核对行对齐；RQ-VAE 输入 `input_dim=768` |
| 2 | **`product_id` ↔ 内部索引** | `iid_map`（raw→idx）、`item_ids`（idx→raw）可直接用；save/load 后保留 | 稳定、可落盘、可审计 | 写 `product_index.json`（product_id、内部 idx、SID、SID 桶大小）；由 `TIGER.item_ids`/`sid_to_item` 生成 | 112,590 行唯一；随机 1,000 个 SID 反查 100% 命中 |
| 3 | **`InteractionEvent` → 序列** | 需要 `USIT` 行；`from_timestamps` 提供全局时间切分 | SQLite 事件按用户时间排序；train/val/test 时间切分；序列长度统计 | 写 `interaction_sequences.py` 适配（现有 `sequences.py` 属自研实现，不复用该文件名）：`(user_id, session_id, product_id, timestamp)`；session=user 或时间间隔切分；`min_events=3`、`max_len` 由分位数定（建议 10/20 两档对照） | 每个用户序列时间单调；无未来事件；样本数与长度分布写入 manifest |
| 4 | **Semantic ID 解码** | `sid_to_item` + `item_ids`；`generate_beam` 返回完整 SID；非法 SID 不进入结果 | 输出 `product_id` + rank；碰撞桶消歧 | 写 `sid_decode.py`：SID→idx→product_id；碰撞桶按“生成分数 + 确定性 tie-break”返回多个候选并标注 `collision_bucket_size`；**必须统计解码成功率/非法率/重复率** | Top-K 无重复；非法 SID=0（约束 beam）；碰撞桶全部可追溯 |
| 5 | **全目录 SID（112,590）** | 仅覆盖数据中出现的商品（112,565）；需 subclass 覆写 `_get_item_features`/`_build_semantic_ids` 或在训练前注入全目录 ids | TIGER 应能解码到 catalog 全部商品（至少不丢 25 个冷商品） | 方案 A：接受 112,565（文档声明 25 个冷商品永不由 TIGER 返回）；方案 B：在 seq2seq 训练前用全目录 codes 构建 `sid_table`/Trie（需重训生成器，消歧层 vocab 可能变化） | 明确二选一；若 B，`sid_table` 112,590 行且解码覆盖 100% |
| 6 | **checkpoint / manifest** | pickle + `.meta`；无 git/向量/码本/配置版本 | Q10：权重、SID 表、映射放 NFS；可复现 | 写 `checkpoint.py`：`{run_id, git_commit, e5_ckpt, embeddings_sha256, rqvae_ckpt_sha256, sid_manifest_sha256, config, seed, dedup_size, timestamp}`；目录 `semantic_ids/<run_id>/` | 从 manifest 可完整复现；ckpt 可 load 并复算同一分数 |
| 7 | **约束 beam / 合法率** | `prefix_children` Trie 约束；`n_beams ≥ K` | 生成 SID 合法率接近 100%，Top-K 可复现 | 直接复用 Cornac 的 `generate_beam`；自研路线需补 Trie/level-mask（当前 `generate_ids` 未做 level 掩码，可能产出跨层非法 token） | 同一输入两次解码一致；非法/重复 SID 计数=0 |
| 8 | **TIGER → Fusion 接口** | 无项目适配 | 只输出 `product_id + rank + (诊断用 score)`；不参与跨模型原始分数相加 | 写 `tiger_service.py`：`rank(history_product_ids, k) -> [(product_id, rank, score?)]`；Fusion 只用 rank 做 RRF（AGENTS §2.5） | 单测：两路排名 RRF 可复现；TIGER 分数不进入加权 |
| 9 | **时间切分评测** | `from_timestamps` + `semantic_id_metrics=True` | Recall@K/NDCG@K/MRR + SID 指标；不得用 leave-last-out 冒充正式结果 | 复用 Cornac 评测；自研脚本需对齐同一测试集与同一目标商品口径 | 指标可复算；与 Cornac 的同一测试集结果对照（差异可解释） |
| 10 | **可复现/记录** | 无 run manifest | `run_id`、git commit、数据快照、种子、耗时、内存 | 统一 manifest 格式（Phase4 方案 §8 的 run 记录要求） | 每个 pilot 产物都有 manifest |

### 5.1 若以 Cornac 为底座的建议代码结构

```text
src/recommender/retrieval/tiger/
├── cornac_adapter.py   # E5 → FeatureModality；InteractionEvent → USIT；from_timestamps；TIGER fit
├── product_index.py    # product_id ↔ idx ↔ SID ↔ 碰撞桶；落盘/加载
├── sid_decode.py       # generate_beam → SID → product_id；非法/重复/碰撞统计
├── checkpoint.py       # 训练配置 + 哈希 + run manifest（Q10：NFS）
└── tiger_service.py    # 对 Fusion 暴露 rank 接口（只传排名）
```

适配层要点：

1. `FeatureModality(features=embeddings, ids=product_ids)`：**不要**先按出现商品裁剪矩阵；Cornac 会按 `iid_map`
   重排（内部 `np.copy` 约 346 MB，CPU 可接受）。
2. `USIT` 的 `session` 建议直接取 `user_id`（每用户一个会话），这样 `from_timestamps` 的“会话原子”就是
   “用户原子”，不会把同一用户的时间线截断到两个 split；若要按时间间隔切会话，需先固定 gap 规则并写入 manifest。
3. `TIGER.item_ids` 的顺序 = `iid_map` 的 keys 顺序 = SID 行序；**落盘 `product_index.json` 时必须固定该顺序**
   （JSON 保持插入序），否则 SID 会错位。
4. `TIGER.save()` 是整模型 pickle；生产上不要把 pickle 当唯一产物，至少同时保存 `state_dict`（rqvae/model）、
   `sid_table.npy`、`product_index.json` 与 `manifest.json`，便于跨版本恢复与审计。
5. 若要“全目录 SID”，最小侵入做法是 **subclass TIGER**：`fit` 前把 `self.total_items = len(catalog)`，用
   `rqvae.encode`/`_kmeans` 对全目录编码后调用 `_build_semantic_ids`，再 `_fit_seq2seq`；同时 `enc_token_table`
   按全目录 SID 构建。该 patch 会改变消歧层 vocab（§4.2），必须重训生成器并重新评测。

### 5.2 若沿用现有 A1 实现（推荐）：现有模块与需补项

仓库现状（2026-11-24 抓取时点；审计期间并行出现的实现，非本任务产出；本审计未复跑其 pilot）：

| 模块 | 已实现 | 与本计划关系 |
|---|---|---|
| `src/recommender/retrieval/tiger/rqvae.py` | RQVAE + `train_rqvae`/`load_rqvae`/`encode_vectors`/manifest/metrics | 对应 §5 #1/#5；已全目录编码 112,590 |
| `.../semantic_ids.py` | `SemanticIdTable`（product_id↔3 层 codes、碰撞桶、save/load） | 对应 #2/#4；**缺第 4 层消歧后缀** |
| `.../sequences.py` | 全局时间分位切分、`TigerSample`、样本落盘/加载 | 对应 #3 |
| `.../generator.py` | token-level causal Transformer、`generate_ids`、`rank_products_for_ids`、时间切分评测 | 对应 #7/#9；**缺 level-mask/Trie 约束解码** |
| `configs/tiger/` | pilot 配置（A1） | 对应 #10 |

需补的 3 项（估计 4–6 人时，未实测）：

1. **消歧层**：在 `codes` 后追加 dedup 列（RQ-VAE 最大桶 25 / rkmeans 86），生成器 vocab 从 768 扩到
   `768+dedup`，重训生成器；或由负责人书面接受桶内歧义方案。
2. **约束解码**：`generate_ids` 每步只允许当前 level 的 token 范围 + 合法前缀 Trie；输出非法率统计。
3. **Fusion 适配 + 端到端验收脚本**：`rank -> product_id`、与 E5 的两级 RRF 单测、时间切分
   Recall/NDCG/MRR、SID 合法率/解码覆盖率报告。

另外两个观察（不阻塞 A1，但应在 pilot 报告中记录）：

- `SemanticIdTable.semantic_id_of()` 用 `list.index` 线性扫描（O(112,590)/次）；批量构建序列时会较慢，建议
  在适配层加 `dict[product_id] -> code` 索引（pilot 已完成，说明规模可接受，但推理/评测应避免）。
- `GeneratorMetrics` 有 `best_epoch/best_val_loss` 字段，但 `train_generator` 当前不更新它们（保存末轮权重，
  无 best-on-val 选择）；若后续需要“按验证集保存最优”，要显式实现。

### 5.3 复用 vs 自研边界

| 可复用（Cornac/公开代码） | 必须自研（本项目特有） |
|---|---|
| RQ-VAE / rkmeans 结构、T5-style seq2seq、约束 beam、exact scoring | E5→RQ-VAE 的 112,590 目录接入与 manifest |
| `NextItemEvaluation`（USIT、`from_timestamps`、`semantic_id_metrics`） | `InteractionEvent` → 时间序列（8,201 用户、19,222 样本口径） |
| `FeatureModality`、`Recommender.save/load`、`item_ids/sid_to_item` | `product_id ↔ idx ↔ SID ↔ 碰撞桶` 映射产物与反查 |
| SID-Recall/NDCG/CCE 指标实现 | 全目录 SID（25 个冷商品）策略与文档化 |
| Apache-2.0 允许的修改/再分发（保留声明） | checkpoint/manifest（Q10）、Fusion rank 接口、验收脚本与测试 |

---

## 6. 结论与建议

### 6.1 底座选择

| 选项 | 内容 | 优点 | 代价/风险 | 建议 |
|---|---|---|---|---|
| **A. 自研 A1（推荐）** | 沿用 `src/recommender/retrieval/tiger/` 现有实现，补 §5.2 三项 | 已覆盖全目录 112,590；与项目 schema/存储/测试直接集成；pilot 产物齐全；无新增依赖 | 缺消歧层/约束解码；生成器不是论文 T5 结构（可比性弱） | **作为工程底座；Cornac 作独立校验** |
| **B. Cornac + 适配层** | 以 Cornac v3.2.0 为运行时底座，按 §5.1 写 adapter | 论文对齐结构、约束 beam、exact scoring、SID 指标现成；Apache-2.0 | 需 patch 全目录 SID；新增 scipy/matplotlib/powerlaw；pickle ckpt 无版本；约 6–10 人时适配（未实测）；与现有 pilot 并行会产生双实现 | 仅当负责人坚持“外部代码底座”时采用 |
| **C. 双实现并行** | A 生产 + B 离线对照 | 交叉验证最强 | 120 人时预算下不可持续 | 只做“同一测试集、同一 SID 表”的一次性对照，不长期维护 |

**推荐结论**：

1. **工程底座**：选项 A（自研 A1），因为 Q1 的 B→A 允许“复用结构/流程 + 自训练”，而仓库内实现已经
   覆盖 RQ-VAE → SID → 时间序列 → 生成 → 解码 → manifest 全链路，且对全目录 112,590 个商品做了编码。
2. **Cornac 的定位**：**外部参考实现 + 独立校验基线**（不作为默认运行时依赖）：
   - 用 Cornac 的 `from_timestamps` + `semantic_id_metrics` 在同一 USIT 数据上跑一次对照，验证自研
     序列构造、SID 表与评测口径；
   - 用 Cornac 的 `scoring="exact"` 对自研 beam 的 Top-K 做一致性抽查（Cornac 与自研模型权重不可直接共享；
     只能对齐到同一数据/同一测试集，比较实现级指标、合法率与解码覆盖，差异需逐项解释）。
3. **若负责人选择 B**：按 §5.1 实施，并把“全目录 SID patch”作为第一个里程碑（否则 25 个冷商品永远不可召回，
   且 `SemanticIdTable`/Cornac 的 SID 行数不一致会导致 Fusion 对不上）。
4. **不论 A/B**：先完成 §5 #1–#4（数据契约、映射、序列、解码），再做 Step 2/3 的小规模训练；**Step 1 未闭环前
   不启动全量训练**（Phase4 §2.5 要求）。

### 6.2 为什么不是“直接复用外部权重”

- 没有任何 TIGER 生成器权重；RQ-VAE tokenizer 权重只覆盖 Amazon 2014 Beauty/Sports、MovieLens 等，商品集合与
  embedding 空间都不同；维度同为 768 不代表可迁移。
- TIGER decoder 的 token 空间 = 各层 codebook + 消歧层；换 codebook 必须重训。**结论：权重不可复用，代码可复用。**

### 6.3 下一步（与 Phase4 方案 §2.5 Step 2–4 对齐）

| 步骤 | 动作 | 产出 | 预算（估） |
|---|---|---|---|
| Step 2a | 固定 A1 配置（3×256、768、dedup 层、max_len、seed），跑 RQ-VAE 小规模/全目录 | 重构误差、码本使用率/dead codes、SID 唯一率、映射覆盖率、训练时间/内存/ckpt 大小 | 0.5–1 人时（CPU 实测 RQ-VAE 100 epochs ≈ 3 min 量级） |
| Step 2b | 构建 8,201 用户时间序列（train 13,710 / val 2,827 / test 2,685） | 序列 manifest（切点、长度分布、跳过原因） | 0.5 人时（已有 pilot 产物可复用） |
| Step 3 | 生成器小规模训练（T5 或 token-level 均可，记录结构/参数/超参）+ 约束解码 | Recall/NDCG/MRR、SID 合法率、解码成功率、覆盖、延迟 | 1–2 人时（CPU 20 epochs 约 1 小时量级，取决于样本数） |
| Step 4 | Pilot 评审（Phase4 §2.5 Step 4 的 7 个问题） | `docs/phase4/TIGER_PILOT_REPORT.md` | 0.5 人时 |
| 对照 | Cornac 在同一 USIT 数据上跑一次（可选，若选 A 则只做指标对照） | 对照表 + 差异解释 | 1–2 人时 |

> 时间预算来自本报告 §2.7 的微基准与现有 pilot 产物，**不是**正式训练承诺；全量训练需按实际样本数外推。

---

## 7. 风险与未验证项

| ID | 风险 / 未验证项 | 现状与证据 | 影响 | 验证/缓解 | 状态 |
|---|---|---|---|---|---|
| R1 | 无 TIGER 端到端公开权重 | 全部候选仓库无 TIGER decoder 权重；Cornac README 明确无官方代码 | 必须自训；不能“加载即用” | 按 A3 重训；保留检索记录 | 基本证伪（沿用既有结论） |
| R2 | 训练成本超预算 | CPU 微基准：RQ-VAE 1.7s/epoch、seq2seq 39.6s/5k samples；pilot 实测 RQ-VAE 100 epochs 77s、generator 40 epochs 108s | 全量/多 seed 成本需实测 | Step 2/3 记录时间/内存；先 A1 后全量 | 部分验证（量级） |
| R3 | SID 碰撞导致解码歧义 | E5 上 rkmeans 3×256：仅 72.6% SID 唯一、Σ(bucket−1)=27.4%、最大桶 86；RQ-VAE pilot：89.4% 唯一、17.5% 商品在碰撞桶、最大桶 25 | “唯一映射”验收不成立；桶内排序影响指标 | 增加第 4 层消歧后缀并重训生成器；或书面接受桶内 tie-break | Open（关键） |
| R4 | 全目录 SID 缺 25 个冷商品 | catalog 112,590 vs DB 112,565；Cornac 默认 SID=数据域 | 25 个商品永不由 TIGER 召回（E5 仍可） | 方案 A 接受并文档化；方案 B 全目录 patch | Open（需决策） |
| R5 | transformers 版本差异 | 5.19.0 可跑但有 `eos_token_id` warning；4.57.6 干净通过；项目 pin `<5` | 未来升级可能破坏 T5Stack（半内部 API） | 锁 `transformers>=4.57,<5`；升级前跑 toy 回归 | 已验证（4.57.6/5.19.0） |
| R6 | 依赖膨胀/环境漂移 | Cornac 引入 scipy/matplotlib/powerlaw；`numpy>2`；venv 438MB | 与现有 Phase 1/2 环境合并时增加风险 | 单独 venv/或仅作离线对照；bootstrap 脚本固定版本 | 中 |
| R7 | checkpoint 不可审计 | Cornac pickle + `.meta` 无 git/向量/码本版本；自研 ckpt 也无统一 manifest | 复现困难、跨版本不可加载 | §5 #6 manifest；至少保存 state_dict + sid_table + product_index | Open |
| R8 | 时间切分/序列泄漏 | 已用 `from_timestamps`（会话原子）；自研用全局分位切点；需保证目标不在历史 | 指标虚高 | 单测：每个用户历史时间严格 < 目标；测试期目标不出现在历史 | 设计已覆盖，待测试 |
| R9 | 生成 SID 非法/跨层 token | Cornac 有 Trie；自研 `generate_ids` 当前只做 top-k，可能产生跨层 token 再被丢弃 | 合法率/召回下降 | 自研补 level-mask + Trie；报告非法率 | Open（自研路线） |
| R10 | 共享机 CPU 计时不稳定 | loadavg≈60，128 核共享；单次测量 | 预算估算偏差 | 多次重复、报告 P50；优先低负载时段 | 已知 |
| R11 | 商品主键粒度 | 项目 `product_id=parent_asin`；外部实现是 asin/内部 idx | 若接入层聚合有误，SID 与 E5 行错位 | 已有 catalog manifest + 112,590 行对齐核验；序列构建复用同一映射 | 已缓解（数据层） |
| R12 | 数据许可 | Amazon Reviews 2023 数据集 `license: other`；`edobotta` 模型卡未声明 license | 再分发受限 | 不随项目分发数据/权重；仅在 NFS 内部使用 | 已知 |
| R13 | Cornac TIGER 无专属单测 | v3.2.0/master 无 `test_tiger*.py`；仅 SID adapter/metrics 测试 | 回归风险 | 若采用 B，自建最小回归测试（toy 数据固定 seed） | 中 |
| R14 | Cornac SID 商品域 = 数据域 | toy 实证 80 特征 → 71 SID；`total_items=len(iid_map)` | 全目录/冷商品问题 | §5 #5 决策 | 已验证（限制存在） |
| R15 | A1 是否达到验收 | 本审计未复跑 pilot；pilot 指标需按 Step 2/3 验收口径评审 | Phase 4 不能仅凭“能跑”验收 | 按 §6.3 Step 4 的 7 问评审 | 未验证（下一步） |

---

## 8. 证据链接（2026-11-24）

**Cornac**

- 仓库：https://github.com/PreferredAI/cornac
- Release v3.2.0：https://github.com/PreferredAI/cornac/releases/tag/v3.2.0
- TIGER 源码（tag）：https://github.com/PreferredAI/cornac/tree/v3.2.0/cornac/models/tiger
  - `tiger.py`：https://github.com/PreferredAI/cornac/blob/v3.2.0/cornac/models/tiger/tiger.py
  - `recom_tiger.py`：https://github.com/PreferredAI/cornac/blob/v3.2.0/cornac/models/tiger/recom_tiger.py
  - README：https://github.com/PreferredAI/cornac/blob/v3.2.0/cornac/models/tiger/README.md
  - requirements：https://github.com/PreferredAI/cornac/blob/v3.2.0/cornac/models/tiger/requirements.txt
- 示例：https://github.com/PreferredAI/cornac/blob/v3.2.0/examples/tiger_example.py
- 评测/时间切分：https://github.com/PreferredAI/cornac/blob/v3.2.0/cornac/eval_methods/next_item_evaluation.py
- 基类（save/load、item_ids）：https://github.com/PreferredAI/cornac/blob/v3.2.0/cornac/models/recommender.py
- FeatureModality：https://github.com/PreferredAI/cornac/blob/v3.2.0/cornac/data/modality.py
- Amazon loader（仅 2014）：https://github.com/PreferredAI/cornac/blob/v3.2.0/cornac/datasets/amazon_review.py
- PyPI：https://pypi.org/project/cornac/3.2.0/
- tag commit：https://github.com/PreferredAI/cornac/commit/bb3b42d425f5d44079c001ffd0c197a1a7c2a6f9
- master HEAD：https://github.com/PreferredAI/cornac/commit/8923b8b0074b38d20a28c38df31a2bd83249ca03

**其他实现**

- Botta：https://github.com/EdoardoBotta/RQ-VAE-Recommender
  - `train_decoder.py`：https://github.com/EdoardoBotta/RQ-VAE-Recommender/blob/main/train_decoder.py
  - `modules/rqvae.py`：https://github.com/EdoardoBotta/RQ-VAE-Recommender/blob/main/modules/rqvae.py
  - decoder 配置：https://github.com/EdoardoBotta/RQ-VAE-Recommender/blob/main/configs/decoder_amazon.gin
  - HEAD：https://github.com/EdoardoBotta/RQ-VAE-Recommender/commit/957d32bda43958ba641ae919abb1092ac6738798
- XiaoLongtaoo：https://github.com/XiaoLongtaoo/TIGER
  - `model/main.py`：https://github.com/XiaoLongtaoo/TIGER/blob/main/model/main.py
  - `rqvae/generate_code.py`（硬编码 cuda:0）：https://github.com/XiaoLongtaoo/TIGER/blob/main/rqvae/generate_code.py
  - requirements：https://github.com/XiaoLongtaoo/TIGER/blob/main/requirements.txt
- torch-rechub：https://github.com/datawhalechina/torch-rechub
  - TIGER：https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/generative/tiger.py
  - RQ-VAE：https://github.com/datawhalechina/torch-rechub/blob/main/torch_rechub/models/generative/rqvae.py
  - 中文复现说明：https://github.com/datawhalechina/torch-rechub/blob/main/docs/zh/blog/tiger_reproduction.md
- genrec：https://github.com/phonism/genrec
  - TIGER：https://github.com/phonism/genrec/blob/main/genrec/models/tiger.py
  - 配置：https://github.com/phonism/genrec/blob/main/config/tiger/amazon/tiger.gin
- GRID（备查，未重跑）：https://github.com/snap-research/GRID
- 论文：TIGER https://arxiv.org/abs/2305.05065 ｜ GRID https://arxiv.org/abs/2507.22224 ｜
  PAISCHER/Mender https://arxiv.org/abs/2412.08604
- 既有核查：`docs/audit/tiger-compatibility-report.md`（HF 权重/数据集检索，2026-11-24）

**项目内**

- 决策与风险台账：`docs/decisions/FINAL_DECISIONS_AND_RISK_REGISTER.md`（Q1/Q10/Q11/Q12）
- ADR：`docs/decisions/ADR-TIGER-RETRAINING-ROUTE.md`
- Phase4 方案：`docs/decisions/Phase4_TIGER%E5%AE%9E%E6%96%BD%E6%96%B9%E6%A1%88%E4%B8%8EPhase5-7%E9%9B%86%E6%88%90%E8%AE%A1%E5%88%92_V1.0 (1).md`
  §2.2/§2.5/§8（文件名含 URL 编码，按实际文件读取）
- 数据/产物（NFS，只读）：
  - E5：`/mnt/agr-data/artifacts/faiss/e5-base-v2_all_beauty_v2/{embeddings.npy,product_ids.jsonl,manifest.json}`
  - RQ-VAE pilot：`/mnt/agr-data/artifacts/semantic_ids/rqvae_pilot_a1/`
  - 序列样本：`/mnt/agr-data/artifacts/semantic_ids/samples_a1/`
  - 生成器 pilot：`/mnt/agr-data/artifacts/semantic_ids/generator_pilot_a1/`
  - SQLite：`/mnt/agr-data/data/processed/interaction_history.db`

---

## 附录 A：服务器命令与原始输出摘录

> 全部命令在 `/tmp` 临时 venv/脚本中执行；未修改项目与系统环境。SSH 凭据在仓库外，未写入本文。

### A.1 venv 与安装

```bash
python3 -m venv --without-pip --system-site-packages /tmp/tiger_audit_venv
/tmp/tiger_audit_venv/bin/python -V                      # Python 3.10.12
/tmp/tiger_audit_venv/bin/python -m pip -V               # pip 26.2.1（user-site）
/tmp/tiger_audit_venv/bin/python -m pip install --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple cornac==3.2.0
```

```text
Successfully installed cornac-3.2.0 contourpy-1.3.2 cycler-0.12.1 fonttools-4.65.0
  kiwisolver-1.5.1 matplotlib-3.10.9 pillow-12.3.0 powerlaw-2.0.0
  pyparsing-3.3.3 scipy-1.15.3
# 已有：numpy 2.2.6, torch 2.14.1+cpu, tqdm 4.70.1
```

```bash
# transformers 版本兼容复测（项目 pin <5）
/tmp/tiger_audit_venv/bin/python -m pip install --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple 'transformers==4.57.6'
# 安装后：transformers 4.57.6 / torch 2.14.1+cpu / cornac 3.2.0
# 重跑 /tmp/tiger_smoke.py → fit 2.15s，save/load 与分数复现通过
```

### A.2 import / 参数计数

```text
cornac 3.2.0 | torch 2.14.1+cpu | transformers 5.19.0
TIGER import OK; GRID keys: [...]
RQVAE params: 1149472
Seq2Seq params: 4657280          # level_sizes=[256,256,256,2]
FeatureModality OK dim= 768
```

### A.3 toy 端到端（合成数据，非效果实验）

```text
train sets: num_items=62 num_users=21 num_sessions=21
Semantic IDs assigned: 71 items, 8 collisions, dedup level size 3
fit seconds 2.88
sid_table shape (71, 4) level_sizes [16,16,16,3]
total_items 71 pad_idx 71
score shape (71,) top5 [36,29,14,9,59]
TIGER model is saved to /tmp/tiger_ckpt_.../TIGER/....pkl
ckpt size_MB 0.3
load OK: TIGER sid_table (71, 4) scoring beam
score reproducible: True
```

### A.4 时间切分 + RQ-VAE + exact

```text
time-split: train items 24 users 6 | test items 24
rqvae+seq2seq fit seconds 2.1
sid_table (24, 4) levels [16,16,16,2] collisions 2
exact scoring sid_table (24, 3) max score idx 23
```

### A.5 CPU 微基准

```text
torch threads: 64
RQVAE params: 1149472
kmeans_init_codebooks(112590x768) seconds: 5.4
RQVAE 1 epoch (112590 items, batch1024) seconds: 1.7
=> 20 epochs est minutes: 0.6
peak RSS MB: 1190.0
TIGERSeq2Seq params (4+4, d128): 4658816
Seq2Seq 1 epoch (5000 samples, batch256, 80 tokens) seconds: 39.6
=> 8201 samples x 20 epochs est minutes: 21.7
peak RSS MB end: 5444.0
```

> 基准脚本用 `level_sizes=[256,256,256,8]`（仿真消歧层 8）测 seq2seq 参数与速度；脚本中“8201 samples”
> 是 TIGER 可用序列数（§4.3），而现有 pilot 实际构造了 19,222 个 next-item 样本，因此正文 §2.7 按 19,222
> 外推（约 152 s/epoch）。

### A.6 save/load 与 SID→product_id

```text
total_items 32 iid_map len 32 item_ids len 32
sid0 [6, 2, 4, 0] -> idx 0 -> product B000
loaded total_items 32 iid_map len 32 sid_table (32, 4) sid_to_item 32
loaded product of sid0: B000
score shape (32,) == total_items True
```

### A.7 全目录 SID 构建与 checkpoint 大小

```text
build sid/trie seconds 0.87
sid_table (112590, 4) level_sizes [256, 256, 256, 3] collisions 399 max dedup 2
seq2seq params 4657536
pickled model size MB 31.6
prefix_children levels 4 keys per level [1, 256, 53665, 112191]
```

> 该 31.6MB 使用随机 codes（消歧层仅 3）；真实 RQ-VAE 消歧层 25（pilot）或 rkmeans 86，ckpt 预计略大，量级 < 40MB。

### A.8 真实数据只读核验

```text
embeddings shape (112590, 768) dtype float32
finite: True | zero_rows: 0
L2 norm mean 1.0000 min 1.0000 max 1.0000
product_ids 112590 unique 112590
catalog products 112590 unique 112590
ids match catalog order: True | set equal: True
rkmeans 3x256: unique SID tuples 81709 / 112590 | collisions 30881 | max bucket 86
DB: events 694252 | users 631986 | products 112565
ts range (973052658, 1694219976)
user seq len: n=631986 mean=1.10 median=1 p90=1 max=165
users>=3 8201 | users>=2 43044
review file: 326611506 B / 701528 lines
```

### A.9 清理与可复现说明

- 临时 venv：`/tmp/tiger_audit_venv`（438 MB，临时；重启即失，可重建）。
- 临时脚本：`/tmp/tiger_smoke.py`、`/tmp/tiger_smoke2.py`、`/tmp/tiger_bench.py`、`/tmp/tiger_ckpt_size.py`、
  `/tmp/tiger_map_check.py`、`/tmp/tiger_e5_check.py`（均不在项目内）。
- 未修改 `/home` 下项目文件；未写 NFS 大文件；未动其他队友进程。

---

## 附录 B：Cornac README 口径与本次实测的差异

| 项 | README/文档口径 | 本次实测 | 处理 |
|---|---|---|---|
| 默认 seq2seq 参数量 | “~13M” | **4,657,536**（4+4 层、d_model=128、d_ff=1024、level_sizes=[256,256,256,3]；[256,256,256,2] 为 4,657,280） | 以实测为准；配置/报告写实测值 |
| 训练时长 | “约 1 小时/单 GPU”（Beauty 2014，22k 商品） | CPU 未跑全量；微基准见 A.5 | 不引用 GPU 时长做 CPU 预算 |
| Amazon loader | docstring 称 2014/2018/2023 | `_validate` 仅 2014 | 不要用内置 loader |
| 端到端权重 | 无 | 仓库树无 `.pt/.pth` 权重 | 必须自训 |
| TIGER 单测 | README 未承诺 | 未发现 `test_tiger*.py` | 若采用 Cornac，自建回归测试 |

---

*本文档为审计与计划产出，不含正式训练结果；所有 toy/合成数据的数字仅用于验证接口可运行性。*
