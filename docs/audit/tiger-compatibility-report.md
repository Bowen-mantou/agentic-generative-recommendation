# TIGER 外部实现与权重兼容性核查（Gate 1 / R1）

> 核查日期：2026-11-24（Asia/Shanghai）
> 核查人：tiger-recon（只读侦察；未改代码、未装依赖、未在服务器跑训练）
> 核查环境：本机 Windows + git-bash；`huggingface.co` 直连超时（与既有结论一致），全部 HF 查询走 `hf-mirror.com`；GitHub API / raw 可用。文中 stars、最后提交、HF 元数据均为 **2026-11-24** 抓取值。
> 关联决策：`docs/decisions/FINAL_DECISIONS_AND_RISK_REGISTER.md` Q1（外部实现/权重优先，自建备用）、R1（兼容性）、R2（自建成本）；`AGENTS.md` §8.1。
> 范围：只核查公开的 TIGER / RQ-VAE / Semantic ID 实现、权重与数据；**不**评估 E5、Reranker、Planner 等其它模块。

---

## 1. 结论摘要

**结论等级：需改造（代码可复用，权重不可复用）。**

| 问题 | 核查结论 | 证据强度 |
|---|---|---|
| TIGER 官方实现 | **不存在**。arXiv 页无作者官方代码链接；GitHub `google-research` 组织内无 TIGER 仓库；社区实现的 README 明确写 "There is no official code release" | 高（多源一致） |
| 可复用的社区实现 | **6 个主要候选**（Cornac、EdoardoBotta、XiaoLongtaoo、torch-rechub、GRID、genrec），另有 2 个低优先复现 | 高（逐个核对仓库元数据/README） |
| TIGER 端到端预训练权重 | **无**。没有任何公开的 T5/TIGER 生成器/解码器权重，也没有任何面向 Amazon Reviews 2023 All_Beauty 的权重 | 高（GitHub 树、HF 检索均无） |
| RQ-VAE tokenizer 权重 | **有，但只覆盖别的商品集合**：Amazon Beauty/Sports（P5/2014 口径）、MovieLens 32M 等；输入维度可巧合同为 768，但 embedding 空间与商品集合都不同，**不可直接用于本项目 112,590 个商品** | 高（config 与模型卡） |
| 数据格式兼容 | 所有实现原生支持的是 **Amazon 2014 5-core / P5 Amazon 子集**（`reviewerID`/`asin`/`unixReviewTime`）；**没有任何实现原生支持 2023 的 `user_id`/`parent_asin`/毫秒 `timestamp`** | 高（源码/文档） |
| 商品 ID 粒度 | 文献与实现均为 **asin/内部整数 item id 级**；本项目主键是 **parent_asin 聚合后的 `product_id`**，需自建聚合与映射，未验证 | 高（契约明确，兼容性未验证） |
| 是否可离线 | 代码本身可离线；但 Amazon 2014/P5 数据多来自 Google Drive / snap.stanford.edu，HF 直连不可达时需预下载 + 缓存。**训练与推理不依赖联网** | 中高 |
| 对本项目 Phase B 的含义 | **Q1 的"外部权重"路线不成立**；B 只能兑现为"外部代码复用"，实质仍要执行 A（自建 RQ-VAE + 自回归训练），只是少写代码。按 Q1 规则该结论应提交负责人确认，不能自动改决策 | 高 |

**一句话**：公开世界里有成熟的 TIGER/RQ-VAE **代码**，但**没有可直接用的 TIGER 权重**；本项目必须用自己的 112,590 个商品 E5 向量重训 codebook 和生成器。最省事的底座是 **Cornac 的 TIGER 实现**（Apache-2.0、维护活跃、支持自定义 embedding 与自定义 next-item 数据、内置 RQ-VAE + 约束 beam + Semantic-ID 指标），其次是 Botta 的 RQ-VAE 工程（MIT，可作 tokenizer 对照）。

**最大风险**：R1 的"权重不可复用 + 商品集合/ID 粒度不可对齐"已从"未知"变为"基本证伪"；R2 的"自建成本"仍完全未测（Reviews 未下载、112,590 商品全量 RQ-VAE/seq2seq 在 CPU 基线下的时长与显存均无数据）。

---

## 2. 候选实现清单

### 2.1 主要候选（6 个）

| # | 仓库 / 权重 | 类型 | Stars | 最后提交 | 许可证 | 数据口径 / 商品集合 | 是否带权重 | 能否自定义商品集 | 备注 |
|---|---|---|---|---|---|---|---|
| 1 | [PreferredAI/cornac](https://github.com/PreferredAI/cornac)（TIGER 模块 2026 年新增） | 通用推荐框架 + 完整 TIGER 实现 | 1,065 | 2026-11-23 | **Apache-2.0** | 内置 Amazon 2014 5-core（Beauty/Sports/Toys）loader；TIGER 本身可吃任意 `FeatureModality` + 任意 next-item 数据 | **无**（训练时从零训练，无 checkpoint 下载） | **是**（自定义 item embedding 与 item id；内置 loader 只支持 2014） | 最推荐底座。RQ-VAE + T5-style seq2seq + 约束 beam + SID 指标齐全；文档给出 Beauty 复现 ~0.042 R@5（论文 0.0454） |
| 2 | [EdoardoBotta/RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender) | RQ-VAE + T5 decoder 训练/推理全流程 | 858 | 2026-08-31 | **MIT** | P5 Amazon 子集（Beauty/Sports/Toys，Google Drive），另支持 MovieLens 1M/32M tokenizer | **有 tokenizer 权重**（`trained_models/rqvae_amazon_beauty`、`rqvae_amazon_sports`、`rqvae_ml32m`，各约 13.8MB）；**无 decoder/检索权重** | 部分（需改 data pipeline；P5/ML 之外无现成入口） | 工程最干净、检查点可复现；HF 上有同源 RQ-VAE Beauty 权重 |
| 3 | [XiaoLongtaoo/TIGER](https://github.com/XiaoLongtaoo/TIGER) | 论文非官方 PyTorch 复现 | 267 | 2025-12-19 | **MIT** | Amazon 2014 5-core Beauty/Sports/Toys | **有 Beauty RQ-VAE 权重**（`rqvae/ckpt/Beauty/.../best_*.pth`，约 14MB×2）；**无 T5 权重**（`model/ckpt/None` 占位） | 否/需改（路径与 `cuda:0` 硬编码、映射文件固定 Beauty） | 论文级 recipe 完整；依赖陈旧（Python 3.7 / torch 1.10），与项目 Python 3.10 venv 不匹配 |
| 4 | [datawhalechina/torch-rechub](https://github.com/datawhalechina/torch-rechub) | 推荐框架中的 TIGER 实现 + 中文复现文档 | 1,242 | 2026-11-23 | **MIT** | 示例 MovieLens / Amazon Books；数据格式为 `inter.json` + `semantic_ids.json`（ID 无关） | **无**（T5 随机初始化从头训练） | **是**（换 JSON 即可；支持自定义 item id 与语义 ID） | 文档最细（Trie 约束 beam、leave-one-out 切分、RQ-VAE 导出）；无预训练 T5，即使指定 `--base_model` 也只取结构 |
| 5 | [snap-research/GRID](https://github.com/snap-research/GRID)（TIGER 模块为 2026 版权新代码） | 生成式推荐框架（含 TIGER 与 RQ-KMeans/RVQ/RQ-VAE） | 773 | 2025-10-15 | **NOASSERTION → 实际为 Snap 非商业研究许可**（见 §4） | P5 Amazon（Beauty/Sports/Toys，Google Drive）；embedding 可用任意 HF LLM | **无**（仓库内无 checkpoint，训练时指定 `ckpt_path`） | 部分（需按 train/validation/test/items 四目录格式转换） | 工程完整度高；**许可证限制商业用途**，且 `torch==2.6.0+cu124` 锁死，CPU 基线需改依赖 |
| 6 | [phonism/genrec](https://github.com/phonism/genrec) | 生成式推荐 Model Zoo（TIGER 论文对齐 recipe） | 145 | 2026-11-02 | **MIT** | Amazon 2014 Beauty/Sports/Toys/Home（矩阵式 benchmark） | **无**（仓库只有 config/脚本，无权重） | 部分（gin 配置 + Amazon 2014 数据管线；自定义需改数据层，未验证） | 适合做"第二意见" baseline；比 Cornac 复杂，社区更小 |

### 2.2 低优先候选（2 个，仅作备查）

| 仓库 | Stars | 最后提交 | 许可证 | 说明 |
|---|---:|---|---|---|
| [NonameUntitled/tiger](https://github.com/NonameUntitled/tiger) | 43 | 2025-10-17 | MIT | 又一个 TIGER 复现，未见权重与维护记录 |
| [baiyimeng/TIGER](https://github.com/baiyimeng/TIGER) | 23 | 2025-08-20 | MIT | 同上，资料少 |

### 2.3 官方实现核查

| 检查项 | 结果 | 证据 |
|---|---|---|
| 论文 | Rajput et al., *Recommender Systems with Generative Retrieval*, NeurIPS 2023 | [arXiv:2305.05065](https://arxiv.org/abs/2305.05065)（2023-05-08 提交；NeurIPS 2023） |
| arXiv 页代码链接 | 无作者官方仓库链接；"Code, Data, Media" 标签下只有 CatalyzeX 等第三方代码搜索工具 | 抓取 arXiv abs 页（第 341–375 行附近）：无 GitHub repo，只有 CatalyzeX 等第三方入口 |
| `google-research` 组织内 TIGER 仓库 | 0 个 | GitHub Search API `q=tiger org:google-research` → `total_count=0` |
| 社区侧声明 | Cornac TIGER README 明确写 "There is no official code release" | [cornac/models/tiger/README.md](https://github.com/PreferredAI/cornac/blob/master/cornac/models/tiger/README.md) |
| GitHub 全站检索 | 16 个相关仓库，全部为社区复现/衍生，无官方仓库 | GitHub Search API `q=TIGER generative retrieval recommender`（同日抓取） |

---

## 3. 权重与数据集清单（HuggingFace）

### 3.1 权重（截至 2026-11-24）

| HF 仓库 | 内容 | 参数量 / 大小 | 输入维度 / 结构 | 训练数据 | 许可证 | 最后更新 | 能否用于本项目 |
|---|---|---|---|---:|---|---|---|---|
| [edobotta/rqvae-amazon-beauty](https://huggingface.co/edobotta/rqvae-amazon-beauty) | RQ-VAE tokenizer（safetensors） | 1,146,880 参数 / 4.6MB | `input_dim=768`，`embed_dim=32`，`codebook_size=256`，`n_layers=3`，hidden `[512,256,128]` | Amazon Beauty（Botta 仓库 P5 口径，非 2023） | 模型卡**未声明 license**（代码仓库 MIT） | 2025-02-01 | **否**（商品集合与 embedding 空间不同；仅可做实现对照） |
| [junchenfu/diger-rqvae-beauty](https://huggingface.co/junchenfu/diger-rqvae-beauty) | RQ-VAE tokenizer（`best_collision_model.pth`） | 269.9MB | 未核对 config | Beauty（DIGER 论文口径） | **MIT** | 2026-08-30 | **否**（同为 Beauty 但非 2023 All_Beauty；未核对维度） |
| 仓库内权重：[Botta `trained_models/`](https://github.com/EdoardoBotta/RQ-VAE-Recommender/tree/main/trained_models) | RQ-VAE Beauty（3 个 ckpt）、Sports、ML32M | 各约 13.8MB | 同上 | P5 Amazon / MovieLens | 仓库 MIT | 2026-08-31（仓库） | **否**（同 1） |
| 仓库内权重：[XiaoLongtaoo `rqvae/ckpt/Beauty/`](https://github.com/XiaoLongtaoo/TIGER/tree/main/rqvae/ckpt/Beauty) | RQ-VAE Beauty（best_collision / best_loss） | 各约 14MB | 未核对 config | Amazon 2014 Beauty | 仓库 MIT | 2025-12-19（仓库） | **否**（2014 Beauty） |

**明确结论：TIGER 端到端"无预训练权重"。** 仅存在 **RQ-VAE tokenizer 权重**，覆盖 Amazon **Beauty / Sports**（P5/Amazon 2014 口径）与 **MovieLens 32M**；另有 Amazon Pet Supplies、KuaiRand、Yelp、Instruments 等社区 tokenizer，但**没有任何权重覆盖 Amazon Reviews 2023 `All_Beauty`**，也**没有任何 TIGER 生成器（T5/decoder）权重**。

检索方式与结果：

| 检索 | 结果 |
|---|---|
| HF models `search=tiger`（40 条） | 全部为 TigerBot、Tiger 图像模型等无关项，**无 TIGER 推荐模型** |
| HF models `search=recommender` | 无 TIGER / Semantic ID 推荐模型 |
| HF models `search=rqvae` / `rq-vae` | 有多个 RQ-VAE tokenizer（Amazon Beauty、Pet Supplies、KuaiRand、MovieLens 等），**均非 TIGER decoder** |
| HF models `filter=arxiv:2305.05065` | 仅 `edobotta/rqvae-amazon-beauty` 与 `rtferraz/domainTokenizer`；**无 decoder** |
| HF models `filter=generative-recommendation` | 仅 DIGER RQ-VAE（Beauty/Instruments/Yelp）与 OneRec 系列 LLM，**无 TIGER** |
| HF datasets `search=tiger` / `semantic-ids` | 无 "TIGER 权重/数据集"；只有 GRID 数据镜像与少量语义 ID mapping 数据集 |

### 3.2 数据（可作为序列/格式参考）

| HF 数据集 | 内容 | 规模 | 最后更新 | 与本项目关系 |
|---|---|---:|---|---|
| [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | 官方 Amazon Reviews 2023（含 `raw_meta_All_Beauty`，112,590 条，`parent_asin`/`details`/`price` 字符串等；时间戳为毫秒） | 10B–100B 级总量；All_Beauty meta 172.6MB / 下载 59.6MB | 2024-12-08 | **本项目数据源**；证明 2023 schema 与 2014 不同 |
| [GlowBond/amazon-semantic-ids-recommendation](https://huggingface.co/datasets/GlowBond/amazon-semantic-ids-recommendation) | GRID/P5 预处理 Amazon 序列 + 商品元数据（Beauty/Sports/Toys） | 4.75GB | 2026-03-02 | 外部实现的训练数据镜像；**非** All_Beauty 2023 |
| [yufan/amazon2023-user-interactions](https://huggingface.co/datasets/yufan/amazon2023-user-interactions) | Amazon 2023 五品类 5-core/leave-one-out 序列（含 `Beauty_and_Personal_Care`，非 `All_Beauty`） | 10M–100M 交互 | 2026-06-23 | 2023 序列构建的社区参考；**不**等于本项目 All_Beauty 品类 |

> 许可证提示：Amazon Reviews 2023 数据集卡为 `license: other`（amazon-reviews-2023），使用前需按官方页面条款执行；本项目内部使用不改变该约束。

---

## 4. 许可证、依赖与可离线性

| 实现 | 许可证 | Python / 关键依赖 | GPU 要求 | 可离线 | 备注 |
|---|---|---|---|---|---|
| Cornac TIGER | **Apache-2.0**（允许商用/修改/再分发，需保留声明） | Python ≥3.10；TIGER 额外 `torch>=1.12.0`、`transformers>=4.30.0`；示例另需 `sentence-transformers` | 不强制（代码有 device 参数，CPU 可跑；示例"1 小时/单 GPU"是 GPU 口径） | **是**（训练/推理不下载权重；内置 Amazon loader 会联网，但本项目可绕过） | 版本 3.2.0；TIGER 模块代码头部版权年份 2026，属新代码 |
| Botta RQ-VAE-Recommender | **MIT** | Python ≥3.10；`torch>=2.5.1`、`transformers>=4.47,<5`、`sentence-transformers>=3.3.1`、`accelerate`、`gin-config`、`torch-geometric`、`wandb` | 推荐 CUDA/MPS；单元测试 CPU 可跑，全量训练建议 GPU | **是**（代码可离线；P5 数据在 Google Drive，首次需预下载） | 版本 1.0.1；有 Zenodo DOI |
| XiaoLongtaoo/TIGER | **MIT** | Python **3.7.16**、`torch==1.10.1`、`transformers==4.30.2`、`sentence-transformers==2.2.2`（requirements 另列 torch 1.8/pytorch 1.8） | **是**（`generate_code.py` 硬编码 `cuda:0`；训练脚本以 GPU 为主） | 是 | 依赖过旧，与 Python 3.10 项目环境冲突；需改代码 |
| torch-rechub TIGER | **MIT** | Python ≥3.10；`torch>=2.6.0,<3`、numpy/pandas/sklearn；`uv sync --extra generative` | 不强制（示例在 RTX 5080 上测过 toy；CPU 亦可） | 是（toy 数据可离线生成） | 版本 1.0.0；T5 从头训练 |
| GRID | **Snap 非商业研究许可**（GitHub 识别为 NOASSERTION） | Python ≥3.10；`torch==2.6.0+cu124`、`lightning==2.5.0`、`transformers==4.47.0`、hydra、torchrec 等 | **推荐 CUDA GPU**（README 明示） | 是（代码离线；P5 数据在 Google Drive） | **禁止商业用途**；依赖锁死 CUDA 版本 |
| phonism/genrec | **MIT** | Python ≥3.9；PyTorch ≥2.0；分布式训练支持 | 训练建议 GPU | 是 | 模型 zoo；未逐项核对其依赖锁 |

**许可证结论**：
- Cornac（Apache-2.0）、Botta（MIT）、torch-rechub（MIT）、genrec（MIT）可安全用于本项目研究与后续分发。
- **GRID 为 Snap 非商业研究许可**：课程/研究可用，但若未来商业化或对外提供产品，必须替换或单独授权；不建议作为主底座。
- HF 上的 `edobotta/rqvae-amazon-beauty` 模型卡**未写 license**，虽然代码仓库 MIT，再分发权重前应确认（保守做法：不随项目分发该权重）。

---

## 5. 兼容性分析（数据格式 / ID 粒度 / 自定义商品集合 / 权重迁移）

### 5.1 数据格式

| 维度 | 文献/外部实现 | 本项目 | 差距与改造 |
|---|---|---|---|
| 数据集版本 | Amazon 2014 5-core / P5 Amazon（Beauty/Sports/Toys）；Cornac 源码注释称 2018/2023 "available"，但 `_validate()` **实际只允许 2014** | Amazon Reviews 2023 `All_Beauty`（metadata 112,590 商品；reviews 326,611,506 B 尚未下载） | 必须自建 2023 → 2014 风格字段转换；不能依赖现成 loader |
| 用户字段 | `reviewerID` | `user_id` | 重命名 |
| 商品字段 | `asin`（review 级） | `parent_asin` → 本项目 `product_id` | **需聚合**：同一 parent_asin 下多个 asin 变体合并为同一商品；聚合规则未定义 |
| 评分/事件 | `overall`（1–5） | `rating` | 重命名；事件类型记为 `review`/`rating`，不得伪装为点击/浏览 |
| 时间戳 | `unixReviewTime`（秒） | `timestamp`（毫秒） | 统一为秒或毫秒并做单调性校验 |
| 序列切分 | TIGER 论文 leave-last-out；部分实现另有时间切分 | 项目要求**按时间切分**（防泄漏） | 需实现 time-based split；不能直接沿用 leave-last-out 做正式评测 |
| 序列长度 | 常见 `max_len=20`（Cornac）/ 50（GRID/genrec） | 未知（Reviews 未下载） | 先统计 All_Beauty 用户序列长度分布再定 |

### 5.2 商品 ID 粒度

- 所有外部实现的 item id 都是 **review 级 asin 或内部整数 index**（如 Cornac 的 `FeatureModality(ids=item_ids)`、torch-rechub 的 `inter.json` item id）。
- 本项目主键是 **parent_asin 聚合后的 `product_id`**；同一 parent_asin 的多个 asin 变体在 reviews 中会产生多条事件，必须在接入层聚合/去重。
- **没有任何实现内置 parent_asin → product_id 的语义 ID 映射**；需要自建：
  1. `product_id` ↔ 内部整数 index（稳定排序，落盘）；
  2. 内部 index → Semantic ID（`sid_table`）；
  3. Semantic ID tuple → `product_id`（反查表，含碰撞消歧位）。
- 该映射是 TIGER 输出能被 Fusion/Reranker 消费的前提；未验证。

### 5.3 自定义商品集合

| 实现 | 是否支持 | 具体方式 |
|---|---|---|
| Cornac | **支持** | `FeatureModality(features=embs, ids=item_ids)` + `NextItemEvaluation.from_splits(...)`；item id 与 embedding 行对齐即可 |
| torch-rechub | **支持** | 直接喂 `inter.json`（`{user_id: [item_id,...]}`）与 `semantic_ids.json`（`{item_id: ["<a_..>",...]}`） |
| Botta / GRID / genrec | 部分 | 需按各自 data pipeline 转换（P5 zip / train、validation、test、items 四目录 / Amazon 2014 管线），改造成本中等 |
| XiaoLongtaoo | 否/需改 | 路径、`cuda:0`、Beauty 映射文件硬编码 |
| HF 权重 | 否 | 权重绑定固定商品集合与 embedding 空间 |

### 5.4 权重可迁移性（核心否证）

- RQ-VAE tokenizer 的 codebook 是在**特定商品集合、特定文本 embedding（Sentence-T5/SBERT）**上学出来的。把本项目的 E5 向量喂给 `edobotta/rqvae-amazon-beauty`，输出代码只表示"在 SBERT/Amazon Beauty 空间里最近的码字"，**不携带本项目商品的语义**。
- 即使输入维度同为 768（Botta config `input_dim=768`，E5 也是 768），**维度相同 ≠ 空间相同**；不能因为维度巧合而复用。
- TIGER decoder 的 token 词表 = 各层 codebook 的码字（Cornac：3 层×256 + 碰撞消歧层）。换 codebook 就必须**从头训练 decoder**。因此即使有 decoder 权重，也不适配新 codebook。
- **结论：端到端权重不可复用；RQ-VAE 与 decoder 都必须在本项目 112,590 商品上重训。** 这与 Q1 的"外部权重优先"字面要求冲突，需负责人确认解释。

---

## 6. 对本项目的改造清单

> 目标：Phase B 最小真实链路 = E5 向量 → RQ-VAE → Semantic ID → 用户序列 → 自回归生成 → 约束解码 → `product_id`。以下改造以 **Cornac TIGER 为底座**（备选 torch-rechub），不改变项目契约。

| # | 改造项 | 现状 / 风险 | 具体动作 | 验收标准 |
|---|---|---|---|---|
| 1 | **Embedding 替换** | 外部实现默认 Sentence-T5/SBERT；本项目已有 `intfloat/e5-base-v2`（768 维，`passage:` 前缀） | 用 E5 `passage:` 向量替换 item content embedding；确认归一化、 dtype、行序与 `item_id` 对齐；RQ-VAE `input_dim` 保持 768 | 112,590×768 矩阵与 item 表逐行对齐；抽样余弦相似度 sanity check |
| 2 | **Codebook 重训** | 无可复用权重；外部 tokenizer 只覆盖别的商品集合 | 在 112,590 个 E5 向量上训练 RQ-VAE（3 层×256，latent 32，hidden `[512,256,128]`，commitment 0.25）；加碰撞消歧层（Cornac 自动追加）；可并行试 `rkmeans` 作为零梯度基线 | 训练收敛（重构/量化损失记录）；碰撞率与 SID 长度统计；`sid_table` 可导出 |
| 3 | **序列格式** | 外部为 2014/P5 5-core leave-last-out；本项目从 SQLite `InteractionHistory` 构建 | 按 `user_id`、时间排序；`product_id` → 内部整数 id；事件保留 `review`/`rating`、`is_synthetic=false`；按时间切分 train/valid/test；`max_len` 先按分位数（如 20/50） | 每个用户序列单调递增；无未来事件泄漏；样本数与长度分布报告 |
| 4 | **Semantic ID → product_id 映射** | 外部实现只保证 item index ↔ SID，不保证 parent_asin 粒度 | 生成 `product_id` ↔ index、index ↔ SID、SID tuple ↔ `product_id` 三张表；碰撞消歧位纳入 tuple；非法/重复解码结果丢弃并计数 | 随机抽样 1,000 个 SID 可唯一反查；解码结果 100% 映射到合法 `product_id` |
| 5 | **解码 / Beam Search** | Cornac 有约束 beam 与 exact 两种打分；torch-rechub 有 Trie | 用前缀 Trie 约束 beam（`n_beams ≥ K`）；CPU 基线先测 batch=1 的 P50/P95；输出按 SID 概率/对数似然排序 + 确定性 tie-break | 同一输入两次解码结果一致；Top-K 无重复；非法 SID 为 0 |
| 6 | **与 E5 的两级 RRF 对接** | 项目要求 E5 内部 RRF → 再与 TIGER 跨路 RRF；不得相加原始分数 | TIGER 只输出**排名**（`product_id` + rank + score 仅留诊断），Fusion 用 rank 做 RRF；TIGER 分支不得直接进 Reranker | 单测：两路排名融合可复现；分数不参与跨模型加权 |
| 7 | **工程化 / 可复现** | 外部代码风格与项目规范不同 | 适配层放 `src/recommender/retrieval/tiger/`；配置化模型名/层数/码本/beam/`max_len`；产物（codebook、sid_table、映射、checkpoint）放 NFS `artifacts/semantic_ids/`（Q10）；不 import 时初始化模型 | 配置可切换 seed；trace 记录 model/ckpt hash；单测覆盖映射/解码/碰撞 |

**额外建议（R2 降本）**：
- 先用 Cornac 的 `tokenizer="rkmeans"`（GRID 路线，无梯度训练）跑通全链路，再决定是否值得上 RQ-VAE；Cornac README 报告 rkmeans 与 rqvae 精度差距约 4% 以内，训练成本显著更低。
- 生成器先用 Cornac 默认 ~13M 参数 T5-style（d_model=128、4+4 层），不要从 HuggingFace 大 T5 初始化；这与外部实现一致，参数小、CPU 可行性更高。

---

## 7. 风险与未验证项

### 7.1 R1（外部 TIGER 与 All_Beauty 商品集合不兼容）

| 编号 | 未验证项 | 当前判断 | 建议验证（不执行训练） |
|---|---|---|---|
| R1-1 | 是否存在 TIGER 端到端公开权重 | **已证伪（无）** | 保持检索（HF tag、GitHub topic）；记录检索日期 |
| R1-2 | Amazon 2014/P5 Beauty 与 All_Beauty 2023 的 `parent_asin` 重叠度 | 未知；预计很低（版本/品类口径不同） | 下载 2014/P5 商品 id 列表，与 All_Beauty 112,590 做集合交集统计 |
| R1-3 | `parent_asin` 聚合后与 asin 级序列的差异 | 未知 | 统计 All_Beauty 中一个 parent_asin 对应的 asin 变体数与评论分布 |
| R1-4 | 2023 schema → 2014 风格转换的正确性 | 无实现原生支持 | 写转换脚本 + 单测（字段映射、时间单位、去重） |
| R1-5 | 现成 loader 是否真支持 2018/2023 | **Cornac `_validate()` 只允许 2014**（文档注释与实现不一致） | 以源码为准；不要依赖其 2018/2023 分支 |
| R1-6 | 许可证 | GRID 仅研究用途；edobotta 模型卡无 license | 项目内使用前由负责人确认；不随项目分发无 license 权重 |
| R1-7 | 所有 SID 是否能唯一映射回 `product_id` | 未知 | 碰撞率/反查覆盖率实验（纯 CPU，分钟级） |

### 7.2 R2（自建 TIGER 训练成本超预算）

| 编号 | 未验证项 | 当前判断 | 建议验证（不执行训练） |
|---|---|---|---|
| R2-1 | Reviews 文件内容（用户数、序列长度、5-core 后商品数） | **未知**（reviews 326.6MB 未下载，任务背景明确） | 下载后统计用户数、每用户事件数分位数、冷启动商品比例 |
| R2-2 | RQ-VAE 在 112,590×768 上的训练时长/内存 | 未知；Cornac 示例仅 22k Beauty "约 1 小时/单 GPU" | 先跑 1 个 epoch/子集 pilot，外推全量；CPU 与 MUSA 分开测（Q11 第一阶段 CPU） |
| R2-3 | seq2seq 训练样本量与 epoch 时长 | 未知；取决于用户数与 `max_len` | 用 10% 用户跑 1 epoch，记录 tokens/s 与 loss |
| R2-4 | 约束 beam（K=50）CPU 延迟 | 未知 | 单用户 batch=1 的 P50/P95；与 Reranker 延迟一起做预算 |
| R2-5 | 3×256 码本在 112k 商品上的碰撞率 | 未知 | 用现有 E5 向量做 rkmeans 快速上界估计 |
| R2-6 | MUSA 兼容性 | 未验证；Q11 第一阶段不接入，仅 CPU | 不在 Gate 1 范围内；Phase 7 再评估 |
| R2-7 | 训练数据泄漏（未来评论/目标商品） | 项目要求时间切分 | 序列构建按时间切分；目标商品不得出现在历史 |

### 7.3 对 Q1 决策的影响

Q1 写的是"**外部实现/权重优先**；自建为备用"。核查结果：

- **外部实现**：可用（Cornac 等），能显著减少代码量；
- **外部权重**：**不存在可用权重**（端到端为零；tokenizer 权重也不适配本项目商品集合）。

因此 Q1-B 只能兑现"外部代码复用"，**不能兑现"外部权重复用"**；实际落地仍等价于 Q1-A（自建训练），只是工程实现可借鉴外部代码。按 `AGENTS.md` §0.1 与决策变更管理规则，**不得由 AI 自行改写 Q1**；建议负责人确认以下二选一：

1. 将 Q1-B 解释为"外部代码优先、权重自建"，批准以 Cornac 为底座的 Phase B pilot；或
2. 维持"外部权重优先"字面，则判定 B 不成立，按 A 提交训练资源评估后再启动。

在负责人确认前，**不投入完整自建训练**；可先做 R1-2/R1-7、R2-1/R2-5 的零训练核查（下载统计 + 碰撞率），成本以小时计。

---

## 8. 建议的下一步（Gate 1 收口）

1. **决策**：lead/负责人确认 §7.3 的 Q1 解释（建议 1），并登记到风险台账（R1 从 Open → In Progress/Mitigated，R2 保持 Open）。
2. **零训练验证（建议 0.5–1 人时）**：
   - 下载 reviews（326.6MB），统计用户数、序列长度、5-core 后商品数、`parent_asin` 聚合影响；
   - 用已有 E5 向量跑一次 `rkmeans`（3×256）估计碰撞率与 SID 分布；
   - 核对 2014/P5 Beauty 与 All_Beauty 的商品交集。
3. **最小 pilot（需负责人批准后）**：Cornac TIGER + E5 768 维 + 10% 用户 + `rkmeans` 先跑通"真实序列 → 真实 SID → 真实 `product_id`"；再评估是否值得上 RQ-VAE 与全量训练。
4. **不执行**：完整 seq2seq 训练、MUSA 实验、任何 Mock 冒充 Semantic ID 的"验收"。

---

## 9. 证据链接（核查日期 2026-11-24）

**论文 / 官方**
- [Rajput et al., Recommender Systems with Generative Retrieval (arXiv:2305.05065)](https://arxiv.org/abs/2305.05065)
- [GitHub Search API：`tiger org:google-research` → 0](https://api.github.com/search/repositories?q=tiger+org:google-research)（同日抓取）

**主要实现**
- [PreferredAI/cornac](https://github.com/PreferredAI/cornac)｜[TIGER README](https://github.com/PreferredAI/cornac/blob/master/cornac/models/tiger/README.md)｜[TIGER 示例](https://github.com/PreferredAI/cornac/blob/master/examples/tiger_example.py)｜[Amazon loader 源码](https://github.com/PreferredAI/cornac/blob/master/cornac/datasets/amazon_review.py)｜[pyproject（v3.2.0 / Python≥3.10）](https://github.com/PreferredAI/cornac/blob/master/pyproject.toml)
- [EdoardoBotta/RQ-VAE-Recommender](https://github.com/EdoardoBotta/RQ-VAE-Recommender)｜[requirements](https://github.com/EdoardoBotta/RQ-VAE-Recommender/blob/main/requirements.txt)｜[trained_models](https://github.com/EdoardoBotta/RQ-VAE-Recommender/tree/main/trained_models)｜[pyproject（MIT / Python≥3.10）](https://github.com/EdoardoBotta/RQ-VAE-Recommender/blob/main/pyproject.toml)
- [XiaoLongtaoo/TIGER](https://github.com/XiaoLongtaoo/TIGER)｜[requirements（Python 3.7 / torch 1.10）](https://github.com/XiaoLongtaoo/TIGER/blob/main/requirements.txt)｜[Beauty RQ-VAE ckpt](https://github.com/XiaoLongtaoo/TIGER/tree/main/rqvae/ckpt/Beauty)
- [datawhalechina/torch-rechub](https://github.com/datawhalechina/torch-rechub)｜[TIGER 中文复现说明](https://github.com/datawhalechina/torch-rechub/blob/main/docs/zh/blog/tiger_reproduction.md)｜[pyproject（v1.0.0 / MIT）](https://github.com/datawhalechina/torch-rechub/blob/main/pyproject.toml)
- [snap-research/GRID](https://github.com/snap-research/GRID)｜[LICENSE（Snap 非商业研究许可）](https://github.com/snap-research/GRID/blob/main/LICENSE)｜[requirements（torch 2.6.0+cu124）](https://github.com/snap-research/GRID/blob/main/requirements.txt)｜[GRID 论文 arXiv:2507.22224](https://arxiv.org/abs/2507.22224)
- [phonism/genrec](https://github.com/phonism/genrec)
- [NonameUntitled/tiger](https://github.com/NonameUntitled/tiger)｜[baiyimeng/TIGER](https://github.com/baiyimeng/TIGER)

**HuggingFace**
- [edobotta/rqvae-amazon-beauty](https://huggingface.co/edobotta/rqvae-amazon-beauty)（config：`input_dim=768`、3×256 codebook，2025-02-01 更新）
- [junchenfu/diger-rqvae-beauty](https://huggingface.co/junchenfu/diger-rqvae-beauty)（MIT，269.9MB，2026-08-30 更新）
- [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)（All_Beauty 112,590 条，毫秒时间戳）
- [GlowBond/amazon-semantic-ids-recommendation](https://huggingface.co/datasets/GlowBond/amazon-semantic-ids-recommendation)（GRID/P5 数据镜像，4.75GB）
- [yufan/amazon2023-user-interactions](https://huggingface.co/datasets/yufan/amazon2023-user-interactions)（2023 五品类 5-core 序列，非 All_Beauty）

**项目内**
- [Q1–Q12 决策与风险台账](../decisions/FINAL_DECISIONS_AND_RISK_REGISTER.md)（R1 第 269–270 行、R2 第 272–273 行）
- [模块三：Candidate Retrieval（TIGER/Semantic ID/两阶段方案）](../模块三：Candidate%20Retrieval%20商品候选召回系统.md)
- [仓库审计（TIGER 无实现/无权重）](repository-audit-and-plan.md)

---

## 10. 附：核查方法与可复现命令

- GitHub 元数据：`curl https://api.github.com/repos/<owner>/<repo>`（stars、`pushed_at`、license）。
- 仓库文件树：`curl https://api.github.com/repos/<owner>/<repo>/git/trees/<branch>?recursive=1`，过滤 `.pt/.pth/ckpt/trained_models/requirements`。
- README 原文：`cdn.jsdelivr.net/gh/<owner>/<repo>@<branch>/<path>`（本机 raw.githubusercontent 不稳定，jsDelivr 稳定返回原文）。
- HF 检索：`https://hf-mirror.com/api/models?search=<q>`、`?filter=arxiv:2305.05065`、`?filter=generative-recommendation`，以及 `/api/datasets?...`。
- 本文所有 stars / 最后提交 / 下载量均为 2026-11-24 单次抓取，后续可能变化；结论以"是否有权重、是否支持 2023/自定义商品集"为主，不依赖具体数值。
- **未执行**：任何 clone 后的训练、pip 安装、服务器命令、模型下载（仅元数据/README/配置文件读取）；因此所有"~1 小时/单 GPU"等时长均为**外部文档引用**，不是本项目实测。
