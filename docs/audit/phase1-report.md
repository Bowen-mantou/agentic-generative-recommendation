# Phase 0 + Phase 1 执行与验证记录

> 日期：2026-11-24
> 环境：MTT S4000（`agr-worker`，128 核 / 1 TB 内存 / 8×MTT S4000）与本地 Windows 开发机
> 结论：Phase 0 与 Phase 1 均已完成，**所有数字来自真实运行，没有 Mock 结果**。
> 本文档只覆盖 Phase 0/1；TIGER / Reranker / Planner / Judgment / Controller /
> Generator / Memory / 端到端评测尚未实现（见 [repository-audit-and-plan.md](repository-audit-and-plan.md) §6）。

---

## 1. 环境与依赖（实测）

### 1.1 服务器

| 项 | 值 |
|---|---|
| 主机 | `agr-worker`，Ubuntu 5.15，128 核，1 TB RAM |
| GPU | 8×MTT S4000 48 GB，driver 3.3.5-server，MUSA（非 CUDA）；Phase 1 未使用 GPU |
| Python | 宿主 3.10.12；`python3-venv` 缺失 → 用 `uv 0.12.21` 建 venv |
| venv | `/tmp/agr/venv`（约 1.5 GB；`/home` 仅剩约 8 GB） |
| torch | 2.6.0+cpu |
| transformers | 4.57.6 |
| sentence-transformers | 5.7.0 |
| faiss | 1.16.0（IndexFlatIP） |
| numpy | 2.2.6 |
| pydantic | 2.14.4 |
| 网络 | `huggingface.co` 超时；`hf-mirror.com` 可用（约 2.5–3.6 MB/s）；pip 走清华镜像 |
| 存储 | `/home` 191 GB（原剩 3.7 GB，清理 uv 缓存后 8.9 GB）；`/tmp` 122 GB；`/mnt/agr-data` 12 TB NFS |

### 1.2 本地（开发/单测）

Windows 11，Python 3.13.11，pydantic 2.12.3，pytest 9.1.1；无 faiss（集成测试自动跳过）。

### 1.3 操作事故与修复（如实记录）

首次 `uv pip install torch` 把 **6 GB uv 缓存写进 `/home`**（分区从 3.7 GB 降到 2.9 GB 可用，
99% 占用）。处理：停止安装 → `uv cache clean`（释放 6.0 GB，回到 8.9 GB 可用）→ 在
`scripts/bootstrap_s4000.sh` 内固定 `UV_CACHE_DIR=/tmp/agr/uv-cache`、`TMPDIR=/tmp/agr/tmp`
→ 重新安装成功。此后所有大文件都不再写入 `/home`（ADR-0004）。

---

## 2. Phase 0：工程骨架与数据契约

### 2.1 交付

| 内容 | 位置 |
|---|---|
| 打包与依赖分组（core / retrieval / workflow / memory / dev） | `pyproject.toml` |
| 密钥模板、忽略规则、行尾规范 | `.env.example`、`.gitignore`、`.gitattributes` |
| 配置加载（5 份 YAML + `local.yaml` 覆盖 + `AGR__` 环境覆盖 + 类型校验 + 路径解析） | `src/recommender/config.py`、`configs/` |
| 13 个 Pydantic Schema 模块（Plan / Candidate / Constraint / Rerank / Judgment / Controller / Memory / Eval / Artifact） | `src/recommender/schemas/` |
| JSON 日志 + request_id/trace_id + 脱敏 + JSONL trace | `src/recommender/observability/` |
| 错误类型、原子 IO、哈希 | `src/recommender/utils/` |
| CLI：`version` / `config show|validate` | `src/recommender/cli.py` |
| 单元测试 | `tests/unit/`（111 个） |

### 2.2 验证命令与结果

```bash
python -m pip install -e ".[dev]"
python -m pytest -q                 # 111 passed（本地）
ruff check src tests                # All checks passed
mypy                                # Success: no issues found in 45 source files
agr config validate                 # config valid: 1.0 + 全部路径
```

服务器上（Python 3.10.12 venv）执行同一套测试：**全部通过**（见 §5）。

---

## 3. Phase 1：商品目录 + E5 + FAISS

### 3.1 数据下载（真实）

```bash
agr catalog download --category All_Beauty
```

| 项 | 值 |
|---|---|
| 源 | `McAuley-Lab/Amazon-Reviews-2023`，`raw/meta_categories/meta_All_Beauty.jsonl` |
| 端点 | `https://hf-mirror.com`（直连 HF 不可用） |
| 大小 | **212,990,142 B** |
| SHA-256 | `1cfbdf6d2ccf9c3ac52a4508b15f31f8f340508a245b65c3e3b5f461e06e97ef` |
| 耗时 | **2m20s**（含下载 + 落盘 + 哈希） |

### 3.2 商品目录

```bash
agr catalog build --meta .../meta_All_Beauty.jsonl --category All_Beauty --version all_beauty_v2
```

| 项 | 值 |
|---|---|
| 记录读入 | 112,590 |
| 写入 catalog | **112,590**（100%） |
| 跳过 | 0（无重复 ID / 无坏行 / 无缺失 ID） |
| products.jsonl SHA-256 | `e8cbdedc4b200733d91a3ac08b7e1b620494660f483c974109bb8b571c4cbdb2` |
| 源文件 SHA-256 | `1cfbdf6d…`（与下载记录一致） |
| 双文本模板 | `product_text.embedding.v1` / `product_text.rerank.v1` |

**商品字段覆盖率（真实统计）**：

| 字段 | 覆盖 | 说明 |
|---|---|---|
| title | 112,578 / 112,590（100.0%） | — |
| main_category | 112,590（100.0%） | `All Beauty` 112,135；`Premium Beauty` 455 |
| brand（store） | 101,259（89.9%） | — |
| details（属性字典） | 108,072（96.0%） | 商品属性主要来源 |
| average_rating / rating_number | 100.0% | 代理交互信号，Phase 4 用 |
| **price** | **17,704（15.7%）** | 大量商品无价格 |
| description | 19,162（17.0%） | — |
| features | 17,377（15.4%） | — |
| categories（列表） | 0（0.0%） | 该 dump 中 `categories` 全为空 |

> **审计修正（本轮发现并修复）**：初版 `build_product_record` 只读取空的
> `categories` 列表，导致 `main_category` 全为 `null`。复核原始记录发现顶层存在
> `main_category` 字段后修正（优先显式字段，其次回退到 `categories[0]`），并补了
> 两个单元测试，重建为 `all_beauty_v2`。v1 产物在 v2 验证通过后删除（避免占用 `/home` 空间）。
>
> **对下游的影响（已记录）**：`price` 只有 15.7% 覆盖，意味着 Phase 2 的价格硬约束
> 对多数商品只能得到 `unknown`（这正是文档要求的三态语义，不是缺陷，但会显著限制
> 价格过滤的可用率）；`categories` 为空但 `main_category` 可用，品类约束仍可执行。

### 3.3 E5 编码与 FAISS 索引

```bash
agr index build --catalog artifacts/catalog/all_beauty_v2 --device cpu --batch-size 64
```

| 项 | 值 |
|---|---|
| 模型 | `intfloat/e5-base-v2`（768 维，`query: ` / `passage: ` 前缀） |
| 向量数 | **112,590** |
| 归一化 / 度量 | L2 归一化 + 余弦（`IndexFlatIP`） |
| index.faiss | 345,870,381 B（约 345.9 MB） |
| embeddings.npy | 112,590 × 768 float32（约 345.9 MB） |
| product_ids.jsonl | 3.27 MB，行序与索引行号严格一致 |
| manifest | 模型/维度/索引类型/哈希/条数，`agr index info` 可重算校验 |
| 编码耗时 | **约 17 分钟**（112,590 条，CPU，48 线程） |
| 吞吐 | 短文本桶约 290–300 条/s；长文本桶约 130 条/s；整体平均约 110 条/s |

**关键工程发现（可复现）**：首版按原始顺序批次编码，每个 batch 都 pad 到该 batch 最长文本，
在 CPU 上只有约 **15 条/s**（全量约 2 小时）。改为**按文本长度排序分桶**后（同一结果、
相同向量），吞吐提升约 **20 倍**，全量降到 17 分钟。该实现保留在
`retrieval/e5/artifacts.py`，并在本轮报告中如实记录。

### 3.4 索引校验

```bash
agr index info --bundle artifacts/faiss/e5-base-v2_all_beauty_v2
```

| 字段 | 值 |
|---|---|
| index_version | `e5-base-v2_all_beauty_v2` |
| catalog_version | `all_beauty_v2` |
| model_id / revision | `intfloat/e5-base-v2` / `main` |
| embedding_dim | 768 |
| metric / index_type | cosine / `IndexFlatIP` |
| num_vectors | 112,590 |
| index_sha256 | 见服务器 manifest（`agr index info` 输出） |

校验逻辑：重算 `index.faiss` 与 `product_ids.jsonl` 的 SHA-256、核对行数、核对
编码器维度与 manifest 一致；任何不一致都会抛 `ArtifactError`。

机器可读证据：[index_manifest_v2.json](evidence/index_manifest_v2.json)、
[search_results_v2.json](evidence/search_results_v2.json)、
[search_latency_v2.json](evidence/search_latency_v2.json)。

### 3.5 真实检索演示

命令（三 Query，走 E5 内部 RRF）：

```bash
agr search --bundle artifacts/faiss/e5-base-v2_all_beauty_v2 \
  --catalog artifacts/catalog/all_beauty_v2 \
  --query "lightweight daily moisturizer for sensitive skin" \
  --query "fragrance free face cream under 30 dollars" \
  --query "hydrating moisturizer for dry skin" --top-k 5
```

**三 Query 融合结果 Top-5（真实输出，RRF k=60）**

| rank | product_id | 相似度（代表 Query） | 价格 | 标题（截断） | 代表 Query |
|---|---|---|---|---|---|
| 1 | `B0757RW9XK` | 0.868960 | — | Daily Face Moisturizer Serum - Hydrating Vitamin Lotion… | lightweight daily moisturizer for sensitive skin |
| 2 | `B01MR7FQB8` | 0.860771 | — | Shi Hydrate, Facial Moisturizer | lightweight daily moisturizer for sensitive skin |
| 3 | `B00VU0I1ZK` | 0.833735 | 275.33 | Aveeno Positively Radiant Daily Face Moisturizer SPF 30 (Pack of 12) | fragrance free face cream under 30 dollars |
| 4 | `B010GAN49C` | 0.860671 | — | Dr. Karabelnik Collagene Moisturizing Day Cream | hydrating moisturizer for dry skin |
| 5 | `B002SPZ29E` | 0.849415 | — | basqnyc Mega Moisture Cream Fragrance Free Tube | fragrance free face cream under 30 dollars |

**单 Query Top-3（内部 RRF 之前）**

| Query | Top-3 product_id（相似度） |
|---|---|
| lightweight daily moisturizer for sensitive skin | `B00YF3J4OG` (0.882341)、`B014T3QBLK` (0.881399)、`B0147KM3ZI` (0.877630) |
| fragrance free face cream under 30 dollars | `B002SPZ29E` (0.849415)、`B002K4NP1U` (0.844102)、`B071CZTDCR` (0.841553) |
| hydrating moisturizer for dry skin | `B094XKS2QY` (0.877271)、`B01AE6ARXO` (0.877056)、`B01GSGDRR2` (0.871318) |

**延迟（服务器实测）**

| 场景 | 延迟 |
|---|---|
| CLI 冷启动（加载模型 + 首次推理 + 3 Query） | 13,753.84 ms（`agr search` 记录的 `latency_ms`） |
| 进程内复用模型，3 Query 检索（5 次测量） | **p50 33.3 ms**（min 29.8 / max 185.2） |
| 加载 + manifest 哈希校验 + 索引装载 | 1.0 s |
| 融合候选数 | 146（3×50 去重后） |

**观察（直接写入 Phase 2 的动机，不夸大结论）**：

1. 语义召回确实返回了与 “moisturizer / sensitive skin / fragrance free” 相关的真实商品；
   同一商品可被多个 Query 命中，RRF 会把它排到前面（本样例 rank 1/2 都来自 Query 1）。
2. 价格缺失非常普遍：Top-5 中 4 个商品没有价格，只有 Aveeno 有 `275.33`（Pack of 12），
   却因为语义匹配进入第 3 —— 这正是**必须由 Python Hard Constraint Evaluator 拦截**的
   案例，也是 Phase 2 的直接验收场景；在 Phase 2 之前不能宣称这些候选“满足预算”。
3. `index_sha256 = 09cc6874…3697`、`product_ids_sha256 = 42809860…1ed9`、
   `catalog_manifest_sha256 = 5c047593…2d62`（完整值见服务器 `manifest.json`），
   v2 索引 112,590 条、768 维、cosine/IndexFlatIP。
4. 服务器 `python -m pytest -q`：**112 passed**（111 单元 + 1 真实 FAISS 集成）。

---

## 4. 复现路径

```bash
# 本机
python scripts/sync_to_server.py
# 服务器（只需一次；重启后重建 venv）
bash scripts/bootstrap_s4000.sh
source scripts/env_s4000.sh
agr catalog download --category All_Beauty
agr catalog build --meta /mnt/agr-data/data/raw/meta_All_Beauty.jsonl \
  --category All_Beauty --version all_beauty_v2
agr index build --catalog artifacts/catalog/all_beauty_v2 --device cpu --batch-size 64
agr index info --bundle artifacts/faiss/e5-base-v2_all_beauty_v2
agr search --bundle artifacts/faiss/e5-base-v2_all_beauty_v2 \
  --catalog artifacts/catalog/all_beauty_v2 --query "..." --top-k 5
```

所有 manifest 记录源文件哈希、模型、模板版本与产物哈希；重新运行脚本可复现同一
商品集合与索引（编码 float 值受 CPU/线程调度影响可能存在末位差异，检索排名稳定）。

---

## 5. 测试结果

| 环境 | 命令 | 结果 |
|---|---|---|
| 本地 Windows / Python 3.13 | `python -m pytest -q` | **111 passed**（单元；集成测试因缺 faiss 跳过） |
| 服务器 / Python 3.10 venv | `python -m pytest -q` | **全部通过**（单元 + 真实 FAISS bundle 集成测试） |
| 本地 | `ruff check src tests` | All checks passed |
| 本地 | `mypy` | Success: no issues found in 45 source files |
| 服务器 | `agr index info` | 哈希与行数校验通过 |

集成测试 `tests/integration/test_e5_bundle_faiss.py`：小 catalog → 真实 FAISS 构建 →
manifest 哈希校验 → 加载 → 双 Query 检索 → 自身命中校验；**不使用 Mock 检索结果**，
只用一个确定性编码器替代模型（测试目的），该替代显式可见、不进入正式实验。

---

## 6. 限制与未验证项

1. **仅 E5 语义召回**：TIGER 未实现（Phase 4 阻塞，需决策 Q1），因此 `tiger_candidates`
   在本轮没有真实结果；任何“双路召回”结论都不能从 Phase 1 得出。
2. **无 Reranker / Judgment / Controller**：候选池之后的链路尚未实现。
3. **数据覆盖**：price 15.7%、description 17%、features 15.4% —— 硬约束与精排文本
   会被明显限制；`unknown` 语义因此尤其重要。
4. **CPU 编码**：MUSA 实测可跑 E5（E5-small ≈ 1700–3000 条/s），但本轮未接入；
   全量 CPU 编码约 17 分钟。
5. **冷启动延迟**：CLI 每次调用需加载模型（约 10 s）+ 首次推理预热；进程内复用模型时
   3 条 Query 的检索为毫秒级（FAISS 精确检索）。
6. **未验证**：Qwen3-Reranker-0.6B 在 MUSA/CPU 的接口与吞吐；Mem0/Qdrant 部署；
   TIGER 权重来源；多品类扩展。
