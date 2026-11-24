# Agentic Generative Recommendation

面向商品购物需求的生成式推荐系统：Planner 解析需求 → TIGER（Semantic ID 生成式召回）
与 E5 + FAISS（语义召回）双路召回 → Candidate Fusion（两级 RRF）→ 硬约束校验 →
Qwen3-Reranker-0.6B 精排 → DeepSeek Judgment 逐商品判断 → Adaptive Controller
（LangGraph + Python）决定 REPLAN / STOP → DeepSeek Generator 输出有证据的推荐 →
有效反馈写入 Mem0（语义记忆）与 SQLite（结构化交互事件）。

工程约束见 [AGENTS.md](AGENTS.md)；六份 V1.1 设计文档在 [docs/](docs/)。
当前是 **Agentic Generative Recommendation** 项目，**不是**旧版 JEV / Wikipedia
多跳检索项目；Amazon Reviews 2023 是当前数据来源。

## 当前状态（Phase 0 + Phase 1）

| 模块 | 状态 | 说明 |
|---|---|---|
| 工程骨架 / 配置 / 日志 / trace | ✅ Phase 0 | `src/recommender/config.py`、`observability/` |
| 跨模块 Pydantic 契约 | ✅ Phase 0 | `src/recommender/schemas/`（Plan / Candidate / Constraint / Judgment / Controller / Memory / Eval） |
| 商品目录（Amazon Reviews 2023） | ✅ Phase 1 | 原始 metadata → `ProductRecord` + 双文本模板 + 版本化 manifest |
| E5 + FAISS 索引与检索 | ✅ Phase 1 | 真实 All_Beauty **112,590** 商品：CPU 编码（768 维）、L2 归一化、`IndexFlatIP`、多 Query 内部 RRF；索引带 manifest/哈希 |
| Candidate Fusion + 唯一硬约束 Evaluator + 候选池 | ✅ Phase 2 | 跨路 RRF、去重、来源保留、三态约束、eligible/unverified/rejected 分区；真实 E5 候选池演示 99→15 pass / 75 unknown / 9 fail |
| Qwen3-Reranker-0.6B 精排 | ✅ Phase 3 | Transformers 原生 CPU（yes/no logits）；真实候选池 15 eligible → 精排 10.3 s；分数仅用于同 Query 排序 |
| TIGER 生成式召回 | ⛔ No-Go（D11） | A1/B1/B2 均未通过 D11（TIGER Recall@10 0.15%/0.25%/0.07% vs popularity 0.34–0.82%、E5-profile 2.9–4.4%）；未全量训练，Phase 4 未完成（[报告](docs/phase4/D1_B1_B2_GATE_REPORT.md)） | 当前只有配置占位；**没有** Mock 冒充正式召回 |
| Reranker / Planner / Judgment / Controller / Generator / Memory | ⏳ Phase 3–6 | 只有契约与配置，无实现 |
| 端到端评测 | 🟡 Phase 7 准备 | D10 三类评测协议与模板完成；人工 Query 尚未标注 |

可复现证据、命令与指标见 [docs/audit/phase1-report.md](docs/audit/phase1-report.md)；
审计与实施计划见 [docs/audit/repository-audit-and-plan.md](docs/audit/repository-audit-and-plan.md)。

需要负责人决定的事项（仅事实与选项，不含推荐意见）：
- 当前待决（D1–D11）：[docs/decisions/OPEN_DECISIONS_PHASE4-7.md](docs/decisions/OPEN_DECISIONS_PHASE4-7.md)
- 历史提问记录：[docs/decisions/open-questions.md](docs/decisions/open-questions.md)

## 快速开始

```bash
python -m pip install -e ".[dev]"      # 契约/配置/测试（不装 torch/faiss）
python -m pytest -q                    # 111 个单元测试（Phase 0/1 逻辑，不下载模型）
ruff check src tests && mypy           # 静态检查
agr config validate                    # 配置与路径自检
```

检索链路（需要 `retrieval` 额外依赖 + 真实数据）：

```bash
cp .env.example .env                    # 填 DEEPSEEK_API_KEY；HF_ENDPOINT 用镜像
export HF_ENDPOINT=https://hf-mirror.com

agr catalog download --category All_Beauty
agr catalog build --meta data/raw/meta_All_Beauty.jsonl --category All_Beauty --version all_beauty_v2
agr index build --catalog artifacts/catalog/all_beauty_v2
agr index info --bundle artifacts/faiss/e5-base-v2_all_beauty_v2
agr search --bundle artifacts/faiss/e5-base-v2_all_beauty_v2 \
  --query "lightweight daily moisturizer" --query "fragrance-free face cream" --top-k 5
```

## S4000 部署

代码在独立 Git 仓库 `/home/agr/Agentic-Generative-Recommendation`；原始数据、HF Cache、
catalog/FAISS 产物与大型日志在 12 TB NFS `/mnt/agr-data`（Q10）；venv 在
`/tmp/agr/venv`（重启后按 `scripts/bootstrap_s4000.sh` 重建）。

```bash
# 本机（Windows）
python scripts/sync_to_server.py
# 服务器
bash scripts/bootstrap_s4000.sh          # 需要 ~/.local/bin/uv（宿主没有 python3-venv）
source scripts/env_s4000.sh              # HF_HOME / hf-mirror / venv
```

## 目录结构

```text
configs/          五份可校验 YAML + 机器本地 local.yaml（gitignored）
docs/             V1.1 设计文档、audit、decisions、architecture
src/recommender/
  schemas/        跨模块 Pydantic 契约（唯一权威）
  catalog/        Amazon metadata 接入、商品目录、商品文本
  retrieval/e5/   E5 编码器、FAISS 索引、多 Query RRF 检索
  fusion/         RRF（Phase 2 跨路融合复用）
  observability/  JSON 日志、request/trace 上下文、JSONL trace
  planner/ judgment/ reranking/ controller/ generation/ memory/ ...（Phase 3–6）
scripts/          sync / bootstrap / env / 数据与索引脚本
tests/unit/       契约与纯逻辑测试（默认无外部依赖）
tests/integration/ 需要模型/索引/服务的集成测试
```

## 关键契约（详见 `src/recommender/schemas/`）

- 商品主键统一 `product_id`，Amazon `parent_asin` 在接入层映射并保留。
- `RecommendationPlan` 是 Planner / Replanner / Retrieval / Reranker / Controller 的唯一计划协议。
- 硬约束只有唯一的 Python Evaluator，结果 `pass` / `fail` / `unknown`；`unknown` 不算满足。
- 两级 RRF：E5 多 Query 内部先融合，再与 TIGER 跨路融合；不直接相加异构分数。
- Judgment 只判断，Controller 只决策（REPLAN / STOP / CLARIFY），预算耗尽 ≠ 质量达标。
- Mem0 语义记忆与 SQLite `InteractionEvent`（TIGER 序列）严格分离；模拟数据必须 `is_synthetic=true`。

## 已知限制

- Phase 1 只跑通 E5 + FAISS 语义召回；TIGER 尚无真实实现（禁止 Mock 冒充）。
- E5 当前在 CPU 上编码（全量约 17 分钟）；MUSA GPU 实测可跑 E5，但尚未接入编码器。
- Amazon All_Beauty 元数据覆盖不均：price 15.7%、description 17.0%、features 15.4%（title/main_category 100%，details 96%），硬约束多数会得到 `unknown`。
- Controller 阈值尚未用验证集校准（`controller.quality.calibrated=false`）。
- Qwen3-Reranker-0.6B 在 MUSA 上的推理尚未验证（Phase 3 前置冒烟测试）。
