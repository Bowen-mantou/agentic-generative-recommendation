# Architecture Overview

> 状态：Phase 0 + Phase 1（E5 语义召回基线）。TIGER / Reranker / Planner / Judgment /
> Controller / Generator / Memory 仅有契约与配置，尚未实现。设计依据见 `docs/` 六份 V1.1
> 文档与 `docs/audit/repository-audit-and-plan.md`。

## 1. 目标架构（AGENTS.md §1）

```text
User Input + Mem0 User Context
        │
        ▼
 Planner / Replanner (DeepSeek) ──► RecommendationPlan (唯一 Pydantic 协议)
        │
   ┌────┴──────────────┐
   ▼                   ▼
 TIGER               E5 + FAISS
 (Semantic ID)       (自然语言 Query)
   └────┬──────────────┘
        ▼
 Candidate Fusion（E5 内部 RRF → 跨路 RRF → 去重 + 来源/排名保留）
        ▼
 Python Hard Constraint Evaluator（唯一实现，pass/fail/unknown）
        ▼
 Qwen3-Reranker-0.6B（冻结；BGE-v2-m3 对照）
        ▼
 Judgment (DeepSeek，逐商品结构化判断)
        ▼
 Adaptive Controller（LangGraph + Python；REPLAN / STOP / CLARIFY）
        │                    │
    REPLAN                  STOP
        │                    ▼
   Replanner          Recommendation Generator (DeepSeek)
                             │
                     User Feedback（有效反馈才写入）
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
        Mem0（语义记忆）          SQLite InteractionEvent（TIGER 序列）
```

## 2. 已实现范围（Phase 0/1）

```text
configs/*.yaml ─► config.AppConfig（YAML 深合并 + AGR__ 环境覆盖 + local.yaml 覆盖）
                      │
Amazon meta.jsonl ─► catalog.amazon ─► catalog.build ─► artifacts/catalog/<ver>/
                      │                                  products.jsonl
                      │                                  texts.jsonl（embedding/rerank 双模板）
                      ▼
                 retrieval.e5.encoder（E5，懒加载）─► retrieval.e5.artifacts.build_bundle
                                                         │
                                                         ▼
                                              artifacts/faiss/<model>_<catalog>/
                                                index.faiss + product_ids.jsonl
                                                + embeddings.npy + manifest.json
                                                         │
Planner queries ─► retrieval.e5.service.E5Retriever ◄────┘
                      │  每条 Query 独立检索
                      │  fusion.rrf（E5 内部 RRF）
                      ▼
                 E5PathOutput(fused=RecallResult, per_query=[...])
```

## 3. 模块职责与边界

| 包 | 职责 | 明确不做 |
|---|---|---|
| `schemas/` | 所有跨模块数据契约（Pydantic，`extra=forbid`）；Plan/候选池/约束/精排/判断/状态/记忆/评测 | 不含业务逻辑 |
| `catalog/` | Amazon metadata 解析、`parent_asin → product_id`、商品文本模板、catalog manifest | 不生成 Semantic ID；不推断缺失属性 |
| `retrieval/e5/` | E5 编码、FAISS 索引构建/校验、多 Query 检索 + 内部 RRF | 不做硬约束、不重排、不生成理由 |
| `fusion/rrf.py` | 纯函数 RRF（两级共用）；只依赖排名 | 不读取原始分数做加权求和 |
| `observability/` | JSON 日志、request_id/trace_id、JSONL trace、脱敏 | 不做业务决策 |
| `utils/` | 错误类型、原子 IO、哈希 | — |
| `cli.py` | `agr version/config/catalog/index/search` | 不做隐式网络/模型初始化 |

## 4. 配置与版本

- 配置优先级：`configs/*.yaml`（深合并，重复键报错）< `configs/local.yaml`（机器本地，
  gitignored）< `AGR__SECTION__FIELD` 环境变量。
- 密钥只从环境变量/`.env` 读取（`Secrets`），不进入配置快照与 trace。
- 索引版本 = bundle 目录名；`manifest.json` 记录模型、revision、维度、前缀、索引类型、
  catalog 版本、文本模板版本、行数、三个 SHA-256（index / product_ids / catalog manifest）。
- `agr index info` 会重算哈希并校验行数，防止“索引与映射错位”。

## 5. 观测与可复现

- 每个请求：`request_id` + `trace_id`（contextvars），日志与 trace 自动携带；
- trace 写入 `runs/<request_id>/trace.jsonl`，按 seq 递增、UTC 时间戳、敏感字段脱敏；
- 所有数据/模型/索引产物都有 manifest 与哈希；实验需记录 manifest 与 git commit。

## 6. 测试策略

| 层 | 位置 | 依赖 | 说明 |
|---|---|---|---|
| 单元 | `tests/unit/` | 无外部服务 | 契约不变量、配置优先级、RRF、catalog 解析、E5 服务（stub encoder/index）、日志/trace |
| 集成 | `tests/integration/` | faiss（服务器 venv） | 小 catalog → 真实 FAISS bundle → 校验 → 检索 |
| 真实 API | 待 Phase 5 | DeepSeek key | 显式运行，不进默认 CI |
| E2E | 待 Phase 7 | 全链路 | 正常 / Replan / 预算耗尽 / 澄清 / API 失败 / 无候选 |

## 7. 阶段映射

Phase 0 工程骨架与契约 → **完成**；Phase 1 E5 召回基线 → **完成**；
Phase 2 融合与硬约束 → 待开始；Phase 3 Reranker → 待开始；Phase 4 TIGER → 阻塞（Q1）；
Phase 5 Planner/Judgment/Controller → 待开始；Phase 6 Generator/Memory → 待开始；
Phase 7 评测 → 待开始。
