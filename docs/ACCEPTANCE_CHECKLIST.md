# 24 小时 E2E 验收清单

- 日期：2026-11-25
- 对应方案：`/c/Users/15976/Downloads/Agentic_Generative_Recommendation_下一阶段24小时自主执行方案_V1.0.docx`
- 结论：**E5-only 真实 E2E G0–G6 通过；TIGER 仍 No-Go，Q12 偏差待外部确认。**

## 主线功能

- [x] `agr recommend` 真实执行（`docs/e2e/run_stop_latest_summary.json`）
- [x] Memory Read 在 Planner 前完成；空记忆/1 条记忆均可
- [x] DeepSeek Planner 真实调用并通过 Schema（`plan_initial`，2 条 E5 Query）
- [x] E5/FAISS 返回真实商品（78/117 eligible）
- [x] Candidate Fusion、Hard Constraints 真实执行（REPLAN 场景 0/149/38 分区）
- [x] Qwen3-Reranker 真实执行（50 候选评分，24.4 s）
- [x] Evidence Builder 使用真实商品字段（10 候选）
- [x] Judgment 对真实候选真实调用（10 条真实 DeepSeek 结果）
- [x] LangGraph Controller 真实运行；quality STOP 有 Trace
- [x] REPLAN 真实调用 Replanner 并二次检索；`budget_exhausted` STOP 有 Trace
- [x] Generator 真实调用（`deepseek-flash`，5 条）；商品均来自 selected
- [x] Generator 理由引用证据；修复英文请求返回英文 reasons
- [x] Memory Read/Add；`demo_user_002` 反馈写入并可读回
- [x] 用户隔离：其他用户读取 0 条
- [x] 显式反馈写回；未把推测当反馈

## 工程质量

- [x] 原有测试未删除/屏蔽；本地 `308 passed / 15 skipped`
- [x] 真实 E2E 集成测试在服务器 `1 passed`（35.9 s）
- [x] CI Python 3.10/3.11 通过（提交后）
- [x] 至少 1 条完整真实 E2E Trace；实际 3 次完整运行
- [x] API 预算/超时/重试/循环上限已配置（单请求 API 20 次、max_replans=1）
- [x] Qdrant 未暴露公网；密钥未进入 Git
- [x] Git commit、配置、Prompt、数据版本已记录
- [x] `agr feedback` 可运行；模拟数据可标记

## 研究与验收口径

- [x] 明确 TIGER Phase 4 D11 No-Go、本轮 E5-only 主线
- [x] 未宣称双路召回验收通过
- [x] 未把 67 条人工复核描述成 2,224 条全量人工确认
- [x] 未把 Recall-only 结果描述为最终排序结论
- [x] 未宣称 Controller 阈值已校准（`calibrated=false`）

## 尚未通过/未完成

- [ ] 2,224 条全量 Reranker 与 User Cluster Bootstrap（方案明确不阻塞本轮）
- [ ] Q12 偏差的课程/老师确认
- [ ] Controller 统计校准
- [ ] Mem0 `infer=True`、BM25、并发压测
- [ ] Qdrant 鉴权/TLS 与备份调度

## 证据路径

- `docs/E2E_INTEGRATION_REPORT.md`
- `docs/E2E_RUNBOOK.md`
- `docs/e2e/run_stop_latest_*`、`run_stop_previous_*`、`run_stop_initial_*`
- `docs/e2e/run_replan_*`
- `docs/e2e/feedback_writeback.json`
- NFS：`/mnt/agr-data/runs/e2e/<request_id>/`
