# Known Issues / 已知问题

- 日期：2026-11-25
- 范围：24 小时 E2E 主线（E5-only）
- 说明：以下问题不伪装为已解决；影响 Gate 的项单独标注。

| ID | 问题 | 影响 | 现状/后续 |
|---|---|---|---|
| K1 | TIGER D11 No-Go；E5-only 不是双路召回 | 研究结论/答辩口径 | 保留 No-Go 报告；不宣称双路验收；Q12 偏差待确认 |
| K2 | Controller 阈值未校准（`calibrated=false`） | 质量 STOP 的统计有效性 | 当前为工程初值；Phase 7 校准 |
| K3 | 全量 2,224 条 Reranker / Cluster Bootstrap 未跑 | 最终排序指标缺失 | 方案明确不阻塞 E2E；后续完成 |
| K4 | Mem0 `infer=True` LLM 抽取未验证 | 反馈自动抽取能力 | 当前固定 `infer=False`，只写显式反馈 |
| K5 | Qdrant 无鉴权/TLS，仅回环 | 共享服务器安全 | 保持 127.0.0.1；公网部署前必须加鉴权/TLS |
| K6 | Mem0 BM25/spaCy 未安装 | 混合检索不可用 | 当前 dense-only；不影响本轮 G5 |
| K7 | API 预算无 provider 级日上限 | 成本风险 | 单请求 Controller 预算 20 次 API；后续加日预算 |
| K8 | 真实 E2E 只有单 query/单用户、无并发压测 | 性能/鲁棒性证据有限 | 3 次重复运行；后续多 query/并发 |
| K9 | Generator 首轮英文请求曾返回中文 reasons | 语言一致性 | 已修复 prompt + 单测；`run_stop_initial` 保留为缺陷证据 |
| K10 | REPLAN 场景 Generator 返回 `fallback_template` | 无候选时无真实生成 | 正确行为：不伪造商品；不应算 G5 通过 |
| K11 | `demo_user_002` 测试反馈仍在 Qdrant | 测试数据残留 | 是显式 E2E 证据；清理需负责人确认，避免误删 |
| K12 | 全量评测数据集人工只抽样 67/2,224 | 结论代表性 | 已明确披露；不是全量人工确认 |
| K13 | Phase 7 v1.0 最终集与 E2E 用户/场景不同 | 不能相互替代 | E2E 是单请求链路验收；评测集是离线资产 |
| K14 | LangGraph 在 `/tmp` venv，重建需补装 | 环境可复现性 | runbook 记录；bootstrap 需补 workflow/memory extra |
| K15 | 服务器文档使用占位路径/IP | 公开仓库脱敏 | 运行用真实 NFS 路径；公开文档已用占位 |

## 不应出现的说法

- ❌ “双路召回验收通过”；
- ❌ “TIGER 已完成/全量训练成功”；
- ❌ “2,224 条全部人工复核”；
- ❌ “Controller 阈值已校准”；
- ❌ “Generator/REPLAN 的 fallback 等于真实成功”。
