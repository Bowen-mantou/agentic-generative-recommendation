# ADR-0003: Hard-constraint status has exactly three states

- **状态**：已采纳
- **日期**：2026-11-24
- **上下文**：文档中出现三种不一致的 `constraint_status` 形状：
  1. 候选池里是字符串 `"pass"`（模块三 §5.6）；
  2. Judgment 输入里是按字段的字典 `{"price": "pass", "category": "pass"}`（最终阶段 §3.3）；
  3. Python 示例在“没有配置该约束”时返回第 4 态 `"not_required"`（最终阶段 §3.5）。
  AGENTS §2.4 明确只有 `pass` / `fail` / `unknown` 三态，且 `unknown` 不得当作满足。

## 决策

1. 唯一 Python Constraint Evaluator 的**每商品聚合状态**只有三态，枚举
   `ConstraintStatus = {pass, fail, unknown}`。
2. “该约束未配置”（文档中的 `not_required`）**不进入状态枚举**，而是表现为
   **没有对应的 `ConstraintCheck`**；此时整体状态为 **空（`None`）**，不得伪装为 `pass`
   （2026-11-24 决策 Q3 最终确认）。
3. 逐字段信息保留在 `ConstraintCheck(field, operator, expected, observed, status, reason)`
   列表中；Judgment 消费的是上游 `ConstraintStatus` + `ConstraintCheck[]`，**不重新做数值比较**。
4. 只有全部 check 为 `pass` 的商品进入 `CandidatePool.candidates`；`fail` 进入
   `rejected_candidates`（诊断用），`unknown` 进入 `unverified_candidates`（可用于 Replan，
   但不计入有效候选数）。

## 理由

- 三态符合 AGENTS 的硬性规定，避免把“未配置”误算成“满足”或“未知”；
- 空 check → `None`，与“已验证通过”严格区分；有效候选数 = pass + not_configured；
- 用空 check 表达“无约束”比增加第 4 态更简单，聚合规则（fail > unknown > pass）保持单调；
- per-field 信息不丢失，Judgment 与 Controller 都能追溯证据。

## 备选与取舍

- 增加 `not_required` 第 4 态：会让 `pass` 的语义分裂（“满足” vs “无需满足”），且与
  AGENTS §2.4 冲突，故不采用。
- Judgment 输入使用整体状态 + checks（当前），而不是嵌套 dict；如果实验证明逐字段 dict
  更利于 Prompt，可在 Payload 层做投影，**不改变存储/候选池契约**。

## 影响

- `src/recommender/schemas/constraints.py` 与 `fusion.py` 的校验依赖该规则；
- Phase 5 Judgment Prompt 需要把 `ConstraintCheck[]` 作为只读证据传入。
