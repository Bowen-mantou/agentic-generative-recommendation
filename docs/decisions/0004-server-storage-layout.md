# ADR-0004: S4000 storage and runtime layout

- **状态**：已采纳
- **日期**：2026-11-24
- **上下文**：S4000 宿主 `/home` 只剩 3.7 GB（共 191 GB，已被其他项目占用 178 GB）；
  `python3 -m venv` 因缺少 `ensurepip/venv` 不可用；`/data` 为 root-only；`/tmp` 在
  122 GB 空闲的根分区上；`/mnt/agr-data` 是 12 TB 可写 NFS（370 MB/s）；
  HuggingFace 直连超时，`hf-mirror.com` 可用；pip 走清华镜像 ~119 MB/s。

## 决策

| 资源 | 位置 | 理由 |
|---|---|---|
| 代码仓库 | `/home/agr/Agentic-Generative-Recommendation` | 小（< 5 MB），用户可见 |
| Python venv | `/tmp/agr/venv` | 需 ~1.5 GB，`/home` 承受不起；重建脚本一键恢复 |
| uv 缓存 | `/tmp/agr/uv-cache` | 首次安装曾占满 `/home` 6 GB（已清理并重定向） |
| 原始数据（metadata/reviews） | `/mnt/agr-data/data/raw` | 12 TB 持久 NFS |
| HF 模型缓存 | `/mnt/agr-data/hf`（`HF_HOME`） | 避免 440 MB+ 权重写入 `/home` |
| catalog / FAISS / embeddings | 项目内 `artifacts/` | 约 1.2 GB，便于随代码检查；manifest 可追溯 |
| 运行 trace | 项目内 `runs/` | JSONL 小文件，按 request_id 分目录 |

约定：`scripts/env_s4000.sh` 统一导出 `HF_ENDPOINT=https://hf-mirror.com`、`HF_HOME`，
并激活 venv；`configs/local.yaml`（gitignored）把 `paths.raw_dir` / `paths.interim_dir`
指向 NFS。

## 后果与风险

- `/tmp` 是易失的：重启后必须重跑 `scripts/bootstrap_s4000.sh`；数据与产物在
  持久卷上，不受影响。
- NFS 无备份保证（env-recon 未验证备份策略）：原始数据可通过脚本 + SHA-256 重新下载，
  catalog/manifest 记录了源文件哈希。
- `/home` 只剩约 8.9 GB（清理 uv 缓存后），后续若引入 Reranker 权重（~1.2 GB）或多品类
  数据，需要迁移产物到 NFS——决策点 Q10。
