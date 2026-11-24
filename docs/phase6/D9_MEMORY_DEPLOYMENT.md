# D9：Memory 部署验证（独立 Qdrant + 本地多语言 Embedding + NFS 快照备份）

| 项 | 值 |
|---|---|
| 文档 | `docs/phase6/D9_MEMORY_DEPLOYMENT.md` |
| 对应任务 | 共享任务板 `task-9`（D9）；Q7（Memory 部署）、R10、R11 |
| 执行时间 | 2026-11-24 15:54 – 16:30（Asia/Shanghai） |
| 执行人 | memory-deploy（Agent） |
| 服务器 | `192.0.2.10`（hostname `agr-worker`），用户 `agr`（uid 1003，属 `docker` 组） |
| 代码基线 | 本地 Git 仓库 `main` @ `4324ea9`（本次只新增 `docs/`、`scripts/memory/`，未改 `src/`、`configs/`、`cli.py`） |
| 结论 | **Qdrant 独立容器已部署并通过写入/检索/删除/重启恢复/NFS 备份恢复全部测试；`intfloat/multilingual-e5-base` 本地推理验证通过（768 维、中文单条查询 ~16 ms@64 线程 / ~33 ms@4 线程）；mem0ai 2.2.1 + qdrant-client 1.20.0 接口调研完成。** 决策状态：D9 推荐方案在本报告中有实测证据，最终确认仍需项目负责人批准。 |

> 本报告只记录**真实执行**过的命令与输出。凡是根据源码/签名推断、但没有在真实链路跑通的内容，均显式标为「未验证」。

---

## 0. 结论摘要

| 项目 | 结果 | 证据位置 |
|---|---|---|
| Qdrant 容器 | 运行中。`agr-qdrant`，镜像 `qdrant/qdrant:v1.20.2`，仅绑定 `127.0.0.1:6333/6334`，数据卷 `agr-qdrant-data`（本地盘 `/var/lib/docker/volumes/...`） | §3 |
| Qdrant 功能 | HTTP `/healthz`、`/collections`、建 collection、`user_id` payload index、写入、向量检索（含过滤）、按 ID 读取、按 filter/ID 删除、删除 collection：**15/15 通过** | §4.1 |
| 重启恢复 | `docker restart` 0.76 s，24 条数据、过滤检索结果在重启前后完全一致 | §4.2 |
| 多用户隔离 | `user_id` 过滤检索只返回该用户 payload；删除按 filter 精确生效（R11 的存储层前提） | §4.1 / §4.3 |
| 备份 | 快照写入 NFS `/mnt/agr-data/qdrant_backup`，408 064 B，sha256 记录在 manifest；恢复后 24 条数据与检索一致 | §5 |
| Embedding | `intfloat/multilingual-e5-base`，768 维，L2 归一化，确定性；单条中文 query 中位 16.23 ms（64 线程）/ 33.26 ms（4 线程）；批 32 条 397.8 条/s（64 线程） | §6 |
| 项目兼容 | 复用仓库真实 `E5Encoder`（md5 `60894343…`）加载该模型成功：768 维、`query:`/`passage:` 前缀、归一化正确 | §6.6 |
| Mem0 | mem0ai **2.2.1**、qdrant-client **1.20.0**；`search` 已改为 `filters=` 传 `user_id`（无 `user_id` 参数）；自定义 E5 embedder 需继承 `EmbeddingBase` 并在 `Memory` 构造后替换/注册；用 mem0 自带 `Qdrant` 类对真实 server 的 insert/search(filter)/update/delete round-trip **全部通过** | §7 |
| 收尾状态 | 测试 collection 已全部删除（Qdrant 当前为**空库**，等待 Phase 6 创建 `agr_memories`）；恢复演练快照保留在 NFS 作为证据 | §4.2、§5 |
| 对 Q7 符合性 | 独立服务 ✅、持久卷 ✅、单 Collection + `user_id` 过滤 ✅（已实测隔离）、本地 Embedding ✅、版本与维度可锁定 ✅；但 `Memory.add()` 全链路（含 DeepSeek LLM 抽取）**未验证** | §8 |
| 未验证项 | `Memory.add()` 端到端、自定义 embedder 注入 → `Memory` 的实际调用、BM25 混合检索、前缀对排序指标的收益、备份自动化调度 | §9 |

---

## 1. 前置检查（部署前状态）

### 1.1 Docker / 端口 / 资源

```bash
# 所有命令通过 cd /c/new/intern/plan/_work && python s4000_ssh.py "<cmd>" 在服务器执行
docker version --format '{{.Server.Version}}'   # 24.0.7
docker info --format 'DockerRootDir={{.DockerRootDir}} StorageDriver={{.Driver}}'
# DockerRootDir=/var/lib/docker StorageDriver=overlay2
ss -ltnp | grep -E ':(6333|6334)\b'            # 部署前：无监听（端口空闲）
nproc                                          # 128
free -g                                        # 总计 1007 GiB，可用 908 GiB
```

- Docker daemon `default-runtime` 为 `mthreads`（MUSA），`runc` 同时可用；Qdrant 容器**显式使用 `--runtime=runc`**，避免 MUSA runtime 干扰普通 amd64 镜像。
- 已存在容器（`dp`、`vllm21`、`p09-*`、`agx-rl`、`swebench-*`、`p09-registry` 等）**未做任何修改**；新容器名 `agr-qdrant`、卷名 `agr-qdrant-data` 均为项目独占前缀。

### 1.2 磁盘

| 挂载点 | 容量 | 可用 | 用途结论 |
|---|---|---|---|
| `/home` | 191 G | **8.9 G（96% 已用）** | 不再放活动数据库；仅代码/仓库 |
| `/var/lib/docker`（独立 LV） | 1.5 T | **84 G（95% 已用）** | **Qdrant 活动卷所在**（named volume） |
| `/`（含 `/tmp`） | 246 G | 120 G（49%） | `/tmp` 临时 venv、临时证据 |
| `/mnt/agr-data`（NFS `192.0.2.11:/data/agr-storage`） | 13 T | 12 T | **仅备份/快照**，不做活动数据库 |

> 说明：D9 推荐「活动数据优先本地持久盘」。`/home` 仅剩 8.9 G，按任务要求不硬塞；`/var/lib/docker` 有独立 84 G 可用空间，且 named volume 随 Docker 持久化，因此活动卷选 `/var/lib/docker/volumes/agr-qdrant-data`。NFS 只承载快照备份。

### 1.3 外网可达性（决定了镜像/模型/依赖的获取方式）

| 端点 | 结果 | 采用方案 |
|---|---|---|
| `registry-1.docker.io` | 连接超时 | 改用 pull-through 镜像 `docker.m.daocloud.io`（`/v2/` 匿名 401 属正常鉴权响应），本地再 tag 为 `qdrant/qdrant:v1.20.2` |
| `github.com` | 超时 | 不使用 |
| `pypi.org` | 超时 | 使用 `https://mirrors.aliyun.com/pypi/simple/`（`200`） |
| `hf-mirror.com` | `200` | 模型下载 `HF_ENDPOINT=https://hf-mirror.com` |
| `pypi.tuna.tsinghua.edu.cn` | 超时 | 未使用 |

---

## 2. 交付物（本次新增文件）

| 文件 | 作用 | 是否自动执行 |
|---|---|---|
| `scripts/memory/deploy_qdrant.sh` | 部署/启动/查看 Qdrant 容器；幂等；端口冲突时**拒绝启动**而不是抢占 | 否，手动执行 |
| `scripts/memory/backup_qdrant.sh` | 快照备份到 NFS、列出备份、从快照恢复（`backup`/`list`/`restore`/`collections`） | 否，手动执行 |
| `scripts/memory/qdrant_smoke_test.py` | 纯标准库 HTTP 冒烟测试（15 项断言），可反复执行 | 否，手动执行 |
| `scripts/memory/bench_e5.py` | multilingual-e5-base 加载/维度/前缀/延迟/内存基准 | 否，手动执行 |
| `scripts/memory/probe_mem0_api.py` | mem0ai/qdrant-client 版本、签名、配置字段与存储层 round-trip | 否，手动执行 |
| `docs/phase6/D9_MEMORY_DEPLOYMENT.md` | 本报告 | — |

所有脚本**不会**被应用启动流程、`src/`、cron 自动调用；本报告不包含任何密钥。`docs/phase6/` 与 `scripts/memory/` 均为本次任务的写 scope。

---

## 3. Qdrant 部署

### 3.1 镜像

```text
本地 tag:      qdrant/qdrant:v1.20.2
镜像 ID:       sha256:8870e3837712f612f997b683dee186a4d10fa48f1535377d007c2dd3429e74ef
manifest digest（拉取来源）:
               docker.m.daocloud.io/qdrant/qdrant@sha256:d277592721d2e3b73f9b797b53cf05e6a5bb2134f6d13fdbda4ca51782afbcac
Qdrant 版本:   1.20.2（镜像 label org.opencontainers.image.version=v1.20.2；`qdrant --version` 输出 1.20.2）
镜像大小:      197 MB
```

> 复现：`docker pull docker.m.daocloud.io/qdrant/qdrant:latest && docker tag docker.m.daocloud.io/qdrant/qdrant:latest qdrant/qdrant:v1.20.2`。`deploy_qdrant.sh` 已内置该回退逻辑（先镜像，后本地 tag）。

### 3.2 容器参数（实际值，来自 `docker inspect`）

```text
容器名:        agr-qdrant
容器 ID:       73ab20882183f480d0564089ebf6e04fc40bfe4aafdddfc56db5a0e6b17550df
镜像:          sha256:8870e3837712f612f997b683dee186a4d10fa48f1535377d007c2dd3429e74ef
runtime:       runc
restart:       unless-stopped
端口:          127.0.0.1:6333 -> 6333/tcp（HTTP）
               127.0.0.1:6334 -> 6334/tcp（gRPC）
卷:            agr-qdrant-data -> /qdrant/storage
               Source=/var/lib/docker/volumes/agr-qdrant-data/_data（本地盘，非 NFS）
环境变量:      QDRANT__TELEMETRY_DISABLED=true
               QDRANT__SERVICE__ENABLE_TLS=false
日志:          json-file，max-size=50m，max-file=3
```

完整等价命令行（`bash scripts/memory/deploy_qdrant.sh --dry-run` 的实际输出）：

```bash
docker run -d --name agr-qdrant --restart unless-stopped \
  -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v agr-qdrant-data:/qdrant/storage \
  -e QDRANT__TELEMETRY_DISABLED=true -e QDRANT__SERVICE__ENABLE_TLS=false \
  --log-driver json-file --log-opt max-size=50m --log-opt max-file=3 \
  --runtime=runc qdrant/qdrant:v1.20.2
```

### 3.3 部署后状态

```text
GET /healthz            -> healthz check passed
GET /                   -> version 1.20.2
GET /collections        -> {"result":{"collections":[]},"status":"ok"}
docker stats --no-stream-> mem_usage≈533–544 MiB / 1008 GiB, cpu≈1.3%
卷占用（空库）          -> 357 B
```

- 镜像默认将索引/向量持久化在 `/qdrant/storage`；named volume 由 Docker 管理，不落在 `/home`，重启/容器重建不丢数据（§4.2 实测）。
- 服务只在回环地址监听；`ss -ltn` 显示 `127.0.0.1:6333/6334`，**未暴露公网**（风险见 §9-R7）。

### 3.4 部署脚本行为

`scripts/memory/deploy_qdrant.sh`：

1. `docker info` 不可达则报错退出；
2. 检查 6333/6334：若被**其他**进程占用 → 打印监听行并退出（exit 1），绝不 kill 他人进程；若占用者是本容器 → 继续（幂等）；
3. 镜像缺失时先走 daocloud 镜像拉取，失败再直连 Docker Hub；
4. 卷不存在则创建（`docker volume create`）；
5. 容器不存在则 `docker run`，已存在则 `docker start`；
6. 轮询 `/healthz`（最多 120 s），失败时打印容器日志尾部；
7. 打印容器状态、镜像 ID、版本、collections、`docker stats`、卷大小。

支持 `status`（只读）、`--dry-run`、以及全部环境变量覆盖（`QDRANT_CONTAINER`、`QDRANT_IMAGE`、`QDRANT_VOLUME`、`QDRANT_HTTP_PORT`、`QDRANT_GRPC_PORT`、`QDRANT_BIND`、`QDRANT_RUNTIME` 等）。

**实测**：重复执行输出 `port 6333 is held by this container (agr-qdrant)` 并正常返回状态，未重建容器、未丢数据。

---

## 4. Qdrant 功能验证

### 4.1 冒烟测试（15/15 通过）

```bash
python3 scripts/memory/qdrant_smoke_test.py
```

```text
Qdrant smoke test @ http://127.0.0.1:6333, collection=agr_d9_smoke, dim=768, points=32
PASS healthz  [http://127.0.0.1:6333]
PASS server_info  [1.20.2]
PASS create_collection  [dim=768 distance=Cosine]
PASS create_payload_index_user_id  [{'result': {'operation_id': 2, 'status': 'acknowledged'}, ...}]
PASS upsert_points  [32 points]
PASS count_after_upsert  [points_count=32]
PASS search_self_hit  [api=query top_id=1 score=1.0]
PASS search_with_user_id_filter  [hits=5]
PASS retrieve_point_by_id  [{'user_id': 'u001', 'text': '示例记忆 1 / memory 1', 'seq': 1}]
PASS delete_points_by_filter
PASS count_after_filter_delete  [points_count=16]
PASS delete_point_by_id
PASS deleted_user_returns_no_hits  [hits=0]
PASS collection_vector_config  [{'size': 768, 'distance': 'Cosine'}]
PASS delete_collection
15/15 checks passed
SMOKE TEST OK
```

覆盖点：`/points/query`（Qdrant 1.20 新查询 API，与 mem0 2.2.1 的 `client.query_points` 一致）、`user_id` keyword payload index、按 filter 删除、按 ID 删除、collection 向量参数核验。

### 4.2 重启恢复

测试集合 `agr_d9_persist`：768 维、24 点、`user_id` 分 `u001`/`u002`、带 `seq`/`text` payload。

```text
=== before restart ===
collection=agr_d9_persist points_count=24 top_id_for_seq7=7 hits=3 payload={'user_id': 'u001', 'seq': 7, 'text': 'memory-7'}
PERSIST_CHECK OK
=== restarting container ===
agr-qdrant
real    0m0.756s
health after restart: healthz check passed
=== after restart ===
collection=agr_d9_persist points_count=24 top_id_for_seq7=7 hits=3 payload={'user_id': 'u001', 'seq': 7, 'text': 'memory-7'}
PERSIST_CHECK OK
```

结论：容器重启后 collection、payload、向量与过滤检索结果完全保留；从 `docker restart` 到 `/healthz` 恢复 < 1 s（空载）。

> 该测试 collection `agr_d9_persist` 在证据采集完成后已删除，Qdrant 当前为空库；其快照仍保存在 NFS（§5）。

### 4.3 用户隔离（R11 的存储层验证）

- 写入时 `u001`/`u002` 各 12 条；`filter={must:[{key:"user_id", match:{value:"u001"}}]}` 检索只命中 `u001`（5/5）；
- 按 `user_id` filter 删除后 `points_count` 从 32 → 16，再按 ID 删除 → 查询 `u001` 返回 0 条；
- 冒烟测试断言 `search_with_user_id_filter` 的**全部**命中 payload 都是目标用户。

> 这证明 Qdrant 层过滤语义正确；Mem0 适配层是否始终带 `user_id` 过滤仍需 Phase 6 adapter 测试（§7.5、§9）。

---

## 5. 快照备份与恢复（NFS）

### 5.1 备份

```bash
bash scripts/memory/backup_qdrant.sh backup
```

实际输出与产物：

```text
[backup_qdrant] agr_d9_persist: downloading agr_d9_persist-4764626224893275-2026-11-24-08-00-07-0.snapshot
[backup_qdrant] agr_d9_persist: saved 408064 bytes sha256=5bab5925eb3cba1d...
[backup_qdrant] backup finished (dir=/mnt/agr-data/qdrant_backup, manifest=.../manifest.jsonl)
```

NFS 目录结构：

```text
/mnt/agr-data/qdrant_backup/
├── manifest.jsonl
└── agr_d9_persist/
    ├── agr_d9_persist-4764626224893275-2026-11-24-08-00-07-0.snapshot        (408064 B)
    └── agr_d9_persist-4764626224893275-2026-11-24-08-00-07-0.snapshot.sha256
```

manifest 单行示例：

```json
{"collection":"agr_d9_persist","snapshot":"agr_d9_persist-4764626224893275-2026-11-24-08-00-07-0.snapshot","file":"/mnt/agr-data/qdrant_backup/agr_d9_persist/...snapshot","bytes":408064,"sha256":"5bab5925eb3cba1d6cc4ae227f471203dba83adda96c2543163403d66e169fd7","created_at":"2026-11-24T08:00:07Z","qdrant_version":"1.20.2","image_id":"sha256:8870e3837712f612f997b683dee186a4d10fa48f1535377d007c2dd3429e74ef"}
```

实现要点：

- 使用 Qdrant **snapshot API**（`POST /collections/{c}/snapshots` → `GET .../snapshots/{name}`），保证备份是服务端一致的磁盘副本，不直接拷贝活动存储目录；
- 下载后计算 `sha256` 并写 `.sha256` 伴生文件与 `manifest.jsonl`（含 bytes、时间、Qdrant 版本、镜像 ID，便于审计与版本追溯）；
- 下载成功后尽力调用 `DELETE .../snapshots/{name}` 释放活动盘上的服务端副本（NFS 文件才是备份正本）；
- 每个 collection 仅保留最近 `KEEP`（默认 14）份快照，自动清理更旧文件及校验文件；**保留策略已实测**：临时 collection 连续备份两次并设 `KEEP=1`，第二次自动输出 `pruning .../…-1.snapshot`，目录最终只剩最新一份（`…-2.snapshot` + `.sha256`）；
- `list` / `collections` 子命令用于巡检。

### 5.2 恢复

```bash
# 先删除 collection 模拟灾难
curl -fsS -X DELETE http://127.0.0.1:6333/collections/agr_d9_persist
bash scripts/memory/backup_qdrant.sh restore agr_d9_persist \
  /mnt/agr-data/qdrant_backup/agr_d9_persist/agr_d9_persist-...-0.snapshot
```

```text
uploading .../agr_d9_persist-...-0.snapshot -> collection agr_d9_persist
restored; points_count=24
=== verify after restore ===
collection=agr_d9_persist points_count=24 top_id_for_seq7=7 hits=3 payload={'user_id': 'u001', 'seq': 7, 'text': 'memory-7'}
PERSIST_CHECK OK
```

恢复前 `sha256sum` 与 manifest 一致（`5bab5925…169fd7`）；恢复后 24 条数据、过滤检索结果与备份前一致。

### 5.3 运维建议

- 备份调度**尚未配置**（脚本可复用，但未加 cron/systemd timer，也未纳入应用启动）；Phase 6 结束前需决定：cron（`KEEP`/频次）或由部署流水线在写入稳定后触发。
- 恢复路径会**覆盖**同名 collection，只允许在维护窗口执行；恢复前先 `backup` 一次当前状态。
- 备份文件不含 `snapshot` 之外的状态（collection 配置在快照内，payload index 会随 snapshot 恢复——实测删除后恢复，`user_id` 过滤检索仍正常）。

---

## 6. 本地多语言 Embedding 验证：`intfloat/multilingual-e5-base`

### 6.1 下载与缓存

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/agr-data/hf
# 23 个文件，snapshot_download 完成
# 缓存目录：/mnt/agr-data/hf/hub/models--intfloat--multilingual-e5-base
```

| 项 | 值 |
|---|---|
| Snapshot commit | `d128750597153bb5987e10b1c3493a34e5a4502a` |
| `model.safetensors` | 1 112 201 288 B，sha256 `a18a44fad1d0b46ded15928144138cff1135d5cc8233bdd90be5f18822de09a7` |
| `pytorch_model.bin` | 1 112 242 989 B（与 safetensors 并存） |
| `tokenizer.json` | 17 082 660 B，sha256 `62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626` |
| `config.json` | 694 B，sha256 `9dab198f24c8c0879e481cf7822005d5ecbceedbacb390ffafa594e28d31bac4` |
| snapshot 总占用（`du -shL`） | 5.0 G（含 safetensors + pytorch + onnx/openvino 变体） |
| 落盘位置 | NFS `/mnt/agr-data/hf`，**未写入 `/home`** |

> 下载使用 `HF_ENDPOINT=https://hf-mirror.com`；镜像站日志提示该账户近期出口流量较大，未来大模型下载可能被限速或限流。本次模型已完整缓存，Phase 6 只需复用缓存（`HF_HUB_OFFLINE=1`），不需要重复下载。

> 模型结构（从 `config.json` + safetensors header 实测）：`xlm-roberta`，12 层、hidden 768、12 头、vocab 250 002、`max_position_embeddings=514`，**278 044 162 参数**（1.11 GB FP32 safetensors 一致）。与项目商品侧 `intfloat/e5-base-v2` **向量空间不兼容**，不可混入同一 Collection（见 D9 决策草稿与 §9-R8）。

### 6.2 隔离运行环境（不污染共享 venv）

服务器事实：`python3 3.10.12`；`python3 -m venv` 因**缺少 ensurepip** 失败（与 `docs/audit/s4000-environment.md` 记录一致）；宿主机 user-site 已有 `torch 2.14.1+cpu`、`transformers 5.19.0`、`numpy 2.2.6`、`huggingface_hub 2.2.0`。

采用方案（等价于 `uv --target`）：

```bash
python3 -m venv --without-pip /tmp/agr_d9/venv          # 仅 /tmp，临时
python3 -m pip install --target /tmp/agr_d9/site --no-deps \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  sentence-transformers==6.3.0 scipy==1.15.3 scikit-learn==1.7.2 \
  pillow==11.3.0 joblib==1.5.2 threadpoolctl==3.6.0
# 运行时显式复用宿主 user-site（只读），不向共享 site-packages 写入任何文件：
PYTHONPATH=/tmp/agr_d9/site:/home/agr/.local/lib/python3.10/site-packages \
  python3 scripts/memory/bench_e5.py
```

版本矩阵（实测 import 成功）：

| 组件 | 版本 |
|---|---|
| sentence-transformers | 6.3.0 |
| transformers | 5.19.0 |
| torch | 2.14.1+cpu |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| scikit-learn | 1.7.2 |
| pillow | 11.3.0 |
| joblib | 1.5.2 |

> `/tmp/agr_d9/*` 仅作临时验证环境；**不**是项目依赖安装位置，也不作为产物唯一存储。生产依赖需在 Phase 6 按 lock 文件重新锁定（建议把上表写入 `pyproject.toml` 的 retrieval/memory extra）。

### 6.3 维度、归一化、确定性

```text
load_time_s=2.99 dim=768 rss_after_load_mb=847
unit_norm_query: true
unit_norm_passage: true
deterministic:   true（同输入两次编码逐元素差 < 1e-6）
dim_768:         true
E5 BENCH OK
```

- 768 维，与 `E5Encoder(expected_dim=768)` 一致；
- `normalize_embeddings=True` 后 L2 范数为 1.0（与项目 `E5Encoder.normalize=True` 相同）；
- 同一输入两次编码完全一致（CPU 确定性推理）。

### 6.4 中文/英文语义 sanity（余弦）

测试文本（Query 加 `query: `，Passage 加 `passage: `）：

- Q_zh：`推荐一款适合敏感肌的无香精保湿面霜`
- Q_en：`a fragrance-free moisturizer for sensitive skin`
- P_zh_match：`这款保湿面霜不含香精，配方温和，适合敏感肌肤使用。`
- P_en_match：`This fragrance-free moisturizer is gentle and suitable for sensitive skin.`
- P_zh_other：`机械键盘 87 键 青轴 有线 游戏电竞`
- P_en_other：`A 27-inch 4K monitor with USB-C hub and height adjustment`

| | P_zh_match | P_en_match | P_zh_other | P_en_other |
|---|---|---|---|---|
| Q_zh | **0.9222** | 0.8285 | 0.8264 | 0.7528 |
| Q_en | 0.8235 | **0.9086** | 0.7411 | 0.7467 |

- 中文 query 对中文匹配 passage 的相似度最高（0.9222 > 0.8264）；
- 跨语言场景：中文 query 与**英文**同义 passage（0.8285）高于无关英文（0.7528）；
- 英文 query 与英文匹配 passage 最高（0.9086）。

> 注意：E5 系列余弦基线整体偏高（不相关文本也可达 0.74–0.83），因此**不能**用固定 cosine 阈值判断相关性；Phase 6 需要 top-k + 相对排序，阈值需在验证集上校准（R12/Phase 7 评测）。

### 6.5 延迟与内存（CPU）

`scripts/memory/bench_e5.py`，模型已缓存（`HF_HUB_OFFLINE=1`），`max_seq_length=512`，单条中文 Query：

| 线程数 | 单条中文 query 中位 | p95 | 单条英文 query 中位 | 批 32 条中位 | 批吞吐 | RSS（加载后） | 峰值 RSS |
|---|---|---|---|---|---|---|---|
| 64（默认） | **16.23 ms** | 17.45 ms | 15.28 ms | 80.44 ms | **397.8 条/s** | 847 MB | 1245 MB |
| 4 | **33.26 ms** | 34.72 ms | 33.09 ms | 346.85 ms | 92.3 条/s | 848 MB | 1228 MB |

- 模型加载（缓存命中）2.99 s；首次 import 4.37 s；
- 4 线程是更现实的「与其它服务共享 CPU」配置；64 线程是宿主机默认（128 vCPU），延迟更低但会抢占资源；
- 内存预算：常驻约 **0.85 GB**，峰值约 **1.2 GB**（FP32、batch 32、512 token）。若 Memory 服务与 E5 商品编码器同进程，需按 2 份模型预算（或复用同一模型实例，但两模型不同 checkpoint，不能复用）。

### 6.6 与项目 `E5Encoder` 的兼容性（真实代码）

服务器仓库 `src/recommender/retrieval/e5/encoder.py` 与本地 md5 一致（`60894343d3a9c2c52a35c47584987047`，即当前 `main` 版本）；用 `PYTHONPATH=<repo>/src` 直接实例化项目编码器：

```python
E5Encoder(
    model_id="intfloat/multilingual-e5-base", device="cpu", expected_dim=768,
    query_prefix="query: ", passage_prefix="passage: ", batch_size=8,
)
```

```text
embedding_dim 768 device cpu
query_shape (2, 768) passage_shape (3, 768) dtype float32
query_norms [1.0, 1.0]
passage_norms [1.0, 1.0, 1.0]
similarity_matrix
[[0.9241 0.8431 0.8335]
 [0.8256 0.9127 0.7548]]
```

结论：**现有 `E5Encoder` 无需修改即可加载 `multilingual-e5-base`**（只换 `model_id` 与 `embedding_dim=768` 配置）；前缀、归一化、batch 逻辑均符合 E5 约定。

### 6.7 关于 `query:` / `passage:` 前缀

- 实测前缀被正确拼接（`E5Encoder._encode` 的 `prepared = prefix + text`），Mem0 侧必须复刻该行为（§7.3）；
- 前缀对**成对余弦**的影响：同一对中英匹配文本，加前缀 `0.9222` vs 不加 `0.9411`——说明「加前缀会提高成对余弦」**不成立**，前缀的作用是让模型按 E5 训练分布区分 query/passage 角色，收益应在**检索排序指标**上体现；
- **未验证**：前缀对 Recall@k/NDCG 的实际收益（需要 Phase 7 人工/合成 Query 集）。当前结论仅作为「遵守 checkpoint 约定」的依据，Q4 已对商品侧固定 `query:`/`passage:`。

---

## 7. Mem0 接口调研（Phase 6 Adapter 依据）

> 安装方式：`/tmp` 临时 target（`python3 -m pip install --target /tmp/agr_d9/mem0site ...`），**未接入项目代码、未写入共享 venv**。
> 版本：`mem0ai==2.2.1`、`qdrant-client==1.20.0`（PyPI 镜像 `mirrors.aliyun.com`）。

### 7.1 版本

| 包 | 版本 | 说明 |
|---|---|---|
| mem0ai | **2.2.1** | PyPI 当前最新（2026-11-24）；`Requires-Python: <4.0,>=3.10`；基础依赖含 `qdrant-client>=1.12.0`、`openai>=1.90.0`、`pydantic>=2.7.3`、`sqlalchemy>=2.0.31`、`httpx>=0.28.0`、`posthog`、`protobuf`、`pytz` |
| qdrant-client | **1.20.0** | 与 Qdrant server 1.20.2 兼容（客户端使用 `query_points`） |
| pydantic | 2.14.4 | 与仓库现有 pydantic 2.x 一致 |

> 依赖边界：`sentence-transformers`/`transformers` **不在** mem0ai 基础依赖中（在可选 extra `mem0ai[extras]` 中，≥5.2.0/≥5.3.0）；本项目走自定义 `EmbeddingBase` 方案，沿用项目 retrieval extra 已锁定的 ST 版本即可，无需 `mem0ai[extras]`。BM25 依赖 `fastembed`，同样不在 extras 中，需要时单独安装。

### 7.2 `Memory` 关键签名（2.2.1 实测）

```text
Memory.from_config(config_dict)                     # classmethod，MemoryConfig(**config_dict)
Memory.add(messages, *, user_id=None, agent_id=None, run_id=None, metadata=None,
           timestamp=None, expiration_date=None, infer=True, memory_type=None, prompt=None)
Memory.search(query, *, top_k=20, filters=None, threshold=0.1, rerank=False,
              explain=False, reference_date=None, show_expired=False, **kwargs)
Memory.get(memory_id)
Memory.get_all(*, filters=None, top_k=20, show_expired=False, **kwargs)
Memory.update(memory_id, text=None, metadata=None, expiration_date=..., data=None[已废弃])
Memory.delete(memory_id)
Memory.delete_all(user_id=None, agent_id=None, run_id=None)
Memory.history(memory_id)
Memory.reset()
```

**关键变化（相对旧版文档）**：`search()` **没有** `user_id` 位置参数，多用户过滤必须走 `filters={"user_id": "..."}`；`get_all()` 同样要求 `filters` 中至少包含 `user_id`/`agent_id`/`run_id` 之一。`update()` 的新内容参数是 `text`（`data` 仅为兼容别名）。

### 7.3 自定义 E5 Embedder 的接口

```text
EmbeddingBase(ABC):
    embed(text, memory_action: Optional[Literal["add","search","update"]]) -> list[float]
    embed_batch(texts, memory_action="add") -> list[list[float]]   # 默认逐条调用 embed
```

- `EmbedderFactory.provider_to_class` 内置 provider：`openai`、`ollama`、`huggingface`、`azure_openai`、`gemini`、`vertexai`、`together`、`lmstudio`、`langchain`、`aws_bedrock`、`fastembed`；
- `EmbedderConfig` 的 pydantic validator **只允许**上述 provider，直接写 `provider: "e5"` 会被拒绝；
- `HuggingFaceEmbedding.embed()` 直接 `self.model.encode(text)`，**不会**加 `query:`/`passage:` 前缀，也没有前缀配置项。

因此 Phase 6 的自定义 Embedder 建议（**顺序按推荐度**）：

1. **构造后注入**：`Memory.from_config(...)` 用 `embedder.provider="huggingface"`（提供合法配置与维度）→ 立刻把 `memory.embedding_model` 替换为 `E5MemoryEmbedder(EmbeddingBase)` 实例；所有 `add/search/update` 都经 `self.embedding_model.embed(text, memory_action)`，替换即生效（无需改 mem0 源码）。
2. **注册 provider**：在构造 `Memory` 前执行 `EmbedderFactory.provider_to_class["e5"] = "recommender.memory.embedder.E5MemoryEmbedder"`；但 `EmbedderConfig` 校验仍会拒绝 `provider="e5"`，需同时绕过 pydantic 校验（不推荐，升级易碎）。
3. **LangChain 包装**：`provider="langchain"` + 自定义 `langchain_core.embeddings.Embeddings`（`embed_query` 固定 query 前缀）；可行但引入额外依赖，且 mem0 的 `LangchainEmbedding.embed` 只调用 `embed_query`，add/search 都用 query 前缀，不符合 E5 双前缀约定（除非包装内部按状态区分，但 mem0 不传状态）。

`memory_action` 映射建议：`add`/`update` → `"passage: "`，`search` → `"query: "`（与 E5 训练一致；`update` 是否用 passage 前缀需在 Phase 6 用真实数据确认）。

### 7.4 Qdrant 向量存储（mem0 内置实现）

`mem0.vector_stores.qdrant.Qdrant`：

```text
__init__(collection_name, embedding_model_dims, client=None, host=None, port=None,
         path=None, url=None, api_key=None, https=None, on_disk=False)
insert(vectors, payloads=None, ids=None)
search(query, vectors, top_k=5, filters=None)         # 内部 client.query_points(...)
update(vector_id, vector=None, payload=None)
delete(vector_id)
get(vector_id)
list(filters=None, top_k=100)
delete_col() / reset() / col_info()
```

存储细节（源码 + 实测）：

- 创建 collection 时同时声明 **dense 向量**与名为 `bm25` 的 **sparse 向量槽**（`SparseVectorParams(modifier=IDF)`）；这是 mem0 2.2.1 的新 v3 混合检索结构；
- 自动为 `user_id`、`agent_id`、`run_id`、`actor_id` 建 keyword payload index（远程 Qdrant 才会建）；
- `Memory._create_memory()` 写入的 payload 至少包含：`data`、`hash`（md5）、`text_lemmatized`（BM25 用）、`created_at`、`updated_at`，以及实体字段 `user_id`/`agent_id`/`run_id`（`add()` 的 metadata 中身份字段不可被用户覆盖）、可选 `memory_type`/`expiration_date`/`actor_id`/`role`；
- `_create_filter()` 支持 `eq/ne/gt/gte/lt/lte/in/nin/contains/icontains` 与 `AND/OR/NOT` 逻辑；`user_id` 过滤最终生成 Qdrant `FieldCondition(key="user_id", match=...)`；
- BM25 需要 `fastembed` 额外依赖；**未安装时** bm25 槽仍会创建（新 collection），但 insert 不写 sparse 向量、混合检索不可用（语义检索不受影响）。是否安装 `fastembed` 属 Phase 6 决策（当前基线只用 dense 检索也能满足 Q7）。

### 7.5 真实存储层 round-trip

用 mem0 **自带** `Qdrant` 类连接本机 Qdrant 1.20.2，写入 6 条 768 维假向量（`u001`/`u002` 各 3 条，payload 模仿 mem0 schema），执行 insert / 过滤 search / get / list / update / delete / 清理。实际输出（`scripts/memory/probe_mem0_api.py`）：

```text
collection=agr_d9_mem0_probe dim=768 inserted=6
search(filters={"user_id":"u001"}) -> hits=3, isolation_ok=true, top_score=1.0
top_payload = {"data":"memory text 0","text_lemmatized":"memory text 0",
               "hash":"a859da37be0e59d3b68cda2371b1f0e1","user_id":"u001",
               "agent_id":null,"run_id":null,
               "created_at":"2026-11-24T08:00:00+00:00","updated_at":"2026-11-24T08:00:00+00:00"}
get_ok=true  list(filters={"user_id":"u002"})=3
update_ok=true（payload-only set_payload 后立即可见）  delete_ok=true（u002 剩 2 条）
collection_sparse_vectors=["bm25"]
payload_indexes=["actor_id","agent_id","run_id","user_id"]
checks: collection_has_bm25_slot=true, user_id_indexed=true,
        search_isolation=true, update_visible=true, delete_removed=true
ok=true
```

> 这段验证的是 **storage + filter 契约**（collection 结构、payload index、`user_id` 过滤、更新可见性、删除生效），不是记忆抽取质量。测试 collection 已在脚本结束时删除。

### 7.6 Phase 6 Adapter 参数清单

| 配置项 | 建议值/来源 | 说明 |
|---|---|---|
| `vector_store.provider` | `"qdrant"` | mem0 内置 |
| `vector_store.config.collection_name` | `"agr_memories"`（待定，单 Collection） | Q7：单 Collection + `user_id` 过滤 |
| `vector_store.config.embedding_model_dims` | **`768`（必须显式设置）** | QdrantConfig 默认 1536；不设置会导致 768 维写入失败 |
| `vector_store.config.host` / `port` | `127.0.0.1` / `6333` | 也支持 `url`/`api_key`（当前未启用鉴权） |
| `vector_store.config.on_disk` | `false` | 向量规模小；活动卷已在本地盘 |
| `embedder.provider` | `"huggingface"`（占位）+ 构造后替换为 `E5MemoryEmbedder` | §7.3 方案 1；模型 `intfloat/multilingual-e5-base`、revision `d1287505…`、CPU、`max_length=512`、`normalize=True` |
| `embedder.config.embedding_dims` | `768` | 与 `QdrantConfig.embedding_model_dims` 一致 |
| `llm.provider` / `config` | `"deepseek"` + 现有 DeepSeek 配置（API key 仅从环境变量） | mem0 2.2.1 `LlmConfig` 支持 `deepseek`；`add()` 需要 LLM 做事实抽取 |
| `history_db_path` | `artifacts/mem0/history.db`（本地盘路径，**非 NFS**） | mem0 内部 SQLite 历史库；与项目 `InteractionEvent` SQLite 是两套表，不得混用 |
| 身份隔离 | 每次 `add()` 传 `user_id`；每次 `search()/get_all()` 传 `filters={"user_id": ...}` | 无 `user_id` 的 `get_all` 会报错；过滤遗漏=跨用户泄漏（R11） |
| `top_k` | 与 `memory.retrieval_top_k` 配置对齐（待定，默认 20） | `search` 默认 20、`threshold` 默认 0.1（E5 余弦基线高，阈值必须校准，不能沿用 0.1） |
| 超时/重试 | 传入自建 `QdrantClient(timeout=...)` 或 mem0 的 `client` 字段；重试包在项目 Adapter 层 | mem0 不暴露完整的 Qdrant HTTP 超时/重试配置 |
| 版本锁定 | `mem0ai==2.2.1`、`qdrant-client==1.20.0`、Qdrant server `v1.20.2`（镜像 digest 见 §3.1）、`sentence-transformers==6.3.0`、模型 commit `d1287505…`、维度 `768` | Q7「版本和向量维度锁定在配置中」 |

其他注意事项：

- `Memory.__init__` 会立即创建 `SQLiteManager(history_db_path)`，目录需可写；该库**不是** `InteractionEvent` 存储，TIGER 序列仍从项目 SQLite 读取，两者不可混用（R12）；
- `Memory.__init__` 的顺序是 embedder → vector store → llm → SQLite；若用「构造后替换 embedder」方案，collection 的维度由 `embedding_model_dims` 决定，与 embedder 实例无关，因此必须显式 768；
- 若开启 `MEM0_TELEMETRY`（默认开启），会外发匿名统计；离线环境应设置 `MEM0_TELEMETRY=False`（建议写入 Adapter 初始化）；
- `add(infer=True)` 需要 LLM；离线/无 key 环境可用 `infer=False` 直接写入文本（Phase 6 仅在受控测试中使用）。

### 7.7 实测输出（probe_mem0_api.py）

版本：`{"mem0ai":"2.2.1","qdrant-client":"1.20.0","pydantic":"2.14.4","openai":"3.58.0","transformers":"5.19.0","torch":"2.14.1+cpu"}`

```text
Memory.from_config(config_dict: Dict[str, Any])
Memory.add(self, messages, *, user_id=None, agent_id=None, run_id=None, metadata=None,
           timestamp=None, expiration_date=None, infer=True, memory_type=None, prompt=None)
Memory.search(self, query: str, *, top_k: int = 20, filters: Optional[Dict] = None,
              threshold: float = 0.1, rerank: bool = False, explain: bool = False,
              reference_date=None, show_expired: bool = False, **kwargs)
Memory.get(self, memory_id)
Memory.get_all(self, *, filters=None, top_k: int = 20, show_expired: bool = False, **kwargs)
Memory.update(self, memory_id, text: Optional[str] = None, metadata=None,
              expiration_date=<unset>, data: Optional[str] = None /*deprecated*/)
Memory.delete(self, memory_id)
Memory.delete_all(self, user_id=None, agent_id=None, run_id=None)
Memory.history(self, memory_id) / Memory.reset(self)
EmbeddingBase.embed(self, text, memory_action: Optional[Literal['add','search','update']])
EmbeddingBase.embed_batch(self, texts, memory_action='add')
Qdrant.__init__(self, collection_name: str, embedding_model_dims: int,
                client: QdrantClient = None, host=None, port=None, path=None,
                url=None, api_key=None, https: bool|None = None, on_disk: bool = False)
Qdrant.insert(self, vectors: list, payloads: list = None, ids: list = None)
Qdrant.search(self, query: str, vectors: list, top_k: int = 5, filters: dict = None) -> list
Qdrant.update(self, vector_id: int, vector: list = None, payload: dict = None)
Qdrant.delete(self, vector_id: int) / get(vector_id) / list(filters=None, top_k=100)
QdrantConfig 字段: collection_name, embedding_model_dims, client, host, port,
                 path, url, api_key, https, on_disk
内置 embedder provider: aws_bedrock, azure_openai, fastembed, gemini, huggingface,
                        langchain, lmstudio, ollama, openai, together, vertexai
```

存储层 live 结果见 §7.5（全部 5 项 checks 为 true）。

---

## 8. 对 Q7 的符合性

| Q7 要求 | 本次结论 | 证据 |
|---|---|---|
| Mem0 OSS | 版本锁定 `mem0ai==2.2.1`，接口已核对 | §7.1–7.2 |
| 独立 Qdrant 服务 | 独立容器 `agr-qdrant`，与应用生命周期分离，`restart unless-stopped` | §3.2 |
| 持久化数据卷 | named volume `agr-qdrant-data` 落本地盘 `/var/lib/docker/volumes/...`；重启恢复实测 24 条不丢 | §1.2、§4.2 |
| 单 Collection + `user_id` 过滤 | 存储层支持并实测隔离；Adapter 必须每次带 `filters={"user_id":...}` | §4.3、§7.6 |
| 本地 Embedding | `intfloat/multilingual-e5-base` 本地 CPU 推理，无外部 Embedding API | §6 |
| 版本 + 向量维度锁定配置 | 版本矩阵、模型 commit、768 维、镜像 digest 均已记录；建议写入 `pyproject.toml`/配置 | §3.1、§6.1、§7.6 |
| 数据边界（Mem0 语义 vs SQLite 事件） | 文档保持不变；mem0 自带 SQLite 仅 history，不与 `InteractionEvent` 混用 | §7.6 |
| NFS 用于备份/快照 | 快照备份到 `/mnt/agr-data/qdrant_backup` 并完成恢复演练；活动目录不在 NFS | §5 |
| 不暴露公网 | 仅 `127.0.0.1:6333/6334` | §3.2 |
| 决策状态 | D9 在 `Phase4-7_D1-D11_DECISION_DRAFT_V1.0.md` 仍为「待确认」；本报告提供实测证据，需负责人批准后更新决策记录 | — |

**结论：Q7 的服务形态要求在存储/服务/Embedding 层已满足；剩余风险集中在 Phase 6 Adapter 的真实 `add()` 链路与 `user_id` 过滤的代码级保证（本报告只验证了存储层）。**

---

## 9. 未验证项与风险

| # | 风险 / 未验证项 | 级别 | 影响 | 建议 |
|---|---|---|---|---|
| R1 | `Memory.add()` 端到端未执行（需 DeepSeek API key、LLM 事实抽取、embedder 真实注入） | 高 | 无法证明「反馈→记忆」真实链路 | Phase 6 用测试 user_id + 真实 DeepSeek key 做一次受控 add/search/update/delete；记录 token/延迟 |
| R2 | 自定义 E5 Embedder 注入 `Memory` 未实测（只验证了 `EmbeddingBase` 签名与存储层） | 高 | 前缀/维度错误会导致检索质量或写入失败 | Adapter 落地后跑一次「写入带 `passage:`、检索带 `query:`」的集成测试，校验向量维度 768 |
| R3 | BM25 混合检索未启用（fastembed 未安装；bm25 槽存在但为空） | 中 | 关键词召回缺失，仅 dense 检索 | Phase 6 决策：安装 `mem0ai[extras]`/fastembed 或明确 dense-only 基线并记录 |
| R4 | 前缀对检索排序的实际收益未测（只测了成对余弦） | 中 | 无法量化 `query:`/`passage:` 的收益 | Phase 7 用标注 Query 集评测（R@k/NDCG） |
| R5 | Docker Hub 不可达，镜像来源为第三方 mirror | 中 | 镜像可复现性依赖 mirror | 记录 digest（已记录）；如需长期可复现，考虑 `p09-registry` 本地存档镜像 |
| R6 | 备份未自动化（无 cron/计划任务） | 中 | 数据丢失窗口取决于人工备份 | Phase 6 结束前配置调度并演练一次定时备份；`KEEP=14` |
| R7 | Qdrant 无 API key/TLS，仅回环绑定 | 中 | 同机其他用户可访问 6333，可读所有用户记忆 | 生产化前增加 API key（mem0 `url`+`api_key` 支持）或 Unix socket/防火墙；风险登记 |
| R8 | `multilingual-e5-base` 与商品侧 `e5-base-v2` 同为 768 维但向量空间不同 | 高 | 误混会导致检索错乱 | 记忆用独立 Collection；禁止跨模型共用 Collection/索引；写入 manifest/配置 |
| R9 | E5 余弦基线高（无关文本 0.74–0.83），`threshold=0.1` 无意义 | 中 | 低相关记忆被召回 | Phase 6/7 在验证集校准 `threshold` 与 `top_k` |
| R10 | `multilingual-e5-base` 中文效果未经标注集评测（仅 sanity） | 中 | 中文记忆质量未知 | Phase 7 用小规模人工标注对评测；必要时对比 e5-base-v2/其他模型 |
| R11 | 内存占用 ~0.85–1.2 GB/进程 | 低 | 与其它模型同进程时挤占内存（宿主 1 TB，可接受） | 单进程单实例、惰性加载；压测并发 |
| R12 | Qdrant named volume 位于 `/var/lib/docker`（95% 使用率） | 低 | 长期增长可能触发磁盘告警 | 监控卷大小；NFS 快照作为恢复手段；需要时迁移到更大本地盘 |

---

## 10. 运维速查

```bash
# 部署/启动（幂等；端口被他人占用时拒绝执行）
bash scripts/memory/deploy_qdrant.sh

# 状态（只读）
bash scripts/memory/deploy_qdrant.sh status

# 冒烟测试（15 项，含过滤/删除/重建 collection）
python3 scripts/memory/qdrant_smoke_test.py

# 备份全部 collection 到 NFS
bash scripts/memory/backup_qdrant.sh backup
# 只备份一个 collection
bash scripts/memory/backup_qdrant.sh backup --collection agr_memories
# 查看备份清单 / collection 清单
bash scripts/memory/backup_qdrant.sh list
bash scripts/memory/backup_qdrant.sh collections
# 从快照恢复（覆盖同名 collection，维护窗口执行）
bash scripts/memory/backup_qdrant.sh restore agr_memories \
  /mnt/agr-data/qdrant_backup/agr_memories/<snapshot>

# Embedding 基准（/tmp 隔离环境）
HF_HOME=/mnt/agr-data/hf HF_HUB_OFFLINE=1 \
PYTHONPATH=/tmp/agr_d9/site:/home/agr/.local/lib/python3.10/site-packages \
  python3 scripts/memory/bench_e5.py --threads 4 --json /tmp/e5_bench.json

# mem0 接口探测（/tmp 隔离环境）
PYTHONPATH=/tmp/agr_d9/mem0site python3 scripts/memory/probe_mem0_api.py \
  --url http://127.0.0.1:6333 --json /tmp/mem0_probe.json
```

故障排查：

- 容器不在：`docker logs --tail 100 agr-qdrant`；确认 6333 未被他人占用（脚本会拒绝启动而不是抢占）。
- 端口冲突：修改 `QDRANT_HTTP_PORT`/`QDRANT_GRPC_PORT` 或停掉冲突方（需负责人确认，不得 kill 他人进程）。
- 恢复失败：确认快照文件与 manifest 的 sha256 一致；恢复会覆盖同名 collection。
- 磁盘：`docker system df -v` 查看 `agr-qdrant-data` 卷大小；`/var/lib/docker` 剩余 84 G。

---

## 11. Git 边界

- 本次只新增 `docs/phase6/D9_MEMORY_DEPLOYMENT.md`、`scripts/memory/*`（5 个脚本），未改 `src/`、`configs/`、`cli.py`；
- 模型权重、HF cache、Qdrant 卷、快照文件均在 NFS/本地盘，**不进入 Git**；
- Qdrant 容器与卷属于服务器运行时状态，**不进入 Git**；镜像 digest、版本、卷名、端口记录在本报告，便于重建；
- 本报告不含任何 API key、密码或用户真实数据（测试 payload 为匿名假数据）。

---

## 12. 附录：证据命令与输出摘要

### A. 容器与镜像

```text
docker inspect agr-qdrant:
  id=73ab20882183f480d0564089ebf6e04fc40bfe4aafdddfc56db5a0e6b17550df
  image=sha256:8870e3837712f612f997b683dee186a4d10fa48f1535377d007c2dd3429e74ef
  restart=unless-stopped runtime=runc
  mounts=[{Name:agr-qdrant-data, Source:/var/lib/docker/volumes/agr-qdrant-data/_data, Destination:/qdrant/storage}]
  ports={"6333/tcp":[{HostIp:127.0.0.1, HostPort:6333}], "6334/tcp":[{HostIp:127.0.0.1, HostPort:6334}]}
docker image inspect qdrant/qdrant:v1.20.2:
  sha256:8870e3837712…  docker.m.daocloud.io/qdrant/qdrant@sha256:d277592721d2…
```

### B. Embedding 基准原始 JSON（关键字段）

```json
{
  "model": "intfloat/multilingual-e5-base",
  "dim": 768,
  "load_time_s": 2.99,
  "rss_after_load_mb": 847.1,
  "cosine": {"zh_query~zh_match": 0.9222, "zh_query~en_match": 0.8285,
             "zh_query~zh_other": 0.8264, "zh_query~en_other": 0.7528,
             "en_query~zh_match": 0.8235, "en_query~en_match": 0.9086},
  "single_query_latency": {"median_ms": 16.23, "p95_ms": 17.45},
  "batch_latency": {"median_ms": 80.44, "texts_per_s": 397.8},
  "peak_rss_mb": 1244.9,
  "checks": {"dim_768": true, "unit_norm_query": true, "unit_norm_passage": true,
             "deterministic": true, "zh_match_ranked_above_other": true,
             "zh_query_close_to_en_passage": true},
  "ok": true
}
```

### C. 执行时间线

| 时间（CST） | 动作 |
|---|---|
| 15:54 | 前置检查（docker/端口/磁盘/NFS），确认 Docker Hub/GitHub/PyPI 官方源不可达 |
| 15:56 | 经 `docker.m.daocloud.io` 拉到 qdrant v1.20.2（197 MB）并打本地 tag |
| 15:58 | `deploy_qdrant.sh` 创建卷 + 容器，`/healthz` 通过（当时空闲资源约 533–544 MiB） |
| 16:00 | 冒烟测试 15/15；写入 24 条测试数据；`docker restart` 恢复验证；NFS 快照备份（408 064 B）与删除后恢复验证 |
| 15:57–16:01 | 下载 `multilingual-e5-base`（23 文件，`hf-mirror.com`，5.0 G 含全部格式） |
| 16:02–16:10 | `/tmp` 隔离环境（venv `--without-pip` + `--target site`）装 ST 6.3.0/scipy/sklearn；bench（64 线程与 4 线程两轮） |
| 16:10–16:13 | 项目真实 `E5Encoder` 兼容测试（768 维、前缀、归一化） |
| 16:07–16:22 | mem0ai 2.2.1 + qdrant-client 1.20.0 安装（目标 `/tmp/agr_d9/mem0site`）与接口探测（源码签名 + 真实 Qdrant round-trip） |
| 16:20–16:25 | 清理测试 collection、脚本复核、撰写本报告 |

---

*本报告由 D9 执行过程直接生成；所有数字来自实测命令输出，未使用估算值。*
