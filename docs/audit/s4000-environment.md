# S4000 环境侦察报告（Phase 0/1 前置）

> 实测时间：2026-11-24 01:09–01:23 CST
> 目标机：`192.0.2.10`（hostname `agr-worker`），用户 `agr`（uid=1003，gid=1003，附组 `render(109)`、`docker(120)`）
> 执行方式：本机（Windows git-bash）→ `python /c/new/intern/plan/_work/s4000_ssh.py "<cmd>"`（paramiko，密码登录，本机无 sshpass）
> 约束：未修改项目源码；未在宿主机/容器系统环境 pip install（临时 venv 均在 `/tmp` 且已清理）；未删除服务器已有文件；未动队友进程/GPU 任务。
> 清理状态：本轮临时目录（容器内 `/tmp/probe-venv*`、`/tmp/hfprobe`，宿主机 `/tmp/torchdl*`、`/tmp/agr-venv`，NFS `/mnt/agr-data/agr_probe`）均已删除；`/home` 可用空间恢复为 3.7G。

---

## 1. 结论摘要

| 维度 | 结论 | 能否满足 Phase 0 / Phase 1 |
|---|---|---|
| 存储 | `/home` 仅剩 **3.7G**（191G，98% 已用），不能放大文件；`/mnt/agr-data` 是 **13T 的 NFS（192.0.2.11），剩 12T，777 可写，实测 1GB `dd` 370 MB/s**；`/data` 虽 13T 但 **root-only（agr 不可写，sudo 密码也不可用）**；`/tmp` 在 `/` 上剩 122G（可写但易失）；`/var/lib/docker` 仅剩 84G（95% 已用） | ✅ Phase 0 代码放 `/home`，数据/缓存/产物放 `/mnt/agr-data/agr/...`；⚠️ 持久化依赖 NFS，需接受其风险 |
| 网络 | 直连 `huggingface.co` **不通（HTTP 000，7.7s 超时）**；`hf-mirror.com` **可用（200，1.8s）**；`pypi.org` 可用但极慢（15s 打到 max-time），**清华镜像 200 / 0.73s**；`api.deepseek.com` 带 key **200（0.22s）**；GitHub 慢但可用（首页 11.2s，`git ls-remote` 2.2s） | ✅ Phase 0/1 依赖可装、模型/数据可下（必须走 hf-mirror + 清华源） |
| 依赖 | 宿主机 `python3` **3.10.12**，但 **`python3-venv`/ensurepip 未安装**（`python3 -m venv` 直接失败）；宿主机已有 user-site `torch 2.14.1+cpu`、`transformers 5.19.0`、`numpy 2.2.6`、`pydantic 2.14.0`、`datasets 2.19.0`；`uv 0.12.21` 可用（可建 3.12 venv）。容器 `dp`：Python **3.10.12**、`torch 2.9.0` + `torch_musa 2.9.0`、`transformers 5.8.1`、`numpy 1.25.0`、`pydantic 2.13.4`；**无 faiss / sentence-transformers / langgraph / mem0 / qdrant-client / datasets** | ✅ Phase 0 在容器内用「`--without-pip --system-site-packages` venv + 系统 pip」方案可装（实测 14s 装完 faiss+ST+langgraph）；⚠️ 必须锁 `numpy==1.25.0` + `faiss-cpu==1.9.0`，否则 faiss/ torch numpy 桥 ABI 冲突 |
| GPU | `mthreads-gmi 2.3.2`、Driver `3.3.5-server`，8×MTT S4000 48GB，全部 0% 利用率；**GPU3/4/5 完全空闲**，GPU0/1/2/6/7 上有 root 的队友进程 PID 1909527（GPU7 占 1401MiB）；`torch.musa.is_available()==True`，8 卡；新建容器 `MTHREADS_VISIBLE_DEVICES=3` 可单卡绑定并跑通 matmul | ✅ Phase 1 E5 编码可上 GPU（实测 E5-small-v2 短文本 ≈ 1700–3000 条/秒，冷启动后）；注意 MUSA 非 CUDA、无 FlashAttention |
| 数据/模型 | 服务器上**没有 Amazon Reviews 2023**、没有 E5/Qwen3-Reranker 权重；HF 缓存仅 5.1M（Qwen3-0.6B 指针）；`p09/models` 有队友的 Qwen3-0.6B(1.5G)/Qwen3-8B(16G)/Qwen3.5-9B(19G)。All_Beauty meta **212,990,142 B**、reviews **326,611,506 B**，hf-mirror 50MB 实测 **3.1–3.6 MB/s** | ✅ 可下载（全量约 540MB ≈ 2–3 分钟）；模型需另下：E5-large-v2 ≈1.3GB、Qwen3-Reranker-0.6B ≈1.2GB |

**一句话**：Phase 0/1 可以在 S4000 上真实跑通——代码放 `/home/agr`（小），大文件全部放 `/mnt/agr-data/agr/...`（12T 可写），在 `dp` 同镜像的**专用容器**里用 `--without-pip --system-site-packages` 建 venv 装依赖（Python 3.10 + torch_musa 2.9.0），数据/模型走 `hf-mirror.com`、pip 走清华源；最大风险是 **home 只剩 3.7G + NFS 是共享/无备份的临时性存储 + faiss/numpy ABI 必须锁版本**。

---

## 2. 逐项证据

### 2.1 主机与存储

**命令**

```bash
hostname; whoami; date; uname -a
df -h
id
mount | grep -E ' /data | /mnt/agr-data | /home '
du -sh /home/agr/* 2>/dev/null | sort -rh | head
test -w /data; echo $?     # 1 = 不可写
sudo -n true; echo $?      # 1 = 无免密 sudo
echo '<password>' | sudo -S -k id   # rc=1，密码亦不可用
```

**关键原始输出**

```text
agr-worker
agr
Tue Nov 24 01:09:49 AM CST 2026
Linux agr-worker 5.15.0-105-generic #115-Ubuntu SMP ... x86_64 GNU/Linux

Filesystem                             Size  Used Avail Use% Mounted on
tmpfs                                  101G  359M  101G   1% /run
/dev/mapper/ubuntu--vg-ubuntu--lv      246G  112G  122G  48% /
tmpfs                                  504G   12M  504G   1% /dev/shm
/dev/mapper/ubuntu--vg-home--lv        191G  178G  3.7G  98% /home
/dev/mapper/data--vg-docker--data--lv  1.5T  1.4T   84G  95% /var/lib/docker
/dev/mapper/data--vg-data--lv           13T   28K   12T   1% /data
192.0.2.11:/data/agr-storage          13T  129G   12T   2% /mnt/agr-data

uid=1003(agr) gid=1003(agr) groups=1003(agr),109(render),120(docker)

/dev/mapper/ubuntu--vg-home--lv on /home type ext4 (rw,relatime,stripe=64)
/dev/mapper/data--vg-data--lv on /data type ext4 (rw,relatime,stripe=512)
192.0.2.11:/data/agr-storage on /mnt/agr-data type nfs
  (rw,relatime,sync,vers=3,rsize=1048576,wsize=1048576,namlen=255,hard,proto=tcp,timeo=600,retrans=2,sec=sys,...)

# /home 占用大头
89G  /home/agr/7606C
61G  /home/agr/p09
8.7G /home/agr/musa-toolkit
3.0G /home/agr/plugcharger_train.h5
2.8G /home/agr/7606C-main

# /data: root:root drwxr-xr-x，仅 lost+found；agr test -w /data → 不可写
# sudo：sudo -n true → 需要密码；用给定密码 sudo -S -k id → rc=1（无 root 权限）
```

**NFS 写入/文件系统实测**

```bash
mkdir -p /mnt/agr-data/agr_probe
dd if=/dev/zero of=/mnt/agr-data/agr_probe/probe1g.bin bs=1M count=1024 oflag=direct
for i in $(seq 1 200); do echo $i > f$i.txt; done   # 计时
ln -s f1.txt link1; mv f1.txt f1renamed.txt
```

```text
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 2.90304 s, 370 MB/s
200 个小文件创建：0.081 s（real）
1MB (256×4KB) dd：23.7 MB/s
symlink_ok / rename_ok
```

**结论**：`/mnt/agr-data` = 13T/12T 可写、实测顺序写 370 MB/s、小文件/符号链接/rename 正常，是当前**唯一可用的大容量共享存储**；`/data` 13T 本地盘因无 root 不可用；`/home` 只剩 3.7G；`/tmp` 在 `/` 上（122G）可作临时区。

---

### 2.2 容器与挂载关系

**命令**

```bash
docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
docker info | grep -E 'Server Version|Storage Driver|Docker Root|Runtimes|Default Runtime'
docker inspect --format '{{range .Mounts}}{{.Type}} {{.Source}} -> {{.Destination}} (rw={{.RW}}){{end}}' dp p09-train vllm21
docker inspect --format 'runtime={{.HostConfig.Runtime}} privileged={{.HostConfig.Privileged}} ipc={{.HostConfig.IpcMode}} shm={{.HostConfig.ShmSize}} net={{.HostConfig.NetworkMode}} binds={{json .HostConfig.Binds}}' dp
docker inspect --format 'entrypoint={{.Config.Entrypoint}} env={{range .Config.Env}}{{.}} {{end}}' <image>
```

**关键原始输出**

```text
Server Version: 24.0.7
Storage Driver: overlay2
Docker Root Dir: /var/lib/docker
Containers: 19    Images: 1248
Runtimes: mthreads mthreads-experimental runc io.containerd.runc.v2
Default Runtime: mthreads

dp          Up 7 weeks   registry.mthreads.com/presale/devtech/vllm-mt:d0930
p09-train   Up 6 weeks   registry.mthreads.com/mcctest/training-suite:v2.1.7.rc3-ut-verify
p09-train2  Up 6 weeks   registry.mthreads.com/presale/devtech/vllm-mt:d0930
vllm21      Up 5 weeks   registry.mthreads.com/presale/devtech/vllm_musa:s4000_4.3.5_d0708
agx-rl      Up 4 days    registry.mthreads.com/presale/devtech/vllm-mt:d0930
p09-registry Up 6 weeks  registry:2
12 个 swebench 沙箱容器 Up 4 weeks（coding agent 项目遗留）
wizardly_brown  Exited (0) 7 weeks ago

# 挂载（宿主机路径 = 容器路径）
dp:         bind /home/agr/7606C -> /workspace/code (rw)
p09-train:  bind /home/agr/p09   -> /workspace/p09 (rw)
vllm21:     bind /home/agr/p09   -> /workspace/p09 (rw)
agx-rl:     bind /home/agr/p09   -> /workspace/p09 (rw)

# dp 关键 HostConfig
runtime=mthreads privileged=false ipc=private shm=34359738368 net=host
binds=["/home/agr/7606C:/workspace/code"]

# 镜像（vllm-mt:d0930）环境变量关键项
MTHREADS_VISIBLE_DEVICES=all
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
MUSA_HOME=/usr/local/musa
LD_LIBRARY_PATH=/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/musa/lib:/usr/local/openmpi/lib
entrypoint=[true]   # 即默认 ENTRYPOINT 是 true，自定义命令需 --entrypoint /bin/bash
```

**容器内存储**

```text
容器 dp: df -h / → overlay 1.5T 1.4T 84G 95%；/workspace/code 在 /home（3.7G 可用）
容器 dp: /dev/shm = 32GiB（--shm-size 32g）
```

**新建容器 + NFS 挂载 + 单卡绑定实测**（临时容器，已 `--rm` 清理）

```bash
docker run --rm --network host --shm-size 32g --entrypoint /bin/bash \
  -v /mnt/agr-data/agr_probe:/probe \
  -e MTHREADS_VISIBLE_DEVICES=3 \
  registry.mthreads.com/presale/devtech/vllm-mt:d0930 -c '...'
```

```text
/probe 显示 192.0.2.11:/data/agr-storage 13T 129G 12T 2%
id=uid=0(root)
DEVCOUNT 1  AVAIL True  MATMUL 780.59
root 写 NFS 文件 → 属主 root:root（演示了铁律：容器 root 写 NFS 会留下 root 文件）

# 以宿主 uid/gid 运行（避免属主问题）
--user 1003:109 -e USER=agr -e LOGNAME=agr
→ uid=1003 gid=109；DEVCOUNT 1 AVAIL True MATMUL -454.39；写入文件 ls -ln = 1003 109
# 注意：不加 USER/LOGNAME 时 torch 导入会因容器 /etc/passwd 无 1003 而 KeyError: 'getpwuid(): uid not found: 1003'
```

---

### 2.3 Python 环境与依赖

#### 2.3.1 宿主机

```bash
python3 -V; ls -la /usr/bin/python3*; python3 -m pip -V
python3 -m ensurepip --version
python3 -m venv /tmp/agr-venv
/home/agr/.local/bin/uv --version
/home/agr/.local/bin/uv venv /tmp/agr-venv
/home/agr/.local/bin/uv pip install --python /tmp/agr-venv/bin/python \
  -i https://pypi.tuna.tsinghua.edu.cn/simple pydantic pyyaml httpx
python3 -m pip list | grep -Ei 'torch|transformers|faiss|numpy|pydantic|langgraph|datasets|huggingface|mem0|qdrant'
```

```text
Python 3.10.12
/usr/bin/python3 -> python3.10
/usr/bin/python3.10
pip 26.2.1 from /home/agr/.local/lib/python3.10/site-packages/pip (python 3.10)

python3 -m ensurepip → No module named ensurepip
python3 -m venv /tmp/agr-venv → 失败：ensurepip is not available（python3.10-venv 未安装）

uv 0.12.21
uv venv → Using CPython 3.12.14（uv 管理的 3.12，不是系统 3.10）
uv pip install pydantic pyyaml httpx（清华源）→ 1s：pydantic 2.14.4, pyyaml 6.0.4, httpx 0.28.1

宿主机 user-site 已装（只读盘点）：
torch 2.14.1+cpu（cuda False）, transformers 5.19.0, numpy 2.2.6, pydantic 2.14.0,
datasets 2.19.0, huggingface_hub 2.2.0, openai 3.26.1, requests 2.34.2
无 faiss / sentence-transformers / langgraph / mem0 / qdrant-client
注：宿主机 PATH 里没有 `pip3`，用 `python3 -m pip`（26.2.1）即可。
```

#### 2.3.2 训练容器 `dp`

```bash
docker exec dp python -V
docker exec dp pip list | grep -Ei '^(torch|torch_musa|transformers|sentence-transformers|faiss|numpy|pydantic) '
docker exec dp python -c 'import importlib.util as u; [print(m, bool(u.find_spec(m))) for m in ["faiss","sentence_transformers","langgraph","mem0","qdrant_client","datasets"]]'
```

```text
Python 3.10.12
torch 2.9.0
torch_musa 2.9.0
transformers 5.8.1
numpy 1.25.0
pydantic 2.13.4
huggingface_hub 1.33.0
pandas 2.3.3, scipy 1.15.3, sentencepiece 0.2.1, vllm 0.24.1.dev54+gc18c12ea6.d20260814.musa

faiss False
sentence_transformers False
langgraph False
mem0 False
qdrant_client False
datasets False
```

其他容器：

```text
p09-train: Python 3.10.12; torch 2.7.1.post1, torch_musa 2.7.1.post1, transformers 4.57.0,
           datasets 3.0.0, huggingface_hub 0.36.2, numpy 1.26.4, pydantic 2.13.4
vllm21:    torch 2.9.0, torch_musa 2.9.0, transformers 5.8.1, numpy 1.25.0, huggingface_hub 1.14.0
```

#### 2.3.3 容器内临时 venv + pip 安装实测（`/tmp`，未动系统 pip）

```bash
# python -m venv 在容器内同样因 ensurepip 缺失而失败；容器内已有 /usr/local/bin/uv、virtualenv 21.3.3
docker exec dp python -m venv --without-pip --system-site-packages /tmp/probe-venv
docker exec dp /tmp/probe-venv/bin/python -m pip --version     # 借用系统 pip 22.0.2
docker exec dp /tmp/probe-venv/bin/python -m pip install --no-cache-dir \
  faiss-cpu sentence-transformers langgraph -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```text
venv 创建成功（--without-pip --system-site-packages）
pip 22.0.2 from /usr/lib/python3/dist-packages/pip
install_elapsed_sec=14
Successfully installed faiss-cpu-1.16.0 sentence-transformers-6.3.0 langgraph-1.2.24
  langchain-core-1.7.7 scikit-learn-1.7.2 ...（其余为纯 Python 依赖）
```

**faiss × numpy ABI 关键实测（重要）**

| 组合 | 结果 |
|---|---|
| `faiss-cpu 1.16.0` + 系统 `numpy 1.25.0` | ❌ `ModuleNotFoundError: No module named 'numpy._core'`（faiss 1.16 的 loader 需要 numpy 2.x） |
| `faiss-cpu 1.16.0` + `numpy 2.2.6` | ⚠️ faiss 检索 OK，`torch.musa` matmul 也 OK，但 torch 报 `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6... Failed to initialize NumPy: _ARRAY_API not found`；且 pip 冲突：`vllm requires numpy==1.25`、`tilelang-musa requires numpy<2` |
| `faiss-cpu 1.9.0` + `numpy 1.25.0` | ✅ faiss 检索 OK（`IndexFlatIP` search 正常）+ `torch 2.9.0` MUSA matmul OK，无 numpy 警告 |

**结论**：Phase 0 容器环境应 **锁 `numpy==1.25.0`（与 torch_musa/vllm 一致）+ `faiss-cpu==1.9.0`**；不要用 faiss-cpu ≥1.10 的默认最新版，也不要在同一环境把 numpy 升到 2.x。

---

### 2.4 网络连通性（宿主机与容器 `dp` 结果一致）

**命令**

```bash
for u in https://huggingface.co https://hf-mirror.com https://pypi.org/simple/ \
         https://pypi.tuna.tsinghua.edu.cn/simple/ https://api.deepseek.com/v1/models https://github.com; do
  curl -sS -o /dev/null --max-time 15 -w 'HTTP=%{http_code} time=%{time_total}s\n' "$u"
done
```

**关键原始输出（宿主机）**

```text
https://huggingface.co                        HTTP=000 time=7.714241s     # 直连不可用
https://hf-mirror.com                         HTTP=200 time=1.836749s     # ✅ 可用
https://pypi.org/simple/                      HTTP=200 time=15.000078s    # 打到 max-time，极慢
https://pypi.tuna.tsinghua.edu.cn/simple/     HTTP=200 time=0.732664s     # ✅ 快
https://api.deepseek.com/v1/models            HTTP=401 time=0.168100s     # 无 key，符合预期
https://github.com                            HTTP=200 time=11.228214s    # 慢但可达
```

容器 `dp` 内同一组测试：`huggingface.co` 000/7.70s、`hf-mirror.com` 200/1.19s、`pypi.org` 200/15.0s、`tuna` 200/0.74s、`deepseek` 401/0.34s —— 与宿主机一致（容器 `net=host`）。

补充：

```text
git ls-remote --heads https://github.com/psf/requests → 2.2s，可正常 clone/ls-remote
ghproxy.net / gh-proxy.com / ghfast.top → 200 / 1.2s、2.3s、1.4s
download.pytorch.org/whl/cpu → 200 / 1.7s
```

#### 2.4.1 实际下载测速

**hf-mirror（HF 文件）**

```text
# All_Beauty 商品元数据（真实大小见 dataset API）
GET https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_All_Beauty.jsonl
curl -r 0-52428799（50MB）
→ try1 HTTP=206 size=52,428,800B time=14.51s speed=3,613,156 B/s（3.4 MiB/s）
→ try2 HTTP=206 size=52,428,800B time=17.02s speed=3,080,413 B/s（2.9 MiB/s）

# 文件真实大小（HF dataset API tree）
raw/meta_categories/meta_All_Beauty.jsonl        212,990,142 B  (约 203 MiB)
raw/review_categories/All_Beauty.jsonl           326,611,506 B  (约 311 MiB)
→ 全量 540 MB 估算 ≈ 2.5–3 分钟（3 MB/s 量级）
```

**huggingface_hub 走镜像（容器内，`HF_ENDPOINT=https://hf-mirror.com`）**

```text
Qwen/Qwen3-Reranker-0.6B tokenizer.json → 11,422,654 B，7.4s（约 1.5 MB/s）
intfloat/e5-small-v2 全量下载 → /tmp/hfprobe 缓存 140M；首次 SentenceTransformer 加载 67.0s（含下载），
  期间出现一次 HTTP 429 + 5.0s 重试；缓存后再次加载 4.9s
警告文本："You have used a large amount of egress over the last 30 days... downloads will be throttled or blocked"
```

**pip（清华源）**

```text
pip index versions torch → 2.14.1 可用（历史版本 1.11.0–2.14.1 全列出）
pip download --no-cache-dir torch==2.9.0 → torch-2.9.0-cp310-cp310-manylinux_2_28_x86_64.whl (899.8 MB)
    899.8/899.8 MB  119.5 MB/s  0:00:09，elapsed=12s，Saved 859M
uv pip install pydantic+pyyaml+httpx（清华源）→ 1s 完成
```

#### 2.4.2 DeepSeek API（带 key 只输出状态码，未回显 key）

```bash
K=<REDACTED>; curl -s -o /dev/null --max-time 15 -H "Authorization: Bearer $K" \
  -w 'HTTP=%{http_code} time=%{time_total}s\n' https://api.deepseek.com/v1/models
curl ... -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
  https://api.deepseek.com/v1/chat/completions
```

```text
GET /v1/models        → HTTP=200 time=0.224749s
POST /v1/chat/completions → HTTP=200 time=0.409732s
模型列表：['deepseek-flash', 'deepseek-v4-pro']
```

---

### 2.5 GPU / MUSA

**命令**

```bash
mthreads-gmi
docker exec dp python -c 'import torch; print(torch.__version__, torch.musa.is_available(), torch.musa.device_count())'
```

**关键原始输出**

```text
mthreads-gmi: 2.3.2     Driver Version: 3.3.5-server
ID  Name        PCIe              %GPU  Mem
0   MTT S4000   00000000:08:00.0  0%    0MiB(49152MiB)        55C
1   MTT S4000   00000000:09:00.0  0%    0MiB(49152MiB)        52C
2   MTT S4000   00000000:0e:00.0  0%    0MiB(49152MiB)        54C
3   MTT S4000   00000000:11:00.0  0%    0MiB(49152MiB)        52C   ← 空闲
4   MTT S4000   00000000:32:00.0  0%    0MiB(49152MiB)        50C   ← 空闲
5   MTT S4000   00000000:38:00.0  0%    0MiB(49152MiB)        51C   ← 空闲
6   MTT S4000   00000000:3b:00.0  0%    0MiB(49152MiB)        54C
7   MTT S4000   00000000:3c:00.0  0%    1401MiB(49152MiB)     55C

Processes:
0 1909527 python <1MiB
1 1909527 python <1MiB
2 1909527 python <1MiB
6 1909527 python <1MiB
7 1909527 python 1380MiB

ps -o pid,user,etime,cmd -p 1909527
→ root  24-14:48:39  python -u retrieval/server.py     # 队友进程，勿动

dp: torch 2.9.0 musa True devs 8
```

**MUSA 计算实测**

```text
容器 dp（系统 python）：
  torch 2.9.0, torch.musa.is_available()=True, device_count=8
  torch.randn(128,128,device="musa") @ 自身 → 成功
临时 venv（numpy 1.25 + faiss 1.9）：musa_matmul_ok
新建容器 + MTHREADS_VISIBLE_DEVICES=3：DEVCOUNT=1, AVAIL=True, matmul OK（单卡绑定有效）
新建容器 + MTHREADS_VISIBLE_DEVICES=4 + --user 1003:109 + USER/LOGNAME：DEVCOUNT=1, matmul OK
```

**E5 推理实测（sentence-transformers 6.3.0，MUSA:3，E5-small-v2）**

```text
load_sec 4.9（本地缓存后）
batch=32 n=256 sec=0.15 emb_per_sec=1742.0 shape=(256, 384)
batch=64 n=256 sec=0.08 emb_per_sec=3071.7 shape=(256, 384)
# 警告（不影响结果）：SDPA 提示 query/key/value 是 float 而非 Half/BFloat16
```

---

### 2.6 已有数据 / 模型缓存

```bash
du -sh /home/agr/.cache/huggingface; ls /home/agr/.cache/huggingface/hub
du -sh /home/agr/p09/models/*
find /home/agr -maxdepth 4 \( -iname '*amazon*' -o -iname '*review*' -o -iname '*Electronics*' \)
```

```text
/home/agr/.cache/huggingface → 5.1M
  datasets--princeton-nlp--SWE-bench                 52K
  models--LogicStar--SWE-Bench-Verified-Compressed   12K
  models--Qwen--Qwen3-0.6B                           12K（仅指针，无权重）
/home/agr/.cache/modelscope → 8.0K（空）

/home/agr/p09/models:
  Qwen3-0.6B   1.5G
  Qwen3-8B     16G
  Qwen3.5-9B   19G          # 均为队友 p09 项目，只读引用有风险

Amazon Reviews 2023：未找到（/home 与 /mnt/agr-data 均无）
E5 权重：未找到
Qwen3-Reranker-0.6B：未找到
```

**可下载数据地址（已在 hf-mirror 验证 HTTP 200/206，无需 token）**

```text
https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_All_Beauty.jsonl
  → 212,990,142 B
https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/All_Beauty.jsonl
  → 326,611,506 B
```

---

## 3. 未验证项（unverified）

1. **NFS 的持久化/备份策略**：`/mnt/agr-data` 来自 `192.0.2.11:/data/agr-storage`，777 共享目录，`sync` + `hard` 挂载；是否为组内正式存储、是否有备份/配额/清理策略未知。断网时 `hard` 挂载可能造成 I/O 卡住。
2. **All_Beauty 全量下载的完整性**：仅实测前 50MB range 与单文件 tokenizer/e5-small 全量；未做全量 540MB 校验（sha/size 对比）。
3. **Qwen3-Reranker-0.6B 在 MUSA 上的推理**：仅验证了 config/tokenizer 可下载，未做 transformers 5.8.1 + torch_musa 2.9.0 的实际推理（输入长度/显存/速度未知）。
4. **E5-large-v2 在 MUSA 的吞吐/显存**：只测了 e5-small-v2（384 维）、256 条短文本；E5-large（1024 维）与真实商品文本长度下的吞吐、batch 上限未测。
5. **TIGER / RQ-VAE 权重来源**：未检查 HF 上是否有与当前数据兼容的 TIGER 实现/权重（项目 AGENTS.md 第 8 节明确列为未验证事项）。
6. **宿主机 CPU-only 路径的速度**：宿主机已有 `torch 2.14.1+cpu` + `transformers 5.19.0`，但未安装 sentence-transformers，也未 benchmark CPU E5 编码（预计远慢于 GPU，不作为 Phase 1 主路径）。
7. **新建容器镜像**：只用本地已有 `vllm-mt:d0930` 验证；未测试 `registry.mthreads.com` 是否可直接 pull 其他镜像、也未测试新建容器长时间运行稳定性。
8. **`/home` 配额**：ext4 未见配额，写满会静默失败（历史坑），未验证是否有隐藏 quota。
9. **GPU 队友进程增长**：仅采到 01:09–01:23 快照；PID 1909527 的 `retrieval/server.py` 未来是否会占用更多显存/GPU 未知。
10. **NFS 上跑 venv 的稳定性**：只测了 200 个小文件创建；未把整个 venv（上千文件）放 NFS 长跑验证。

---

## 4. 风险与建议

### 4.1 存储落位建议（Phase 0）

```text
/home/agr/Agentic-Generative-Recommendation/     ← 代码 + configs + docs（几百 MB，够）
/home/agr/.cache/…                              ← 不放（只剩 3.7G）
/mnt/agr-data/                          ← 数据 / HF 缓存 / embeddings / faiss / runs（12T）
  ├── data/raw/amazon/                              ← All_Beauty jsonl
  ├── hf/                                           ← HF_HOME（E5、Qwen3-Reranker）
  ├── artifacts/embeddings|faiss|semantic_ids/
  └── runs/
/tmp/                                               ← 只放一次性中间文件（122G，易失）
/data                                               ← ❌ 无 root，不能用；不要尝试 sudo
/var/lib/docker                                     ← ❌ 95% 已用，不要用 docker volume 存大文件
```

- NFS 写入属主：容器内以 root 跑会写出 root 文件到 NFS；**统一用 `--user 1003:109 -e USER=agr -e LOGNAME=agr`**（实测可跑 GPU 且文件属主 1003:109）。
- 需要保留的关键产物（FAISS 索引、embeddings）建议在 `/home` 留 manifest/hash，必要时复制一份到 `/tmp` 或压缩备份；NFS 无备份承诺。

### 4.2 Python/依赖落位建议（Phase 0）

宿主机：

- `python3-venv` 缺失 → **不要**尝试 `python3 -m venv`；用 `uv 0.12.21`（`/home/agr/.local/bin/uv`）建 venv（注意它默认拉 3.12）。
- 宿主机已有 `torch 2.14.1+cpu`、`transformers 5.19.0`、`datasets 2.19.0`，可做**纯 CPU 的下载/预处理脚本**；但不要与 GPU 容器环境混用。

容器（推荐主路径）：

- 用与 `dp` 同镜像新建专用容器（不碰 dp 内已有 venv/代码）：`vllm-mt:d0930`，默认 runtime 已是 `mthreads`。
- venv：`python -m venv --without-pip --system-site-packages /path/venv` + `/path/venv/bin/python -m pip`（系统 pip 22.0.2），**不要** `pip install` 到容器系统环境（铁律：torch_musa/驱动绑定）。
- **版本锁定**：`numpy==1.25.0`（与 torch_musa 2.9.0/vllm 一致）+ `faiss-cpu==1.9.0` + `sentence-transformers==6.3.0` + `langgraph==1.2.24` + `pydantic==2.13.4`；其余（mem0ai 2.2.1、qdrant-client 1.20.0、datasets 等）按需装，全部用 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。
- 推荐容器启动模板（已在侦察中验证 GPU + NFS 可用）：

```bash
docker run -d --name agr \
  --network host --shm-size 32g \
  --entrypoint /bin/bash \
  --user 1003:109 -e USER=agr -e LOGNAME=agr -e HOME=/tmp \
  -e MTHREADS_VISIBLE_DEVICES=3 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e HF_HOME=/mnt/agr-data/hf \
  -e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -v /home/agr/Agentic-Generative-Recommendation:/workspace/agr \
  -v /mnt/agr-data:/mnt/agr \
  registry.mthreads.com/presale/devtech/vllm-mt:d0930
# 一 run 一卡：MTHREADS_VISIBLE_DEVICES=3/4/5 任选其一；0/1/2/6/7 有队友 root 进程（PID 1909527）
```

### 4.3 网络/下载建议（Phase 1）

- **HF 直连不可用**，所有 HF 下载必须 `HF_ENDPOINT=https://hf-mirror.com`；`huggingface_hub` 已实测可用（1.33.0 in dp）。
- hf-mirror 会 429/限速（本次 e5-small 下载时出现一次 429 + 5s 重试，并提示近期出口流量大）：下载脚本需 **重试/退避**，大文件（E5-large 1.3GB ≈ 7 分钟；Qwen3-Reranker 0.6B ≈ 1.2GB）尽量一次成功、保留缓存；可评估 modelscope 作为备用源（本机 modelscope 缓存为空，未实测）。
- pip 一律清华源（镜像 env 已默认 `PIP_INDEX_URL`，实测 900MB torch wheel 119.5 MB/s）；不要用 pypi.org（15s 级延迟）。
- DeepSeek 网络可达，模型 ID 实测为 `deepseek-flash` / `deepseek-v4-pro`（项目 AGENTS.md §8 的「DeepSeek 具体模型 ID」可由本次实测确认；API 参数与 JSON 输出校验仍未验证；key 未落盘、未回显）。
- GitHub 可用但慢（首页 11s）；代码上传优先本机 paramiko/SFTP 或 `git ls-remote` 确认后再 clone，大仓库避免直连 GitHub。

### 4.4 风险清单（按优先级）

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| 1 | `/home` 仅 3.7G | 写入即满、静默失败，可能拖垮队友 p09/7606C | 代码/日志之外一律写 NFS；脚本里 `df` 预检；不要往 home 放 HF cache/venv 大文件 |
| 2 | NFS 是共享 777、无备份、`hard` 挂载 | 数据丢失/断网 I/O 卡死 | 关键产物校验 hash + 副本；避免把解释器 venv 放 NFS（放容器 `/tmp` 或 home 小 venv）；监控 `df` |
| 3 | faiss × numpy ABI | `faiss-cpu>=1.10` 需 numpy2，会破坏 torch_musa/vllm | 锁 `numpy==1.25.0` + `faiss-cpu==1.9.0`；或单独 CPU 环境建索引 |
| 4 | 容器系统 pip 不能动 | 升级/替换包可能破坏 torch_musa/驱动 | 只用 `--without-pip --system-site-packages` venv 或 `--target`；不改系统 site-packages |
| 5 | GPU 与队友共享 | 抢卡/显存冲突（GPU7 已 1.4GB） | 只用 GPU3/4/5，`MTHREADS_VISIBLE_DEVICES` 单卡绑定，一 run 一卡 |
| 6 | hf-mirror 限速/429 | 模型下载中断 | 重试退避、断点续传（range 206 已支持）、缓存目录持久化到 NFS |
| 7 | Docker 根目录 95% | 新镜像/容器写满会失败 | 不 build 新镜像、不写大文件到 overlay；用 NFS 挂载 |
| 8 | 无 root | 不能用 `/data` 13T、不能 apt 装 python3-venv | 用 uv/容器 venv 绕开；如需 `/data` 找管理员开目录 |
| 9 | MUSA 生态限制 | 无 FlashAttention、DDP 不可用、MCCL 慢 | E5/reranker 推理走单卡；batch 小步调优；TIGER 训练遵循已有 FSDP 经验 |
| 10 | transformers 5.8.1 较新 | sentence-transformers 6.3 兼容性 | 已实测 E5-small 加载/编码 OK；E5-large、Qwen3-Reranker 需 Phase 1 早期冒烟 |

---

## 5. 对 Phase 0 / Phase 1 的影响（结论）

**Phase 0（工程骨架 + 依赖安装）——可行，按以下方式落地：**

1. 代码目录 `/home/agr/Agentic-Generative-Recommendation`（小）；数据/缓存/产物目录 `/mnt/agr-data/{data,hf,artifacts,runs}`。
2. 专用容器（`vllm-mt:d0930`，`--network host --shm-size 32g --user 1003:109 -e USER/LOGNAME -e MTHREADS_VISIBLE_DEVICES=3`，挂 home 代码 + NFS 数据），**不改造 dp/p09 现有容器**。
3. venv：`python -m venv --without-pip --system-site-packages` + 系统 pip 22.0.2；依赖锁 `numpy==1.25.0`、`faiss-cpu==1.9.0`、`sentence-transformers==6.3.0`、`langgraph==1.2.24`；实测装包 14s（清华源）。
4. 宿主机仅用于轻量脚本/下载（已有 CPU torch 2.14.1 + datasets 2.19.0，venv 用 uv）；**不要**在宿主机装 GPU torch（MUSA 只在容器内）。

**Phase 1（Amazon 商品目录 + E5 编码 + FAISS 索引）——可行：**

1. All_Beauty：meta 203MiB + reviews 311MiB，从 `hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023` 下载，实测 3 MB/s → 全量约 2.5–3 分钟；`HF_ENDPOINT` + 重试即可。
2. E5：`intfloat/e5-small-v2` 已在 MUSA 单卡跑通（256 短文本 batch64 0.08s；384 维）；Phase 1 正式用 E5-large-v2（1.3GB，1024 维）需在 Phase 1 开始先做一次 load+encode 冒烟与显存/吞吐测量。
3. FAISS：CPU `faiss-cpu==1.9.0` 可用；12T NFS 足够放 embeddings（1M×1024×4B ≈ 4GB）与索引；索引构建优先在容器内 CPU/GPU 侧完成，产物写 NFS。
4. 模型权重：E5-large-v2 + Qwen3-Reranker-0.6B 服务器上都没有，需从 hf-mirror 下载（各 ≈7 分钟/1.2GB，限速风险见上）；TIGER RQ-VAE 权重来源仍待 Phase 1 核实。

**当前不建议做的事：** 使用 `/data`（无权限）、在容器系统环境 pip、在 `/home` 放 HF 缓存或大数据、动 GPU0/1/2/6/7、在 p09/7606C 目录里写本项目的文件、用 pypi.org 直连装包。
