# Qwen3-Reranker-0.6B CPU 冒烟测试（R4 / R5）

> 实测时间：2026-11-24 10:42–11:00 CST
> 目标机：`192.0.2.10`（hostname `agr-worker`，Intel Xeon Gold 6430，2×32C/128T，128 逻辑核）
> 执行方式：本机（Windows git-bash）→ `python /c/new/intern/plan/_work/s4000_ssh.py "<cmd>"`（paramiko）
> 运行环境：`/tmp/agr/venv`（Python 3.10.12，torch 2.6.0+cpu，transformers 4.57.6）；先 `source scripts/env_s4000.sh`
> 约束遵守：未修改项目源码与配置；测试脚本全部放服务器 `/tmp`；模型与 HF 缓存写入 `HF_HOME=/mnt/agr-data/hf`（NFS），未写 `/home`；未触碰队友容器/GPU 进程。
> 数据性质：本文 10 组 query-document 对为**手写冒烟测试文本**（非 Amazon 数据、非标注评测集），不得进入任何 Recall/NDCG 实验。

---

## 1. 结论摘要

### R4（Transformers 原生打分实现是否正确）→ 通过（有边界）

| 检查项 | 结果 | 证据 |
|---|---|---|
| 模型能否用 `AutoModelForCausalLM` + fp32 + CPU 跑通 | ✅ 跑通，595,776,512 参数，`Qwen3ForCausalLM` | 第 4、5 节 |
| Yes/No token id 是否确认 | ✅ `yes=9693`、`no=2152`，均为**单 token**；与仓库自带 `1_LogitScore/config.json` 的 `true_token_id=9693 / false_token_id=2152` 完全一致 | 4.3 |
| 官方模板是否确认 | ✅ 三方一致：模型卡 `suffix`、`tokenizer.chat_template`（`enable_thinking=false` 的 assistant 段）、实测 token id 序列 `[151645,198,151644,77091,198,151667,271,151668,271]` | 4.2、4.4 |
| 分数能否用于排序 | ✅ 10 对手写文本，`min(相关)=0.999923 > max(不相关)=0.013828`，100% 区分；两套 instruction 均成立 | 6 |
| 能否复现模型卡自带样例 | ✅ "Beijing" `margin=+7.597` vs 卡片公布 `7.625`（差 0.03）；"Gravity" `margin=-12.192` vs 公布 `-11.375`（差 0.82，fp32 vs 卡片 bf16 推理所致） | 7.1 |
| 分数是否可当概率/阈值用 | ❌ **不可**。相关对全部饱和在 0.9999，不相关对低至 3e-5；域内难负例（beard oil vs 护发油）仍有 0.0138。0.5 阈值无意义，**只可用于同 query 内相对排序** | 6、8 |

**R4 结论**：Transformers 原生打分链路可用，模板与 token 已确认（非猜测），排序语义正确。风险不在"能不能用"，而在**分数饱和导致的可比性问题**：跨 query 的绝对分数不可直接比较，阈值/校准必须用真实标注数据重新确定（当前未做）。

### R5（CPU 延迟/内存是否满足）→ 已给出实测数字，判定留给预算方

不做事后主观"可接受"判断，只列可对比的事实（OMP=32，单条即 batch=1，两轮独立复测）：

| 设置 | 单条 P50（轮1 / 轮2） | batch=4 单条摊薄 | 峰值 RSS（轮1 / 轮2） |
|---|---|---|---|
| ~512 token | **0.414 s / 0.396 s** | 0.288 / 0.315 s | 3.92 / 3.91 GB |
| ~1024 token | 1.908 / 1.658 s | 0.639 / 0.597 s | 3.96 / 3.93 GB |
| ~2048 token | **3.720 s / 3.649 s** | 2.084 / 2.028 s | 5.21 / 5.11 GB |

可对比事实（非评价）：
- 单条 486→2026 token（4.17×）耗时 0.40→3.65 s（**9.1×**，超线性，与注意力 O(n²) 及下述 1k token 异常一致）。
- batch=4 相对 batch=1 的单条提速：512 token 1.26×，1024 token 2.78×，2048 token 1.80×。
- 20 候选短名单（512 token、batch=4）≈ **5×1.15–1.26 s ≈ 6 s/query**；50 候选 ≈ 15 s/query（同条件外推，未实测）。
- 内存：fp32 权重 **2.38 GB**（磁盘 bf16 1.19 GB），batch=1 峰值 3.9–5.2 GB，**batch=4×2048 峰值 11.4 GB**。
- 加载：页缓存热时 0.34–0.71 s；全新进程（含 import + 加载 + 两次前向）墙钟 7.84 s。NFS 冷缓存加载**未测**。
- 测试时宿主 load average 18.4–25.0（128 逻辑核共享机，含他人负载），数字含竞争噪声。

---

## 2. 环境

```text
host agr-worker, 2026-11-24T10:52:44+08:00
CPU: Intel(R) Xeon(R) Gold 6430 @ 3.4GHz(max), 2 socket × 32 core × 2 HT = 128 logical / 64 physical
可用性实测: Cpus_allowed_list=0-127, os.sched_getaffinity=128 核, 无 /sys/fs/cgroup/cpu.max（无 CPU 配额）
NUMA: node0=0-31,64-95  node1=32-63,96-127（Mems_allowed_list=0-1）
Mem: 1007 GB total, 909 GB available
OMP_NUM_THREADS=32（本轮设定值，非机器上限）, torch.get_num_threads()=32, os.cpu_count()=128
venv: /tmp/agr/venv  Python 3.10.12  torch 2.6.0+cpu  transformers 4.57.6
HF_ENDPOINT=https://hf-mirror.com   HF_HOME=/mnt/agr-data/hf (NFS, 12T 可用)
```

> 勘误（避免后人误读）：本轮日志里 `nproc` 曾打印 32，那是 **GNU `nproc` 会读 `OMP_NUM_THREADS` 环境变量**的结果（`nproc`=128，`OMP_NUM_THREADS=32 nproc`=32，实测），**不是 cgroup/affinity 限额**。机器本身 128 逻辑核、无配额；32 线程是任务指定的用量选择。因此"用满 64/128 线程能否更快"是一个**未测**的优化项（见 §10 第 5 条）。

必做前置（缺一则 HF_ENDPOINT/HF_HOME/venv 全部丢失）：

```bash
cd $HOME/Agentic-Generative-Recommendation && source scripts/env_s4000.sh
```

---

## 3. 模型获取

### 3.1 版本与体积

```text
repo      : Qwen/Qwen3-Reranker-0.6B
revision  : e61197ed45024b0ed8a2d74b80b4d909f1255473
snapshot  : /mnt/agr-data/hf/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473
```

| 文件 | 字节 |
|---|---:|
| `model.safetensors` | 1,191,588,280 |
| `tokenizer.json` | 11,422,654 |
| `vocab.json` | 2,776,833 |
| `merges.txt` | 1,671,853 |
| `tokenizer_config.json` | 9,706 |
| `config.json` | 727 |
| `sentence_bert_config.json` | 362 |
| `config_sentence_transformers.json` | 325 |
| `modules.json` | 280 |
| `generation_config.json` | 214 |
| `1_LogitScore/config.json` | 57 |
| **合计** | **1,207,471,291**（1.12 GiB） |

说明：下载用 `allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.py","merges.txt","vocab*"]` 过滤了 `*.md`/`*.png`，故上表不含 README（`README.md` 另从 hf-mirror 单独取回，14,742 B，用于核对模板）。

**重要发现**：仓库自带 SentenceTransformers CrossEncoder 元数据（`modules.json` → `Transformer` + `1_LogitScore`；`config_sentence_transformers.json` → `"model_type": "CrossEncoder"`、`"default_prompt_name": "query"`）。即该模型**也能**用 `sentence_transformers.CrossEncoder` 一行加载（模型卡首例即如此）。本测试按决策 Q2 走 Transformers 原生路径，未使用 CrossEncoder；同时该文件反过来独立印证了 yes/no token id（见 4.3）。

### 3.2 下载故障与处理（原始证据）

下载 1.19 GB 共用约 9 分钟，其中 2 次失败，**未编造任何数据**：

1. `10:42:24` 首次启动 `snapshot_download`（4 workers）。10 个小文件 20 s 内完成；`model.safetensors` 到 `629,150,287 B`（约 600 MiB，总量一半）后**停滞**：
   `stat` 连续 45 s 尺寸与 mtime 不变（`2026-11-24 10:45:20`），`ss -tnp` 显示到 CDN（`65.9.130.49:443`）连接仍 `ESTAB` 但收发队列均为 0 → 判为镜像侧半开连接。同刻 `curl` 单线程实测镜像仍可达（206，5 MB/14.8 s ≈ 0.34 MB/s），故非整体宕机。
2. kill 后改用带 `HF_HUB_DOWNLOAD_TIMEOUT=30` 的重试包装脚本，**但脚本里写的是 `python`** 而非绝对路径 → 8 次 `rc=127`（`python: command not found`，非交互 shell 未激活 venv），**该轮未写入任何数据**。
3. `10:48:53` 修正为 `/tmp/agr/venv/bin/python` 重启。**未从 600 MiB 处续传**：观察到的 `.incomplete` 只有 16,684,966 B（推测镜像侧 etag/分片变化导致失效，未深究），即实际重新拉了整包。**129.9 s 完成 1,207,471,291 B（1.19 GB），约 9.2 MB/s**，`rc=0`（`DONE_OK`，日志时间戳 `10:51:03`）。

教训（供后续模型下载复用）：重试包装脚本必须用 venv 绝对路径；镜像会在 ~600 MiB 处半开卡死，需外部 watchdog（超时/kill/重启），不能只靠 hf 默认重试。

### 3.3 加载期报错（已记录，已绕开）

`AutoModelForCausalLM.from_pretrained(..., device_map="cpu")` 直接失败——venv 内**未安装 `accelerate`**：

```text
ValueError: Using a `device_map`, `tp_plan`, `torch.device` context manager or setting
`torch.set_default_device(device)` requires `accelerate`. You can install it with
`pip install accelerate`
```

绕开方式：去掉 `device_map`，加载后显式 `.to(torch.device("cpu"))`。在 CPU-only 的 torch 构建下二者等价（无 CUDA 可用设备），且这也正是模型卡示例的写法。未安装 `accelerate` 本身不影响结果。

---

## 4. 模型架构与模板（原始读取结果）

### 4.1 `config.json`

```text
architectures            = ["Qwen3ForCausalLM"]
model_type               = "qwen3"
num_hidden_layers        = 28
hidden_size              = 1024
intermediate_size        = 3072
num_attention_heads      = 16
num_key_value_heads      = 8        # GQA，head_dim=128
num_parameters           = 595,776,512（实测 sum(p.numel())）
tie_word_embeddings      = true     # 词嵌入与 LM head 共享，约 151669×1024 ≈ 155 M 参数
vocab_size               = 151669
max_position_embeddings  = 40960
rope_theta               = 1000000
sliding_window           = null     # 未启用
torch_dtype（磁盘）       = "bfloat16"；本次加载为 torch.float32
model_max_length（tokenizer）= 131072
```

三处长度限制必须区分：`config.max_position_embeddings=40960`（架构上限）、`tokenizer.model_max_length=131072`（词表配置）、**模型卡运营上限 `max_length=8192`**（本测试按 8192 截断）。

### 4.2 chat template 与官方模板的一致性

`tokenizer_config.json` **含完整 Qwen3 chat_template**（`chat_template is None: False`）。其 `add_generation_prompt` 分支内的 `enable_thinking is false` 子句（节选，原模板缩进略有出入）：

```jinja
{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
{%- endif %}
```

即推理端"非思考" assistant 段恰为 `'<|im_start|>assistant\n<think>\n\n</think>\n\n'`，与模型卡 rerank 模板的 `suffix` 逐字符相同。这是模板正确性的**第二个独立来源**（不靠猜）。

`apply_chat_template` 实测（未传 `enable_thinking`）：

```text
'<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n'
```

### 4.3 Yes/No token id（两处独立来源一致）

```text
tokenizer.convert_tokens_to_ids("yes") = 9693    encode("yes") = [9693]
tokenizer.convert_tokens_to_ids("no")  = 2152    encode("no")  = [2152]
tokenizer.convert_tokens_to_ids(" yes")/(" No")/("yes.")/("no.") = None   # 均非单 token，不可用
```

| 来源 | yes | no | 独立性 |
|---|---|---|---|
| tokenizer 实测 `convert_tokens_to_ids` | 9693 | 2152 | 本机计算 |
| 仓库自带 `1_LogitScore/config.json` | `true_token_id: 9693` | `false_token_id: 2152` | **独立来源**（模型作者发布时写下的配置） |
| 模型卡示例 `token_true_id/token_false_id` | 同方法 | 同方法 | 复现官方写法（非独立） |

其他特殊 token：`eos=<|im_end|>`(151645)、`pad=<|endoftext|>`(151643)、`<|im_start|>`=151644、**无 BOS**（`add_bos_token=False`）；`<think>`=151667、`</think>`=151668 均为**单 token**。
注：`<｜end▁of▁thinking｜>` **不在词表**（`convert_tokens_to_ids` 返回 `None`），若误用会被 byte-level 拆成普通字符（见 7.2 消融）。

### 4.4 实际送进模型的 token 序列

```text
prefix_len = 39   tokens: "<|im_start|>system\nJudge whether the Document meets the requirements
                            based on the Query and the Instruct provided. Note that the answer
                            can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
body       = "<Instruct>: {task}\n<Query>: {query}\n<Document>: {doc}"
suffix_len = 9    suffix_ids = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]
                  = "<|im_end|>\n" + "<|im_start|>" + "assistant" + "\n"
                    + "<think>" + "\n\n" + "</think>" + "\n\n"
```

`tokenizer("x")["input_ids"] = [87]` → 首 token 无 BOS，与模型卡 `add_special_tokens` 缺省行为一致。

---

## 5. 打分实现（与模型卡逐行对齐）

```python
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
          'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
          "<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MAX_LEN = 8192

tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    path, dtype=torch.float32, low_cpu_mem_usage=True).to(torch.device("cpu")).eval()
token_false_id = tokenizer.convert_tokens_to_ids("no")    # 2152
token_true_id  = tokenizer.convert_tokens_to_ids("yes")   # 9693
prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)

def process_inputs(pairs):                                # pairs = 已拼好的 body 字符串
    inputs = tokenizer(pairs, padding=False, truncation="longest_first",
                       return_attention_mask=False,
                       max_length=MAX_LEN - len(prefix_tokens) - len(suffix_tokens))
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    return tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=MAX_LEN)

def compute_logits(inputs):
    last = model(**inputs).logits[:, -1, :]               # [B, 151669]
    stacked = torch.stack([last[:, token_false_id], last[:, token_true_id]], dim=1)  # [B,2] = [no, yes]
    return torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp()               # P(yes)
```

等价于 `score = sigmoid(logit_yes − logit_no) = exp(ly)/(exp(ly)+exp(ln))`。
实测 `logits.shape = [10, 151669]`（截断到最后一个位置的完整词表分布）。fp32、显式 `model.to("cpu")`、`OMP_NUM_THREADS=32`、`torch.get_num_threads()==32` 均已落实。

---

## 6. 10 组测试对结果（冒烟测试文本，非评测集）

`p_yes` = 上述打分；两套 instruction 各跑一遍（`model_card_default` 为模型卡默认，`shopping_custom` 为电商改写）。

| # | 类别 | query | 文档（截断） | tokens | margin† | **p_yes（默认指令）** | **p_yes（电商指令）** |
|---|---|---|---|---|---:|---:|---:|
| 1 | 相关 | moisturizing face cream for dry sensitive skin | CeraVe Moisturizing Cream…ceramides + HA | 122 | +9.879 | 0.999949 | 0.999871 |
| 2 | 相关 | sulfate free shampoo for color treated hair | Olaplex No.4 Bond Maintenance Shampoo… | 109 | +9.647 | 0.999935 | 0.999732 |
| 3 | 相关 | vegan matte lipstick in deep red | Charlotte Tilbury Matte Revolution… | 106 | +10.280 | 0.999966 | 0.999818 |
| 4 | 相关 | electric facial cleansing brush for acne prone skin | FOREO LUNA 3 silicone device… | 109 | +9.761 | 0.999942 | 0.999681 |
| 5 | 相关 | broad spectrum spf 50 sunscreen spray | Neutrogena Ultra Sheer Spray SPF 50… | 118 | +9.475 | 0.999923 | 0.999829 |
| 6 | 不相关（跨域） | hair dryer with diffuser for curly hair | Maybelline Lash Sensational Mascara… | 106 | −9.272 | 0.000094 | 0.000266 |
| 7 | 不相关（跨域） | perfume with vanilla and sandalwood notes | Sony WH-1000XM5 耳机… | 108 | −10.288 | 0.000034 | 0.000046 |
| 8 | 不相关（跨域） | fast drying nail polish | Dyson V8 无线吸尘器… | 97 | −10.243 | 0.000036 | 0.000037 |
| 9 | 不相关（域内难） | vitamin c serum for dark spots | Beauty of Joseon 防晒霜 SPF50+… | 105 | −7.955 | 0.000351 | 0.000773 |
| 10 | 不相关（域内难） | unscented beard oil for men | Moroccanoil 阿甘油护发精华… | 102 | −4.267 | **0.013828** | 0.002837 |

† `margin = logit_yes − logit_no`，取自**默认指令**那次前向（logits 形状 `[10, 151669]`）。

汇总：

| 指标 | 默认指令 | 电商指令 |
|---|---:|---:|
| min(相关) | 0.999923 | 0.999681 |
| max(不相关) | 0.013828 | 0.002837 |
| 均值差（相关−不相关） | 0.997074 | 0.998994 |
| 间隔 min(相关)−max(不相关) | **+0.986095** | **+0.996844** |
| 完全区分（排序 10/10 正确） | ✅ true | ✅ true |
| 10 对一次 batch 前向耗时 | 1.02 s | 1.26 s |

观察（事实，非推断）：
- 10/10 排序正确；最像"容易翻车"的第 10 对（胡须油 vs 护发油，同域相邻品类）仍判为不相关，但分数抬到 0.0138，是全部负例最高。
- 相关对全部饱和在 ≥0.9999、负例低至 3e-5，分数分布**高度双峰**：适合排序，不适合当概率或固定阈值。取"排序第一"是安全的，取 `score > 0.5` 也是安全的（间隔极大），但没有中间地带可供置信度分级。
- 换 instruction 会整体平移分数（相关 0.99995→0.99978，负例 0.0138→0.0028），排序不变。`<Instruct>` 是有效旋钮，不是装饰。

---

## 7. 正确性交叉验证

### 7.1 复现模型卡自带样例（最强证据）

用 README 原文样例 + 默认指令：

| 文档 | 本机 `p_yes` | 本机 margin | 模型卡公布分数 |
|---|---:|---:|---:|
| "The capital of China is Beijing." | 0.999498 | **+7.597** | 7.625 |
| "Gravity is a force that attracts…" | 0.000005 | **−12.192** | −11.375 |

排序正确（Beijing > Gravity）＝ true；首条 margin 与卡片公布值差 **0.03**，第二条差 **0.82**（本机 fp32，卡片 runs bf16 → 负向饱和区 logit 绝对差属预期量级）。模板、yes/no 读取位置、logit 组合方式由此得到端到端印证。

### 7.2 模板消融（证明模板是"承重"的，而非随便拼）

同一 10 对、仅改 suffix：

| suffix 变体 | min(相关) | max(不相关) | 间隔 | 排序仍正确 |
|---|---:|---:|---:|---|
| ✅ 模型卡正确模板 `…assistant\n<think>\n\n</think>\n\n` | 0.999923 | 0.013828 | +0.9861 | 是 |
| ❌ 缺 think 块 `…assistant\n` | 0.740790 | 0.000100 | +0.7407 | 是（但分数整体压低 0.26） |
| ❌ 误用 `<｜end▁of▁thinking｜>`（不在词表，被 byte-level 拆分） | 0.999628 | 0.027484 | +0.9721 | 是 |

结论：**在容易的冒烟集上，模板写错仍能保住排序，只是绝对分数漂移**。这恰恰是最危险的地方——排序不报警、阈值会漂移。故模板必须以 4.2 的三方一致为准，不能靠"跑出来看着对"验收。

---

## 8. 延迟 / 内存（OMP_NUM_THREADS=32）

方法：每个配置**独立进程**（保证 `VmHWM` 是单配置峰值而非累计峰值），1 次 warmup 后计时 N 次前向；峰值 RSS 取 `/proc/self/status:VmHWM`。两轮独立复测（轮1 `reps=5`，轮2 `reps=7`），脚本 `/tmp/rr_bench.py`。

实际输入长度（`make_document` 按目标长度二分填充，实测而非标称）：

| 设置 | 实际 token（batch=1 / batch=4 padded） |
|---|---|
| 512 | 486 / 487 |
| 1024 | 1014 / 1015 |
| 2048 | 2026 / 2027 |

### 8.1 主表（P50 为轮内中位数）

| batch | 标称长度 | 实际 tokens | P50 轮1 | P50 轮2 | min–max（轮2） | 单条摊薄（轮2） | 峰值 RSS 轮1 | 峰值 RSS 轮2 | tok/s（轮2） |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 512 | 486 | 0.414 s | **0.396 s** | 0.388–0.465 | 0.396 s | 3,916 MB | 3,907 MB | 1,227 |
| 1 | 1024 | 1014 | 1.908 s | 1.658 s | 1.313–2.181 | 1.658 s | 3,963 MB | 3,930 MB | 612 |
| 1 | 2048 | 2026 | 3.720 s | **3.649 s** | 2.795–3.866 | 3.649 s | 5,214 MB | 5,107 MB | 555 |
| 4 | 512 | 487 | 1.152 s | 1.259 s | 1.035–1.519 | 0.315 s | 5,108 MB | 4,990 MB | 1,547 |
| 4 | 1024 | 1015 | 2.556 s | 2.386 s | 2.304–2.715 | 0.597 s | 7,463 MB | 7,629 MB | 1,701 |
| 4 | 2048 | 2027 | 8.334 s | 8.111 s | 6.794–9.558 | 2.028 s | 11,372 MB | 11,185 MB | 1,000 |

（batch=1 即"单条"；两项任务要求"单条在 512/2048 各测一次"已由上表第 1、3 行覆盖。）

### 8.2 内存构成

| 项 | 数值 |
|---|---|
| `model.safetensors`（磁盘 bf16） | 1.19 GB |
| fp32 权重（加载后） | 2.38 GB（595,776,512 × 4 B） |
| 仅加载完成时 `VmHWM`（含 bf16 mmap 页 + fp32 上转） | **3,868 MB**（单独测量，三次一致） |
| batch=1 推理增量 | +40 ~ +1,350 MB（随长度） |
| batch=4×2048 峰值 | 11.2–11.4 GB |
| 加载耗时（页缓存热） | tokenizer 0.37 s + model 0.34 s（另一次 0.67–0.71 s） |

要点：`VmHWM` 在"仅加载"时就已到 3.87 GB，说明**上转 fp32 本身就是内存主因**（bf16 文件页仍 resident）；推理由此只在长序列/batch=4 时才显著抬升（batch=4×2048 的 attention 激活再叠加 ~7 GB）。

### 8.3 需要说明的异常（不掩饰）

- **batch=1 的 1024 token 明显偏离线性**：486→1014→2026 token 对应 0.40→1.66→3.65 s，第二段 4.2×（2.09× token）、第三段 2.2×（2.0× token）。两轮复测（1.908 s / 1.658 s）都复现，**不是单次抖动**。
- 可能原因（**未验证**）：机器 2 个 socket/2 NUMA node、128 逻辑核，而本轮只用 32 个 OMP 线程且**未做 CPU/NUMA 绑定**——线程与内存页可能跨 node 漂移，最坏落位可致 ~2× 差异；也可能是 oneDNN kernel 在序列长度阈值处切换（512 档 1,227 tok/s vs 1024 档 612 tok/s）。判定需 `numactl --cpunodebind=0` + `taskset` 绑定后复测（本轮未做）。
- 宿主 load average 18.4–25.0（128 逻辑核共享机，`free` 显示 90 GB 已被他人占用），所有绝对数字含竞争噪声；`min–max` 的离散度即为噪声量级（如同配置 2.795–3.866 s）。

---

## 9. 与 `BAAI/bge-reranker-v2-m3` 的加载/接口差异

未下载第二个模型（时间/带宽让位于主链路），下表为**文档级接口差异，本轮未执行验证**，仅用于说明"为什么不能共用一套封装"：

| 维度 | Qwen3-Reranker-0.6B（本测） | bge-reranker-v2-m3（待对照，未实测） |
|---|---|---|
| Transformers 类 | `AutoModelForCausalLM`（`Qwen3ForCausalLM`） | `AutoModelForSequenceClassification`（XLM-R 类） |
| 输入构造 | 手工拼 `system + <Instruct>/<Query>/<Document> + assistant 前缀 + think 块`，**必须**用其专属模板 | tokenizer 原生 `(query, passage)` 文本对编码，无 chat 模板 |
| 输出读取 | 取 `logits[:, -1, :]` 中 yes/no 两列做 2 类 softmax | 取 `logits`（`num_labels=1`）单一标量，通常再 sigmoid |
| 额外成本 | LM head 词表 151,669 → 最后一个位置的 logits 张量 `[B,151669]`；词嵌入/LM head 共享占 ~155 M 参数 | 单标量输出，无词表级 logits |
| 截断上限 | 模型卡 `max_length=8192`（架构 40960） | 通常 8192 |
| 结论 | **不能**把 Qwen3-Reranker 塞进 CrossEncoder 式的 `(q,d)` 直通封装（除非用其自带 `1_LogitScore` ST 模块）；二者必须各留一个 adapter | 同左 |

对齐策略建议：`reranking` 层只暴露 `score(query: str, docs: list[str], instruction: str|None) -> list[float]`，Qwen3 与 BGE 各一个实现，模板与 token id 全部走配置/常量文件并配单测（断言 `yes==9693, no==2152, len(prefix)==39, len(suffix)==9`）。

---

## 10. 风险与未验证项

| # | 风险 / 未验证 | 影响 | 建议 |
|---|---|---|---|
| 1 | **分数饱和**（0.9999 / 3e-5，无中间地带） | 跨 query 绝对分数不可比；固定阈值不可用；候选少时"排序"信息量被压缩 | 只用同 query 内排序；如需分级，改用 `margin=ly−ln`（本测范围 −12.2 ~ +10.3，区分度更好）；阈值必须用真实标注集校准 |
| 2 | **batch=1 @1024 token 超线性**（复现两次，原因未定） | 候选文本落在 1k token 附近时延迟预估可能偏差 ~1.8× | 用 `numactl --cpunodebind=0` 复测；或固定 `batch=4/8` 规避 |
| 3 | 宿主共享、load 18–25 | 所有 P50 含竞争噪声，非干净基准 | 在独占窗口复测；报告 P95 与多轮分位 |
| 4 | NFS **冷缓存**加载未测（本轮页缓存热，0.34 s） | 首次部署/重启的冷加载时间未知，可能数分钟 | 重启后首跑计时一次 |
| 5 | **bf16 / AMX 未测**（Xeon Gold 6430 属 Sapphire Rapids，具备 bf16 指令） | 当前 fp32 是保守基线，未必是 CPU 上最优；改成 bf16 可能同时降内存与提速 | 单列对比实验（fp32 vs bf16 vs `torch.compile`），需先确认精度不退化 |
| 6 | 长上下文 >2048 未测（模型卡用 8192） | 真实 Amazon 商品文本若拼接 details 可能超 2k token，延迟按 O(n²) 外推会更快恶化 | 先用 All_Beauty `title+details` 实际长度分布定档 |
| 7 | 真实商品文本长度未知 | 本测 512/1024/2048 是标称档；若真实文本 ~100 token，则延迟远低于本表 | 用 metadata 统计 token 长度分位，替换本表的"标称档" |
| 8 | 20 候选 ≈6 s/query 是**外推**（5×batch=4@512） | 影响 Controller 预算设计 | 与 Planner/Judgment 的 API 延迟一起做端到端排队测量 |
| 9 | `accelerate` 未安装 | 无法用 `device_map`；当前绕开（`.to("cpu")`）不影响结果 | 若要 `device_map`/多进程，需在依赖里显式固定 `accelerate` 版本 |
| 10 | 仅 CPU；MUSA/GPU 未测（Q11 明确第一阶段不接） | GPU 上 Qwen3-Reranker 需 flash-attn / 精度转换，接口另测 | 保持 CPU 基线，GPU 作为后续优化项单独立项 |
| 11 | 10 对手写样本量太小 | 只能证明"链路对、能排序"，不能证明"排序质量" | R4 的模型效果验证必须换真实 All_Beauty 候选 + 标注集（≥200 对） |

### 对 Phase 1 的直接含义（事实陈述）

- 单 query 精排 20 候选（512 token、batch=4）≈ 6 s，峰值内存 ≈5 GB。
- 若候选 50 且文本 2048 token，外推 ≈ 13 个 batch × 2.03 s ≈ **26 s / query**，峰值 ≈11 GB → 此时要么降候选数、要么截断文本、要么加大 batch 摊薄（batch=8 未测）。
- 分数只做排序、不做概率；`score` 建议在内部命名为 `rerank_score`，避免下游误当购买概率或约束判定依据。
- 以上均为**单 query 串行**测量；并发（多 query 同时推理）未测，线程数固定 32，宿主机尚有 96 个逻辑核未被本进程使用。

---

## 11. 复现步骤

```bash
# 服务器侧（每次新 shell 必做）
cd $HOME/Agentic-Generative-Recommendation && source scripts/env_s4000.sh
export OMP_NUM_THREADS=32
export MODEL_PATH=/mnt/agr-data/hf/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473

python /tmp/rr_inspect.py      # config / chat template / yes-no id
python /tmp/rr_score_test.py   # 10 对 × 2 instruction，写 /tmp/rr_pairs.json
python /tmp/rr_verify.py       # 模型卡样例复现 + 模板消融，写 /tmp/rr_verify.json
bash   /tmp/rr_run_bench.sh    # 6 配置 × 5 reps，写 /tmp/rr_bench_results.jsonl
python /tmp/rr_bench.py --batch 1 --len 2048 --reps 7 --out /tmp/x.json   # 单配置复测
```

产物（均在服务器 `/tmp`，不进仓库）：`rr_inspect.log`、`rr_pairs.json`、`rr_verify.json`、`rr_bench_results.jsonl`、`rr_bench_b{1,4}_l{512,1024,2048}.json`、`rr_bench2_*.json`、`rr_dl3.log`、`rr_readme.md`。

项目仓库仅新增本文件 `docs/audit/reranker-smoke.md`；未修改任何源码、配置或数据。
