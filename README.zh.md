# sglang-flashnext-fp8kv-sm120（中文版）

[English README →](README.md)

## 硬件与环境

| 项 | 值 |
|------|-------|
| 主机 | CT110 `sglang-qwen4exp` — PVE 上的 LXC 容器，Debian 13 (trixie) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition，96 GB（SM120），单卡 `CUDA_VISIBLE_DEVICES=0` |
| Driver / CUDA | 610.57.04 / CUDA 13.0（`TORCH_CUDA_ARCH_LIST=12.0`） |
| CPU / RAM | 8 vCPU / 118 GB（PLE 钉载 47.7 GB + HiCache host 13 GB 同机共存） |
| torch / sglang | 2.13.0+cu130 / pennyroyal 树 `0.0.0.dev1+g2c675da09`（editable 形态，ff 跟随 tag pennyroyal-v2.5.1） |
| 端点 | http://10.10.4.12:8000，served-model-name `Qwen3.8-Flash-Next-NVFP4` |

## 实测性能（CT110 在线实测，并发=1，streaming，temperature=0）

| 指标 | 数值 | 条件 |
|------|------|------|
| TTFT 短请求 | ~290–400 ms | radix 热命中后稳定在下沿 |
| TTFT @ 30k token prompt | ~0.6 s | prefill 吞吐 ≈50k tok/s（chunked-prefill 4096） |
| decode 热态突发（400 tok） | ~1.4 s → ≈280–290 tok/s | NEXTN 投机解码（steps 2 / topk 1 / draft 4），每 step 多 token 接受拉高速率 |
| decode 长输出稳态 | ~96–110 tok/s | 长思考链上接受长度随初始突发后衰减 |
| 质量门 | 通过 | 188K token 双锚点 NIAH + ×3 no-repeat 检查 |

复测方式：任意 streaming 客户端打 `POST /v1/chat/completions`，加 `stream_options.include_usage` 拿精确 token 数。注意：按 SSE chunk 数统计会低估真实 tok/s——以 `usage.completion_tokens` 为准，不是 chunk 计数。

Qwen3.8-Flash-Next 在生产环境的 sglang 调优档案：pennyroyal-v2.5.1 源码树 + 单卡 RTX PRO 6000（SM120 / Blackwell），部署于 CT110 @ 10.10.4.12:8000。

技术栈：`dealignai Qwen3.8-Flash-Next-ABLITERATED-NVFP4` 权重 + FP8 KV + HiCache + Online MXFP8 投影 + 专家冷池 → 真 1M 单窗口上下文，热态 decode 约 99 tok/s。

## 目录结构

```
config/    现役启动 YAML、systemd 单元、keep-mask JSON
patches/   基于 pennyroyal-v2.5.1 tag 的 overlay diff + 独立模块 expert_cold_pool.py
```

## 优化项全清单

### 1. Online MXFP8 投影（`SGLANG_SM120_ONLINE_MXFP8=true`）
加载期把 dense projections、HyperConnection mixing weights、output head 从 BF16 转成 MXFP8（行级 UE8M0 scale，后端 `FLASHINFER_CUTLASS`）。NVFP4 experts 和 BF16 recurrent state 不动。
- 实测收益：profiled KV 池 451,776 → 693,056 tokens（+53%）；热态 decode ~96–109 tok/s（原 ~42）；短请求 TTFT ~227 ms；质量无退化（NIAH 双锚点全中）。
- journal 签名行：`Flash-Next online MXFP8 projection ready: ... backend=Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS scale=UE8M0`。

### 2. 专家冷池 — keep-mask + 动态槽位（`patches/expert_cold_pool.py`、`config/expert_keep_330_final.json`）
非保留的 MoE 专家卸载到 host 钉载内存；GPU 每层保留 330/512 个专家（共 48 层），外加 64 个动态 staging 槽做需求轮换。
```
Environment=SGLANG_EXPERT_KEEP_MASK=/opt/sglang-config/expert_keep_330_final.json
Environment=SGLANG_EXPERT_KEEP_OFFLOAD=1
Environment=SGLANG_EXPERT_COLD_POOL_SLOTS=64
Environment=SGLANG_COLD_DEBUG=1
```
- 实测收益：KV 池顶满 YAML cap `#tokens: 1,572,864`（fp8_e4m3，K/V 各约 9 GB）；host pinned 24.15 GB，dynamic=True。
- **k 对齐规则：** hook 默认 `k` 必须等于模型的 `num_experts_per_tok`（本机 =10）；移植到其他模型时显式设 `SGLANG_COLD_TOPK`。不匹配会静默错位 `hot_min` 行分组——decode 质量退化且不报错。
- **keep/slots 不要为显存降档：** 纯静态 keep-mask（keep-only、极小 slots）会压平路由分布，长思考链漂进 "hmm hmm" 复读填充。keep330 + slots64 刚好恢复多样性，这是下限档。

### 3. FP8 KV cache + HiCache 分层 L2
`kv-cache-dtype: fp8_e4m3` 与 HiCache 配套（nvfp4 KV 驱逐回载有静默腐坏问题 #36121，fp8 路径不受影响）。
```
enable-hierarchical-cache: true
hicache-size: 13                 # GB；PLE 已占 47.7G，取实测上限不加码
hicache-host-memory-mode: cache
hicache-write-policy: write_through   # write_back 重启后命中率归零
hicache-io-backend: kernel
hicache-mem-layout: page_first
hicache-storage-prefetch-policy: timeout
```
已知副作用：host 池（827,008）< device 池（1,572,864）→ L2 覆盖约 53%；短会话直接命中 device radix，不受影响。只有长会话*第二次*访问确实变慢时才上调 `hicache-ratio`。

### 4. 钉池上限 + YaRN ×4 实现真 1M 窗口
```
max-total-tokens: 2100000        # ceil(base × lanes × 1.05)；钉值 > profiled 是 cap-protection 语义，引擎取 profiled——正常行为，不是配置错误
context-length: 1048576          # 经 json-model-override-args 的 yarn factor: 4.0，original_max_position_embeddings: 262144
```
- 上限公式：`ceil(262144 × lanes × 1.05)`。实测扩容链：MXFP8-only 693K → 加冷池 1.57M → 重钉后 profiled 2,245,248。
- 启动时 main + draft worker 各打一条无害的 `Warning: User-specified context_length ... greater than derived`——排查日志时先 `grep -v Warning` 再下结论。

### 5. NEXTN 投机解码（MTP 开启）
```
speculative-algorithm: NEXTN
speculative-num-steps: 2 / eagle-topk: 1 / num-draft-tokens: 4
speculative-draft-model-quantization: unquant
```
draft 侧 packed KV 每份约 1.07 GB；每路 5 个 mamba 槽 → `max-mamba-cache-size: 60` 撑 12 路（`max-running-requests: 12`）。

### 6. SM120 attention/kernel 后端路由
```
linear-attn-decode-backend: flashinfer
linear-attn-prefill-backend: flashinfer    # SM120 下不显式钉会回落 triton
mamba-ssm-dtype: bfloat16 / SGLANG_MAMBA_CONV_DTYPE=bfloat16
```
overlay 补丁（`patches/overlay-v2.5.1.diff`）触及：fast_topk kernel、QSA kernel/mqa（unaligned prefix chunked-prefill 修复）、MoE topk、kv_cache_configurator（gcfix）、model_runner、eagle_worker_v2。

### 7. 内存压力纪律（PLE + JIT）
- `--ple-offload-embedding` 把 47.68 GB fp8 embedding 常驻 host RAM（PENNY_PLE_BACKEND=ram）。
- PLE 钉载后 host 内存紧张 → **JIT 编译全部串行化**：`MAX_JOBS=1`、`FLASHINFER_NINJA_JOBS=1`、`TORCHINDUCTOR_COMPILE_THREADS=1`。多线程 nvcc 编译风暴会导致整节点僵死（AntigravityAI 实证）。
- `expandable_segments` 刻意不设——极限 `mem-fraction-static: 0.98` 下它会偷走约 7% 的池。
- 实例专属缓存全部落本 CT 根盘（`/opt/pennyroyal-cache/...`），不放数据盘（存储分层红线）。

### 8. 可观测性
- `enable-cache-report: true` → 响应里 `usage.prompt_tokens_details.cached_tokens` = radix L1 命中数（HiCache L2 回载不计入；重启后至首次 radix 命中前为 `null`，单次冷探测不能判定开关失效）。
- 每次重启验收：188K token 双锚点 NIAH 召回 + ×3 短请求 no-repeat 检查。期望终态：0 restarts、SHRUNK 行、MXFP8 签名、KV allocated == profiled、`Memory pool end avail ≥ 2 GB`。

## Overlay 重放流程

```bash
cd /opt/pennyroyal
git diff > /tmp/backup.patch                # 工作树有本地改动时先存档
git -c pull.rebase=false pull --ff-only --autostash origin pennyroyal-v2.5.1
git apply --3way /path/to/patches/overlay-v2.5.1.diff   # 冲突时 stash 仍在
systemctl restart sglang-dealignai-fp8kv-hicache.service
journalctl -u sglang-dealignai-fp8kv-hicache.service --since today | grep -v Warning
```
