# sglang-flashnext-fp8kv-sm120

[中文版 README →](README.zh.md)

## Hardware & environment

| Item | Value |
|------|-------|
| Host | CT110 `sglang-qwen4exp` — LXC container on Proxmox VE, Debian 13 (trixie) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB (SM120), single-GPU `CUDA_VISIBLE_DEVICES=0` |
| Driver / CUDA | 610.57.04 / CUDA 13.0 (`TORCH_CUDA_ARCH_LIST=12.0`) |
| CPU / RAM | 8 vCPU / 118 GB (PLE 47.7 GB pinned + HiCache host 13 GB live alongside) |
| torch / sglang | 2.13.0+cu130 / pennyroyal tree `0.0.0.dev1+g2c675da09` (editable, ff-tracked to tag pennyroyal-v2.5.1) |
| Endpoint | http://10.10.4.12:8000, served-model-name `Qwen3.8-Flash-Next-NVFP4` |

## Measured performance (CT110 live, c=1 streaming, temperature=0, greedy)

| Metric | Value | Condition |
|--------|-------|-----------|
| TTFT short request | ~290–400 ms | warm radix; stabilizes at low end once cache hot |
| TTFT @ 30k-token prompt | ~0.6 s | ≈50k tok/s prefill throughput (chunked-prefill 4096) |
| Decode hot burst (400 tok) | ~1.4 s → ≈280–290 tok/s | NEXTN spec-dec (steps 2 / topk 1 / draft 4); per-step acceptance lifts the rate |
| Decode sustained (long outputs) | ~96–110 tok/s | accept length decays after initial burst on long chains |
| Quality gate | pass | 188K-token dual-needle NIAH + ×3 no-repeat sanity |

Reproduce with any streaming client against `POST /v1/chat/completions` (`stream_options.include_usage` for exact token counts). Chunk-counting underflows true tok/s — use `usage.completion_tokens`, not SSE chunk count.

Production tuning overlay for **Qwen3.8-Flash-Next** served by sglang (pennyroyal-v2.5.1 tree) on a single RTX PRO 6000 (SM120, Blackwell), CT110 @ 10.10.4.12:8000.

Stack: `dealignai Qwen3.8-Flash-Next-ABLITERATED-NVFP4` weights + FP8 KV + HiCache + MXFP8 online projections + expert cold pool → true 1M single-window context, ~99 tok/s hot decode.

## Layout

```
config/    live YAML, systemd unit, keep-mask JSON
patches/   overlay diff vs pennyroyal-v2.5.1 tag + standalone expert_cold_pool.py
```

## Optimization inventory

### 1. Online MXFP8 projections (`SGLANG_SM120_ONLINE_MXFP8=true`)
Loads dense projections, HyperConnection mixing weights, and output head as MXFP8 (row-level UE8M0 scale, backend `FLASHINFER_CUTLASS`) instead of BF16. NVFP4 experts and BF16 recurrent state untouched.
- Result: profiled KV pool 451,776 → 693,056 tokens (+53%); decode hot ~96–109 tok/s (was ~42); short-request TTFT ~227 ms; no quality regression (NIAH dual-anchor passes).
- Journal signature: `Flash-Next online MXFP8 projection ready: ... backend=Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS scale=UE8M0`.

### 2. Expert cold pool — keep-mask + dynamic slots (`patches/expert_cold_pool.py`, `config/expert_keep_330_final.json`)
Offloads non-kept MoE experts to host-pinned memory; GPU keeps 330/512 experts per layer across all 48 layers, plus 64 dynamic staging slots for demand rotation.
```
Environment=SGLANG_EXPERT_KEEP_MASK=/opt/sglang-config/expert_keep_330_final.json
Environment=SGLANG_EXPERT_KEEP_OFFLOAD=1
Environment=SGLANG_EXPERT_COLD_POOL_SLOTS=64
Environment=SGLANG_COLD_DEBUG=1
```
- Result: KV pool reaches full cap `#tokens: 1,572,864` (fp8_e4m3 K/V ~9 GB each); host pinned 24.15 GB, dynamic=True.
- **k-alignment rule:** hook default `k` must equal model's `num_experts_per_tok` (=10 here); otherwise set `SGLANG_COLD_TOPK`. A mismatch silently misaligns `hot_min` row grouping — decode degrades with no error.
- **Do not shrink keep/slots below this tier** for VRAM reasons: pure static keep-mask (keep-only, tiny slots) flattens router distribution → "hmm hmm" filler loops in long thinking chains. keep330+64 restores diversity.

### 3. FP8 KV cache + HiCache hierarchical L2
`kv-cache-dtype: fp8_e4m3` pairs with HiCache (nvfp4 KV has silent corruption on eviction reload, #36121 — fp8 path unaffected).
```
enable-hierarchical-cache: true
hicache-size: 13                 # GB; PLE is huge (47.7G) so stay at tested ceiling
hicache-host-memory-mode: cache
hicache-write-policy: write_through   # write_back zeroes hit rate after restart
hicache-io-backend: kernel
hicache-mem-layout: page_first
hicache-storage-prefetch-policy: timeout
```
Known side-effect: host pool (827,008) < device pool (1,572,864) → L2 coverage ~53%; short sessions hit device radix regardless. Raise `hicache-ratio` only if long-session second-access actually slows.

### 4. Pool ceiling pinning + YaRN ×4 for true 1M window
```
max-total-tokens: 2100000        # ceil(base × lanes × 1.05); pinned > profiled is cap-protection, engine takes profiled — normal, not misconfig
context-length: 1048576          # via json-model-override-args yarn factor: 4.0, original_max_position_embeddings: 262144
```
- Ceiling formula: `ceil(262144 × lanes × 1.05)`. Expansion chain observed: MXFP8-only 693K → +cold pool 1.57M → re-pin profiled 2,245,248.
- Boot shows benign `Warning: User-specified context_length ... greater than derived` on main+draft workers — filter `grep -v Warning` when triaging journal.

### 5. NEXTN speculative decoding (MTP on)
```
speculative-algorithm: NEXTN
speculative-num-steps: 2 / eagle-topk: 1 / num-draft-tokens: 4
speculative-draft-model-quantization: unquant
```
Draft costs ~1.07 GB packed KV per side; 5 mamba slots per lane → `max-mamba-cache-size: 60` serves 12 lanes (`max-running-requests: 12`).

### 6. SM120 attention/kernel routing
```
linear-attn-decode-backend: flashinfer
linear-attn-prefill-backend: flashinfer    # SM120 auto-resolve falls back to triton without explicit pin
mamba-ssm-dtype: bfloat16 / SGLANG_MAMBA_CONV_DTYPE=bfloat16
```
Overlay patches (`patches/overlay-v2.5.1.diff`) touch: fast_topk kernel, QSA kernel/mqa (unaligned prefix chunked-prefill fix), MoE topk, kv_cache_configurator (gcfix), model_runner, eagle_worker_v2.

### 7. Memory pressure discipline (PLE + JIT)
- `--ple-offload-embedding` keeps the 47.68 GB fp8 embedding in host RAM (PENNY_PLE_BACKEND=ram).
- After PLE pins, host RAM is tight → **serialize all JIT compilation**: `MAX_JOBS=1`, `FLASHINFER_NINJA_JOBS=1`, `TORCHINDUCTOR_COMPILE_THREADS=1`. Multi-threaded nvcc storms freeze the node (AntigravityAI-confirmed).
- `expandable_segments` deliberately NOT set — at extreme `mem-fraction-static: 0.98` it steals ~7% of the pool.
- All per-instance caches on CT root disk (`/opt/pennyroyal-cache/...`), not data disk, per storage-tiering rule.

### 8. Observability
- `enable-cache-report: true` → `usage.prompt_tokens_details.cached_tokens` = radix L1 hits (HiCache L2 reload not counted; `null` right after restart until first radix hit).
- Quality gate on every reboot: 188K-token dual-needle NIAH recall + ×3 short-request no-repeat sanity. Expected after boot: 0 restarts, SHRUNK line, MXFP8 signatures, KV allocated == profiled, `Memory pool end avail ≥ 2 GB`.

## Overlay apply procedure

```bash
cd /opt/pennyroyal
git diff > /tmp/backup.patch                # if working tree diverged
git -c pull.rebase=false pull --ff-only --autostash origin pennyroyal-v2.5.1
git apply --3way /path/to/patches/overlay-v2.5.1.diff   # re-check conflicts on stash
systemctl restart sglang-dealignai-fp8kv-hicache.service
journalctl -u sglang-dealignai-fp8kv-hicache.service --since today | grep -v Warning
```
