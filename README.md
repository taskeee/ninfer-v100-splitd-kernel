# NInfer on Tesla V100 (sm_70): the 1CatAI Split-D D256 prefill kernel, wired in

**English** — [jump to the English section](#english) · **中文** — 见下方中文部分

<!-- topics: v100 tesla-v100 sm70 volta ninfer llamacpp flash-attention split-d splitkv d256 cuda
     qwen3 attention-kernel prefill inference-optimization long-context llm-inference kernel -->

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](licenses/NInfer-LICENSE-Apache-2.0.txt)
[![Kernel license: MIT](https://img.shields.io/badge/kernel%20license-MIT-green.svg)](licenses/sm70-attn-LICENSE-MIT.txt)
[![GPU: Tesla V100 sm_70](https://img.shields.io/badge/GPU-Tesla%20V100%20sm__70-76b900.svg)](#)
[![CUDA: 12.8](https://img.shields.io/badge/CUDA-12.8-76b900.svg)](#)

---

## 中文

### 这是一份什么

把 **1CatAI 的 Split-D D256 FlashAttention 内核**（来自 [`fishlikeX/sm70-attn`](https://github.com/fishlikeX/sm70-attn)）
接进 **NInfer 在 Tesla V100（sm_70）上的长上下文预填路径**，并附上本机实测数据。

**解决的问题（你要是搜到这里，多半也是这个）**：V100 单卡跑 27B 级别模型、上下文拉到十几二十万 token 时，
**首字等待长到没法用**（实测满窗口冷跑 **10 分 35 秒**），而算力还有大量余量——瓶颈是注意力内核。
换上这个内核后，同样的活变成 **7 分 29 秒**，**预填吞吐 +36%~41%**，而**解码速度不受影响**。

一句话：**这是"预填/首字等待"的优化，不是"吐字速度"的优化。** 如果你的痛点是每秒吐字少，换这个内核不会救你（详见"实测数据"里的解码行）。

### 先说清楚：这东西是谁做的（重要）

- **本仓库的内容不是我写的代码**。我叫 **taskeee**，**不写代码、也看不懂英文**。
- 仓库里的代码、说明、实测、脚本，**全部是我让 AI（DeepSeek Harness 里的 AI 助手）帮我整理和发布的**。
  换句话说：**这是"用户让 AI 发的内容"**，如果格式或措辞看着像 AI 写的，那就是 AI 写的。
- **代码不是 AI 发明的**：内核是 **1CatAI / `fishlikeX/sm70-attn` 的原作**（字节未改搬用），
  宿主引擎是 **Neroued/ninfer**。AI 做的是**接线 + 实测 + 写文档**。功劳归原作者。
- 实测数据来自**我这一台机器**（单卡 V100），**你自己机器上的数字会不一样**，请以自测为准。

### 东西是哪来的（逐条注明出处，便于你核对）

| 组件 | 出处 | 许可 | 本仓库怎么处理 |
|---|---|---|---|
| **注意力内核本体**<br>`kernel/fattn-sm70-d256-kernel.cuh`（49,003 B） | [`fishlikeX/sm70-attn`](https://github.com/fishlikeX/sm70-attn)（1CatAI 的 llama.cpp 深分叉）<br>commit `707cf247c1beb5e7c701f47fe6f828c5b84b729d`（2026-09-22） | **MIT**（`LICENSE`：Copyright (c) 2023-2026 The ggml authors） | **字节未改**（SHA256 见下） |
| **宿主推理引擎** | [`Neroued/ninfer`](https://github.com/Neroued/ninfer)（★2,485） | **Apache-2.0** | 未附带源码；只提供接线说明 |
| **V100 / sm_70 移植版** | 本机现役版本，**未公开**（见下"关于移植版"） | Apache-2.0（承自 NInfer） | 只提供接线片段 |
| **接线代码**<br>`integration/gqa_attention_volta_splitd.cu`（18,072 B） | AI 在本机上写的，**本仓库首次公开** | Apache-2.0 | 完整提供 |
| **依赖闭包**（**本仓库不附带**） | NVIDIA **CUTLASS/cute**（152 头文件）+ **flash-attention**（8 头文件），上游 `ggml/src/ggml-cuda/sm70-vendor` | Apache-2.0 / BSD-3-Clause（见 `licenses/`） | 见"接线方法 ②"的拷贝命令 |

**关于"V100 移植版"**：本机的 NInfer 是**已加 sm_70/Volta 支持的移植版本**（其 `CMakeLists.txt` 开头的注释写明
*"Upstream NInfer is compiled only for sm_120a. This port adds sm_70 (Volta / Tesla V100) as a peer compile-time target."*）。
**我没有找到这个移植版的公开可下载源码**——查过 [`dollarwong/ninfer-v100`](https://github.com/dollarwong/ninfer-v100)（描述称是 V100 移植，但仓库内 **0 个文件**）、
也查过 `Neroued/ninfer` 的 `src/ops/launcher/`（52 个文件，**不含** `gqa_attention_volta_flash.cu`）。
所以：**本仓库不提供"打上去就能用"的补丁**，而是给出**改哪个文件、加哪几行**的完整说明 + 可直接使用的内核源码。
如果你手上就是同一个移植版，照着"接线方法"一节改即可（改动量：**新增 1 个文件 + 3 处修改**）。

### 实测数据（本机，单卡 V100-SXM2-32GB）

**测试条件**：CUDA 12.8.93、驱动 581.80、`CMAKE_CUDA_ARCHITECTURES=70`、KV `int8`、MTP `--draft-tokens 3`、
固定内容（`--greedy`）、冷预填（`cache 0%`）、同一提示词、未开性能探针。
"老内核" = 换内核前的归档二进制（同一引擎，走 vendored llama.cpp MMA FA 路径）。

| 提示词深度 | 内核 | 预填 tok/s | 首字等待 | 解码 tok/s | 接受率 |
|---:|---|---:|---:|---:|---:|
| 83,177 | 老 | 847.8 | 98.1 s | 39.09 | 46.3% |
| 83,177 | **新** | 848.1 | **98.1 s** | **42.61** | 52.1% |
| 196,320 | 老 | 437.1 | 448 s | 26.30 | 44.4% |
| 196,320 | **新** | **594.1** | **330 s** | **30.57** | 54.4% |
| 239,183 | 老 | 376.5 | **10 分 35 秒** | 30.62 | 64.55% |
| 239,183 | **新** | **532.5** | **7 分 29 秒** | 30.21 | 61.67% |

**怎么读这张表（别踩我踩过的坑）**

1. **预填大幅变快**：19.6 万深度 **+36%**，24 万深度 **+41%**，首字等待 **−3 分 06 秒**。
2. **解码速度跟内核无关**：内核只走预填路径，解码走另一条路。表里解码的上下浮动来自
   **接受率随内容漂移**（44%~65%），不是快慢变化。
3. **解码速度主要由"深度"和"内容"决定**，不是内核：
   同一台机器同一个新内核，8.5 万深度 **42.6**、19.6 万深度 **30.6**、24 万深度 **30.2**。
   **深度越深越慢是 KV 读取量的物理账**（满窗口 KV 约 7.9 GB），换任何内核都一样。
4. **接受率是内容决定的**：真实 agent 内容（工具调用/代码/续写）60~75%，人造重复填充文 44%~55%。
   **比较 tok/s 时必须同时报"深度 + 接受率"，否则数字不可比。**

**真实负载参考（不是合成提示词）**：同一台机器真实 agent 会话实测
7.5~7.9 万深度 13 轮 = **42.6~54.6 tok/s**；13.2 万 = 40.9；16.4 万 = 34.6；18.9 万 = 36.2；19.0 万 = 28.5；19.2 万 = 34.1。

**一条实用边界**：如果你用 DSH（DeepSeek Harness）这类会在上下文 80% 处自动压缩的客户端，
`--max-context 245000` 的**实际可用天花板是约 196K**——比这更深的数字你在真实使用里摸不到，
**不要拿 24 万档的数字当自己的体验基准**（我当初就是被这个数字绕进去的）。

原始数据（可直接核对）：`bench/probe_old196k.json`、`probe_new196k.json`、`probe_old85k.json`、
`probe_new85k.json`、`oldkernel-239k-greedy.json`、`newkernel-239k-greedy.json`、`oldkernel-239k-mtp.json`、`newkernel-239k-mtp.json`。
复现脚本：`bench/ninfer_bench.py`、`bench/probe_live.py`、`bench/run_greedy_ab.sh`、`bench/run_depth_control.sh`。

### 接线方法（改动量：新增 1 文件 + 3 处修改）

原始代码片段见 `integration/snippet-flash-launcher.txt` 与 `integration/snippet-cmake.txt`（从现役文件里**原样抽出**，未经改写）。

**① 新增文件**：`src/ops/launcher/gqa_attention_volta_splitd.cu`（本仓库 `integration/` 里那份）。
它做四件事：Q 的 `bf16→f16` 暂存（`int8` KV 时过与老路径同一个 normalized Hadamard）、
补齐到 64 的倍数、调 Split-D 内核、输出 `f32→bf16` 回写；暂存用一次 `cudaMalloc`（约 12.6 + 25.2 MiB）。

**② 把内核与依赖放进树里**：`third_party/llama_cpp_sm70_d256/`
= 内核 `fattn-sm70-d256-kernel.cuh`（本仓库 `kernel/` 那份）+ 依赖闭包 `sm70-vendor/`。

那份闭包**本仓库不附带**（162 个头文件 / 14 MB，全是别人的东西），但它**上游就有，原样一份**：

```bash
# 从上游仓库取闭包（NVIDIA CUTLASS/cute 152 个头文件 + flash-attention 8 个）
git clone --depth 1 https://github.com/fishlikeX/sm70-attn.git /tmp/sm70-attn
mkdir -p third_party/llama_cpp_sm70_d256
cp -r /tmp/sm70-attn/ggml/src/ggml-cuda/sm70-vendor third_party/llama_cpp_sm70_d256/
cp kernel/fattn-sm70-d256-kernel.cuh third_party/llama_cpp_sm70_d256/
```

来源与许可（上游 `sm70-vendor/README.md` 原文记载）：`cute/` + `cutlass/` 取自
[NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) commit `62750a2b…`（Apache-2.0）；
`flash/`（8 个头文件）取自 [zhinianqin/flash-attention-v100](https://github.com/zhinianqin/flash-attention-v100)
commit `c2eda5e6…`（BSD-3-Clause）。

**③ 在 `src/ops/launcher/gqa_attention_volta_flash.cu` 里加钩子**（本机行号 38–48 声明、630–660 调用）：

```cpp
// 声明（放在该文件的 detail 命名空间里）
bool volta_splitd_requested();

template <typename Geometry>
bool volta_splitd_block(const void* q_bf16, const void* k_f16, const void* v_f16, int rows,
                        int kv_len, int kv_offset, const void* positions, int position_begin,
                        float scale, bool int8_q, void* out_bf16, cudaStream_t stream);

// 调用（放在 Q-block 循环里，老内核调用之前）
if (volta_splitd_requested()) {
    if (volta_splitd_block<Geometry>(/* … */)) {
        continue;            // 成功即跳过老路径；返回 false 就自动落回老内核
    }
}
```

**④ 在 `src/CMakeLists.txt` 里加源文件与两个 include**（本机行号 346 与 357–360）：
把新 `.cu` 加进 sm_70 源表，并把 `third_party/llama_cpp_sm70_d256` 与
`third_party/llama_cpp_sm70_d256/sm70-vendor` **只作用到这一个文件**（闭包内部按 `cute/...` 直接包含，
所以父目录不够，`sm70-vendor` 本身也要进 include）。

### 操作须知（都是实测踩出来的，照着做能省几小时）

- **回退开关**：`NINFER_VOLTA_SPLITD=0` 强制回老内核（同一份二进制里两条路都在）。**发行版默认是开**。
- **怎么确认换没换成功**：**别只看 `cp` 的退出码**。往正在运行的引擎上覆盖二进制会报
  `Text file busy`，**老内核会继续在跑而你以为换了**。可靠判据是**预填速度指纹**：
  19.6 万深度冷预填 **437 = 老内核 / 594 = 新内核**（8.5 万深度两边都 ~848，**区分不了**）。
- **换二进制必须重启引擎**，而且**会清空 KV 前缀缓存** ⇒ 下一个新回合要吃一次满窗口冷预填。
- **契约测试必跑**（引擎自带 `ninfer_softmax_attention_test`，且必须 `CUDA_VISIBLE_DEVICES=1` 指到 V100）：
  这是唯一能抓到**布局/stride 错误**的手段——**速度数字完全看不出这个错**。
- **布局坑（我踩过）**：内核按**调用者给的 stride** 寻址。上游 launcher 用 head-major
  （`q_head_stride = q_pad * D`），而我的暂存缓冲是**行主序** ⇒ 必须传**行主序 stride**，
  否则等于喂错行/头（测试里表现为 `T=66 keys=129` 那档偏差 3~11%）。
- **因果掩码**：内核自己按 `(kv_len, kv_offset)` 算可见性，**不需要**外层再建 mask / mask 显存。
- **CUDA graph capture 内不能做的事**：`cudaFuncSetAttribute`、任何 D2H 同步（会作废 capture）。
  新路径检测到 capture 中且属性未设置时**自动回退老内核**。
- **测速时不要开性能探针**（`NINFER_TRACE_ATTN` 之类）：探针本身会让预填慢约 2.5%。

### 哪些路我试过、别重试（省你时间）

- **改 vendored flash 内核的配置表**：`ncols 32→64` 慢 18%（共享内存涨到 53 KB ⇒ 每 SM 只剩 1 个 CTA）；
  `Q_in_reg=true` 慢 **5.5 倍**；`nbatch_fa=16` 直接编不过（有 `static_assert(%32)`）。**这个内核的配置表救不了。**
- **用 DFlash2 解释"长上下文变慢"**：在本引擎的 V100 构建里 DFlash2 **根本没实现**
  （`src/ops/dflash2_sm70_stub.cu` 全是抛异常的桩，上游 kernel 要 sm_80+）⇒ 它**不可能是**你速度变化的原因。
- **为 GDN 写并行内核**：实测 GDN 只占预填 **8.2%**，砍半也只省 4 秒，不值得。
- **草稿深度 K=4**：10 万深度比 K=3 快 13%，但 16 万深度只有 +1%，**两档矛盾** ⇒ 保持 K=3。

### 许可

- 内核：**MIT**（`fishlikeX/sm70-attn`）——见 `licenses/sm70-attn-LICENSE-MIT.txt`。
- 接线代码：**Apache-2.0**（派生自 NInfer 的源码形态）。
- 依赖闭包：**BSD-3-Clause**（NVIDIA CUTLASS/cute、flash-attention）——见 `licenses/`。
- **不包含** NInfer 引擎源码本体；如需请去 [`Neroued/ninfer`](https://github.com/Neroued/ninfer)。

### 校验值（便于你确认拿到的是同一份东西）

```
kernel/fattn-sm70-d256-kernel.cuh
  SHA256  f3a45709ed2c88bd466912e7c4e2855bec6226ca4fda918d17855fbe2cb1f71e
  size    49,003 B
  source  fishlikeX/sm70-attn @ 707cf247c1beb5e7c701f47fe6f828c5b84b729d (MIT), byte-identical

integration/gqa_attention_volta_splitd.cu
  size    18,072 B   (Apache-2.0, first published here)
```

### 出问题怎么办

我**不懂代码也不懂英文**，没法回答技术问题——但**可以把你的问题转给 AI 帮我复现**。
开 Issue 时请附上：**你机器的卡型/显存、CUDA 版本、提示词深度（token 数）、解码 tok/s、接受率**，
尤其是**报错原文**（不要转述）。有这几项我这边能对上号。

---

## English

### What this is

Integrates **1CatAI's Split-D D256 FlashAttention kernel** (from [`fishlikeX/sm70-attn`](https://github.com/fishlikeX/sm70-attn))
into **NInfer's long-context prefill path on Tesla V100 (sm_70)**, with benchmark data measured on real hardware.

**The problem it solves** (probably why you are here): running a 27B-class model on a single V100 with a
150k–200k-token context makes **time to first token unbearable** — a full-window cold prefill measured
**10 min 35 s** here while compute sat mostly idle. The bottleneck is the attention kernel.
After the swap the same work takes **7 min 29 s**: **prefill throughput +36%…+41%**, with **no change to decode speed**.

In one line: **this optimizes prefill / TTFT, not tokens-per-second output.** If your pain is slow decoding,
this kernel will not help you (see the decode rows below).

### Authorship — who actually made this (please read)

- **I did not write this code.** I am **taskeee**; I **do not program and cannot read English**.
- Everything here — code, docs, benchmarks, scripts — was **prepared and published by an AI assistant
  (DeepSeek Harness) at my request**. So yes: **this content was posted by the user with AI assistance.**
- **The AI did not invent the code.** The kernel is the work of **1CatAI / `fishlikeX/sm70-attn`** (used byte-identical);
  the host engine is **Neroued/ninfer**. The AI did the **wiring, the measurements, and the writing**. Credit belongs upstream.
- All numbers come from **my single-V100 machine**. **Yours will differ** — measure your own.

### Provenance

| Component | Source | License | Handling here |
|---|---|---|---|
| **Attention kernel**<br>`kernel/fattn-sm70-d256-kernel.cuh` (49,003 B) | [`fishlikeX/sm70-attn`](https://github.com/fishlikeX/sm70-attn), commit `707cf247c1beb5e7c701f47fe6f828c5b84b729d` (2026-09-22) | **MIT** (Copyright (c) 2023-2026 The ggml authors) | **byte-identical** (SHA256 below) |
| **Host engine** | [`Neroued/ninfer`](https://github.com/Neroued/ninfer) (★2,485) | **Apache-2.0** | source not bundled; wiring documented |
| **V100 / sm_70 port** | my local tree, **not publicly available** (see below) | Apache-2.0 (inherited) | snippets only |
| **Wiring code**<br>`integration/gqa_attention_volta_splitd.cu` (18,072 B) | written on this machine, **first published here** | Apache-2.0 | full source |
| **Dependency closure** (**not bundled here**) | NVIDIA **CUTLASS/cute** (152 headers) + **flash-attention** (8 headers), upstream `ggml/src/ggml-cuda/sm70-vendor` | Apache-2.0 / BSD-3-Clause (`licenses/`) | copy command in "Wiring ②" |

**About the V100 port**: my NInfer tree is an **sm_70/Volta port** (its `CMakeLists.txt` states
*"Upstream NInfer is compiled only for sm_120a. This port adds sm_70 (Volta / Tesla V100) as a peer compile-time target."*).
**I could not find a public download for that port** — [`dollarwong/ninfer-v100`](https://github.com/dollarwong/ninfer-v100)
describes itself as a V100 port but contains **0 files**, and `Neroued/ninfer`'s `src/ops/launcher/` (52 files)
does **not** contain `gqa_attention_volta_flash.cu`. Therefore **this repo ships no apply-and-go patch**;
it gives the kernel source plus **exactly which files to touch and what to add**. Total change: **1 new file + 3 edits**.

### Measurements (single V100-SXM2-32GB)

Conditions: CUDA 12.8.93, driver 581.80, `CMAKE_CUDA_ARCHITECTURES=70`, KV `int8`, MTP `--draft-tokens 3`,
fixed content (`--greedy`), cold prefill (`cache 0%`), identical prompt, no profiling probes.
"old" = archived pre-swap binary (vendored llama.cpp MMA FA path); "new" = Split-D D256.

| Prompt tokens | Kernel | Prefill tok/s | TTFT | Decode tok/s | Draft accept |
|---:|---|---:|---:|---:|---:|
| 83,177 | old | 847.8 | 98.1 s | 39.09 | 46.3% |
| 83,177 | **new** | 848.1 | **98.1 s** | **42.61** | 52.1% |
| 196,320 | old | 437.1 | 448 s | 26.30 | 44.4% |
| 196,320 | **new** | **594.1** | **330 s** | **30.57** | 54.4% |
| 239,183 | old | 376.5 | **10 min 35 s** | 30.62 | 64.55% |
| 239,183 | **new** | **532.5** | **7 min 29 s** | 30.21 | 61.67% |

**How to read it (avoid the trap I fell into)**

1. **Prefill gets much faster**: **+36%** at 196k depth, **+41%** at 239k depth, TTFT **−3 min 06 s**.
2. **Decode is untouched by this kernel** — it only routes prefill. Decode variation above tracks
   **draft acceptance drift with content** (44%…65%), not kernel speed.
3. **Decode is governed by depth and content**: same machine, same new kernel —
   **42.6** tok/s at 85k, **30.6** at 196k, **30.2** at 239k. Deeper context is slower because of
   KV reads (~7.9 GB at full window); no kernel changes that.
4. **Always quote depth *and* acceptance with any tok/s number**, or the numbers are not comparable.

**Real-workload reference** (not synthetic filler): real agent turns measured **42.6–54.6 tok/s** at 75k–79k,
40.9 at 132k, 34.6 at 164k, 36.2 at 189k, 28.5 at 190k, 34.1 at 192k.

**A practical ceiling**: clients that auto-compact at 80% context (e.g. DSH) make `--max-context 245000`
effectively **≈196K usable**. Do not benchmark or judge your experience at 240k — you will never reach it.

Raw data: `bench/*.json`. Scripts: `bench/ninfer_bench.py`, `bench/probe_live.py`, `bench/run_greedy_ab.sh`, `bench/run_depth_control.sh`.

### Wiring (1 new file + 3 edits)

Exact original snippets: `integration/snippet-flash-launcher.txt`, `integration/snippet-cmake.txt`
(extracted verbatim from the working tree).

1. **Add** `src/ops/launcher/gqa_attention_volta_splitd.cu` (in `integration/`): Q `bf16→f16` staging
   (same normalized Hadamard as the old path when KV is `int8`), pad to a multiple of 64, call the Split-D
   kernel, write back `f32→bf16`; staging via one `cudaMalloc` (~12.6 + 25.2 MiB).
2. **Vendor** the kernel and its closure into `third_party/llama_cpp_sm70_d256/`
   (`fattn-sm70-d256-kernel.cuh` + `sm70-vendor/`). The closure is **not bundled here**
   (162 third-party headers / 14 MB) but exists verbatim upstream:

```bash
git clone --depth 1 https://github.com/fishlikeX/sm70-attn.git /tmp/sm70-attn
mkdir -p third_party/llama_cpp_sm70_d256
cp -r /tmp/sm70-attn/ggml/src/ggml-cuda/sm70-vendor third_party/llama_cpp_sm70_d256/
cp kernel/fattn-sm70-d256-kernel.cuh third_party/llama_cpp_sm70_d256/
```

   Provenance (from the upstream `sm70-vendor/README.md`): `cute/` + `cutlass/` from
   [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) commit `62750a2b…` (Apache-2.0);
   `flash/` (8 headers) from [zhinianqin/flash-attention-v100](https://github.com/zhinianqin/flash-attention-v100)
   commit `c2eda5e6…` (BSD-3-Clause).
3. **Hook** `src/ops/launcher/gqa_attention_volta_flash.cu` (lines 38–48 declarations, 630–660 call site here):
   declare `volta_splitd_requested()` / `volta_splitd_block<Geometry>(…)`, and inside the Q-block loop call it
   before the vendored kernel — on success `continue`, on `false` fall back to the old kernel automatically.
4. **Build** in `src/CMakeLists.txt` (lines 346 and 357–360): add the new `.cu` to the sm_70 source list and
   scope `third_party/llama_cpp_sm70_d256` **and** `.../sm70-vendor` as include dirs **for that file only**
   (the closure includes `cute/...` directly, so the parent directory alone is not enough).

### Operating notes (learned the hard way)

- **Rollback**: `NINFER_VOLTA_SPLITD=0` forces the old kernel (both paths live in one binary). **Default is on.**
- **Verify the swap, not the copy**: overwriting a running engine's binary fails with `Text file busy`
  while the **old kernel keeps serving**. Use the **prefill fingerprint**: at 196k depth, cold prefill
  **437 = old / 594 = new** (at 85k both are ~848, i.e. indistinguishable).
- **Restarting the engine clears the KV prefix cache** — the next turn pays a full cold prefill.
- **Run the contract test** (`ninfer_softmax_attention_test`, with `CUDA_VISIBLE_DEVICES=1` pointing at the V100).
  It is the only thing that catches **layout/stride** bugs — **speed numbers look fine when they are wrong**.
- **Stride trap**: the kernel addresses by **caller-supplied strides**. Upstream's launcher passes head-major
  (`q_head_stride = q_pad * D`); my staging is row-major, so **row-major strides must be passed**, otherwise
  rows/heads get crossed (visible only as a 3–11% deviation in the `T=66 keys=129` test case).
- **Causal masking** is computed inside the kernel from `(kv_len, kv_offset)` — no outer mask needed.
- **Nothing that synchronizes or sets attributes may run inside CUDA graph capture**
  (`cudaFuncSetAttribute`, any D2H copy): the new path detects capture and falls back automatically.
- **Do not enable profiling probes while benchmarking** — they cost ~2.5% prefill on their own.

### Dead ends (save your time)

- **Retuning the vendored flash config table**: `ncols 32→64` is 18% slower (smem 53 KB → 1 CTA/SM);
  `Q_in_reg=true` is **5.5× slower**; `nbatch_fa=16` does not compile (`static_assert(%32)`).
- **DFlash2 as an explanation for long-context slowdown**: it is **not implemented** in this engine's V100 build
  (`src/ops/dflash2_sm70_stub.cu` is all throwing stubs; upstream kernels need sm_80+) — it cannot be your cause.
- **Parallel GDN kernel**: GDN is only **8.2%** of prefill here; halving it saves ~4 s.
- **Draft depth K=4**: +13% at 100k depth but only +1% at 160k — contradictory across depths, so K=3 stays.

### License

Kernel **MIT** (`fishlikeX/sm70-attn`), wiring code **Apache-2.0** (derived from NInfer's source form),
bundled closure **BSD-3-Clause** (NVIDIA CUTLASS/cute, flash-attention). See `licenses/`.
No NInfer engine source is redistributed here; get it from [`Neroued/ninfer`](https://github.com/Neroued/ninfer).

### Checksums

```
kernel/fattn-sm70-d256-kernel.cuh
  SHA256  f3a45709ed2c88bd466912e7c4e2855bec6226ca4fda918d17855fbe2cb1f71e   (49,003 B)
  source  fishlikeX/sm70-attn @ 707cf247c1beb5e7c701f47fe6f828c5b84b729d (MIT), byte-identical

integration/gqa_attention_volta_splitd.cu
  18,072 B   (Apache-2.0, first published here)
```

### Issues

I **cannot read code or English**, so I answer technical questions by having an AI reproduce them.
When opening an issue, please include: **GPU model and VRAM, CUDA version, prompt depth in tokens,
decode tok/s, draft acceptance**, and above all the **verbatim error message** (not a paraphrase).

> Companion repository: **[`ninfer-v100-sm70-decode`](https://github.com/taskeee/ninfer-v100-sm70-decode)** - the DECODE side (tpx v2 int8 kernel + Flo5k5 sm70 commits), the context-cache/scheduler settings, and a measured 53-request real agent load run with the raw engine logs.
