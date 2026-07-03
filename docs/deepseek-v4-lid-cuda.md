# DeepSeek V4 Flash: CUDA Lightning Indexer Kernel

This branch adds a CUDA kernel for DeepSeek V3.2/V4's lightning indexer
(the DeepSeek Sparse Attention scoring step), fixing a compute-buffer
blowup that made long-context prefill impossible when offloading MoE
experts to CPU/GPU together.

## Build instructions

### Getting the source

If you don't already have a clone of llama.cpp, `--single-branch` avoids
pulling the rest of this fork's branch history (which mirrors a lot of
upstream contributor branches, so it can be large):

```
git clone --branch deepseek-lid-cuda --single-branch https://github.com/spencer-zaid/llama.cpp.git
cd llama.cpp
```

If you already have a llama.cpp clone, you can add this fork as a remote and
fetch just this branch instead of cloning a second copy:

```
git remote add spencer-zaid https://github.com/spencer-zaid/llama.cpp.git
git fetch spencer-zaid deepseek-lid-cuda
git checkout deepseek-lid-cuda
```

### Compiling

Tested on Windows with CUDA 13.3 and Visual Studio Build Tools. CUDA
13.3 has no VS-integration props, so the VS CMake generator doesn't
work - use Ninja with the VS environment imported manually:

```powershell
# import the VS build environment (adjust path to your VS install)
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2]) }
}

cmake -G Ninja -B build -S . -DCMAKE_BUILD_TYPE=Release `
  -DGGML_CUDA=ON -DGGML_CCACHE=OFF -DGGML_NATIVE=ON `
  -DCMAKE_CUDA_ARCHITECTURES=<your-arch> `
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl

cmake --build build --target llama-cli llama-bench llama-server -j
```

Replace `<your-arch>` with your GPU's CUDA compute capability: `86` for
RTX 30-series, `89` for RTX 40-series, `90` for Hopper/H100, `120` for
RTX 50-series (gets auto-adjusted to `120a` by ggml's CMake).

If your nvcc invocation goes through `sccache` (or similar), it will
likely fail at the PTX stage with `fatbinary fatal: Could not open input
file` - that's why `GGML_CCACHE=OFF` is required above.

Linux should work in principle (the CUDA kernel and CMake changes are
not Windows-specific) but has not been tested by me.

## Usage

The routed experts (82.7 GiB of the 90.9 GiB file) are the part you
choose to split between GPU and CPU with `-ot` (override-tensor). `N`
below means "how many of the 43 layers' expert weights are placed on
GPU" - the presets keep the cheapest (2-bit) layers on GPU and always
leave layers 37-42 (the larger Q4_K ones) on CPU.

Example for N=8 (layers 0-7 on GPU, 8-42 on CPU):

```
-ngl 99 -ot "blk\.([89]|[1-3][0-9]|4[0-2])\.ffn_(gate|up|down)_exps=CPU"
```

For a different N, replace the pattern with one that matches blocks
`N..42`.

Set `GGML_CUDA_NO_PINNED=1` in the environment when using `--no-mmap`
with a large amount of CPU-resident expert weight (this model needs
~69-73 GiB of it). CUDA pins host memory by default for anything that
might touch the GPU, but these tensors are computed entirely on CPU, so
pinning them is pure overhead - and can fail to allocate if free RAM is
tight. Disabling it costs nothing and makes resident loads more robust.

Full example command:

```
llama-cli.exe -m <path-to-model>.gguf ^
  -ngl 99 -ot "blk\.([89]|[1-3][0-9]|4[0-2])\.ffn_(gate|up|down)_exps=CPU" ^
  -fa on --no-mmap --jinja ^
  -t <threads> -c 262144 -ub 2048 -b 2048 ^
  -p "your prompt"
```

(with `GGML_CUDA_NO_PINNED=1` set in the environment first)

## Credit

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) - base project
- PR [#24162](https://github.com/ggml-org/llama.cpp/pull/24162) - DeepSeek V4 architecture support (merged)
- [fairydreaming](https://github.com/fairydreaming) - author of PR
  [#24231](https://github.com/ggml-org/llama.cpp/pull/24231) (`deepseek-lid`), which added
  `GGML_OP_LIGHTNING_INDEXER` (the op definition and a CPU kernel),
  open/unmerged at time of writing. This branch is based on that PR and
  adds the missing CUDA kernel plus the model-graph wiring to actually
  use it.
- The mixed-precision quant used for all testing below was produced by
  antirez.

Not included: PR [#25202](https://github.com/ggml-org/llama.cpp/pull/25202)
(quantized KV-cache correctness fix). This branch is **F16 KV cache
only** - quantized KV (`-ctk q8_0 -ctv q8_0`) is known broken (garbage
output) on this codebase without that fix, and it wasn't part of this work.

## The problem

DeepSeek's lightning indexer scores every cached position against every
token in the current batch before picking the top-k most relevant ones.
The unfused implementation materializes a `[ubatch x context_length]`
intermediate tensor to do this. At long context that tensor is enormous:

- 256K context, ubatch 2048: ~67 GiB
- 1M context, ubatch 2048: ~256 GiB (extrapolated - this config was never
  actually run, since the compute buffer alone exceeds what any GPU
  today has)

Both exceed any consumer GPU's VRAM, so the only way to run long context
at all was to shrink `ubatch` down to ~128, which tanks prefill speed
(56 t/s at 256K) - or just not run 1M context, since even ubatch 32 still
overflows.

## The fix

A fused CUDA kernel for `GGML_OP_LIGHTNING_INDEXER`: one warp computes
each output score, with the per-head dot product split across the warp's
32 lanes and finished with a shuffle reduction. Query rows and indexer
weights are staged in shared memory once per block and reused across all
KV positions in that block; nothing per-(query, kv-position) pair is
ever materialized in global memory. This collapses the compute buffer
by roughly two orders of magnitude and makes large ubatch affordable at
any context length.

The model graph in `src/models/deepseek4.cpp` was also updated to call
the fused op instead of the original permute/mul_mat/relu/mul/sum_rows
sequence.

**Correctness:** verified token-identical greedy output against the
unfused path at short context (before real top-k filtering kicks in),
a needle-in-haystack retrieval test at long context, and KL-divergence/
perplexity against the naive path at real context depths - see Results
for the full picture, including a small, understood, and expected
source of non-bit-identical output at the top-k selection boundary.

## Test hardware

RTX 5090 (32 GB VRAM), Ryzen 9 9950X3D, 96 GB DDR5-6200 (dual channel).
All numbers below are specific to this hardware - see "Tuning for your
own GPU" if your VRAM differs.

## Model tested

DeepSeek V4 Flash - 284B total parameters, 43 layers, 256 routed experts
(6 active per token) + 1 always-active shared expert, MLA (multi-head
latent attention) + DeepSeek Sparse Attention (lightning indexer,
top-k=512), native 1M context.

Custom mixed-precision GGUF quant (~90.9 GiB total, ~2.06 bpw effective):

| Tensor group                            | Quant   | ~bits/weight |
|------------------------------------------|---------|--------------|
| Routed expert gate/up (most layers)      | IQ2_XXS | 2.06         |
| Routed expert down (most layers)         | Q2_K    | 2.6          |
| ALL expert tensors, layers 37-42         | Q4_K    | 4.5          |
| Attention, shared expert, output head    | Q8_0    | 8.5          |

Layers 37-42 are elevated to Q4_K because the imatrix calibration showed
the final layers are the most sensitive to quantization error (they feed
the output logits most directly).

## Results

### Before / after (256K context)

| Metric                              | Before (unfused) | After (this kernel)  |
|---------------------------------------|-------------------|------------------------|
| Compute buffer, ubatch=2048           | ~67 GiB (OOM)      | 3.2 GiB                |
| Prefill                               | 56 t/s (forced ubatch=128) | 204 t/s (ubatch=2048) |
| Decode                                | ~16 t/s            | ~15 t/s (unchanged - decode is DDR5-bandwidth bound, not indexer bound) |
| 1M context                            | impossible (~256 GiB, extrapolated - never actually ran) | works (~31.2 GiB VRAM) |

### Validated presets

Measured with a real ~100K-token document (not a short synthetic
prompt - see "Prompt length matters" below for why that distinction
matters):

| Preset | Context | GPU expert layers | ubatch | Prefill  | Decode   | Peak VRAM | CPU RAM (resident experts) |
|--------|---------|--------------------|--------|----------|----------|-----------|------------------------------|
| 256K   | 262144  | 8                  | 2048   | ~263 t/s | ~14.0 t/s | ~28.9 GiB | ~69.2 GiB                    |
| 512K   | 524288  | 6                  | 2048   | 256 t/s  | 13.7 t/s | ~28.4 GiB | ~72.6 GiB                    |
| 1M     | 1048576 | 6                  | 768    | 159 t/s  | 13.7 t/s | ~31.2 GiB | ~72.6 GiB                    |

("GPU expert layers" = how many of the 43 layers' routed-expert weights
are placed on GPU rather than CPU; see Usage below.)

#### Prompt length matters

A short (~2900 token) synthetic prompt at these same configs measures
noticeably lower prefill: 204 t/s (256K), 201 t/s (512K), 151 t/s (1M).
Fixed per-invocation overhead is amortized over far fewer tokens, so
short prompts understate real throughput. The "Before/after" table above
intentionally uses the short-prompt numbers on both sides for an
apples-to-apples comparison of the kernel fix itself; the preset table
above uses the more realistic long-document numbers.

#### Decode also depends on how full the context already is

Decode is a bit slower with a deep KV cache than a shallow one, at the
same N: 256K/N8 goes from 15.0 t/s (shallow) to ~14.0 t/s (~100K tokens
deep); 512K/N6 and 1M/N6 both go from 14.7 t/s to 13.7 t/s. This is
expected: the indexer scores the current query against the *entire*
cached history on every decode step too, not just during prefill, so
decode cost grows with context depth like any attention mechanism - it's
not specific to this kernel.

### Correctness: needle-in-haystack retrieval

A unique, unguessable fact was planted at three depths in a ~100K-token
document (256K preset, temp=0), then asked for at the end of the prompt:

| Depth                  | Result  | Prefill  | Decode   |
|--------------------------|---------|----------|----------|
| 10%                      | correct | 263.9 t/s | 14.1 t/s |
| 50% (lost-in-the-middle) | correct | 264.7 t/s | 14.2 t/s |
| 90%                      | correct | 262.6 t/s | 13.7 t/s |

The 50% (hardest) depth was also spot-checked at the 512K and 1M
presets, both correct: 512K - 256.0 t/s prefill, 13.7 t/s decode; 1M -
158.6 t/s prefill, 13.7 t/s decode.

### Correctness: KL-divergence / perplexity

Needle-in-haystack proves retrieval correctness, not full-distribution
numerical fidelity - it only checks whether one planted fact survives,
not whether every token's output distribution matches the unfused path.
Ran `llama-perplexity --kl-divergence` (llama.cpp's standard tool for
this, also used to validate quantizations) comparing this kernel against
the naive/unfused path on wikitext-2, at two context depths:

| Context | Tokens | Mean KLD | Median KLD | Same top-token | Max single-token Δp |
|---------|--------|----------|------------|-----------------|----------------------|
| 8,192   | 16,384 | 0.0108   | 0.0026     | 96.12%           | 31.5%                 |
| 65,536  | 65,536 | 0.0092   | 0.0029     | 96.33%           | 72.8%                 |

This is **not bit-identical** to the naive path, and it's worth being
upfront about why. The indexer does a hard top-k=512 selection over all
candidate KV positions. Naive computes the selection score via a chain
of separate ops (permute -> mul_mat -> relu -> mul -> sum_rows); this
kernel computes the same value in one warp-parallel reduction. Both are
correct, but floating-point addition isn't associative, so summing in a
different order produces a tiny (~0.01-0.1% relative) difference in the
score - invisible almost everywhere, except for the handful of
candidates sitting right at the top-512 cutoff. There, a margin smaller
than that rounding noise decides who's rank 512 and who's rank 513, so
the two implementations occasionally disagree on that one boundary case.

Confirmed this directly by dumping the raw per-position indexer scores
and the selected top-512 index sets from both builds at a real
(>512-candidate) context. Of 353 query positions, 67 (19%) had their
selection affected - and **every single one was a clean 1-for-1 index
swap**, never a larger or systematic difference. In each case the two
swapped candidates' scores were within 0.0001-0.001 of each other in
both builds independently, confirming a genuine near-tie rather than a
logic error - a real bug (wrong masking, an off-by-one, a missed
Hadamard rotation step) would show up as large or structurally
consistent differences, not scattered single-index coin-flips on
already near-tied candidates.

A single-position swap at one layer is a small perturbation on its own,
but it happens independently at every layer that runs the indexer, so
the effect compounds - and occasionally the swapped-in/out position
happens to matter a lot for that specific prediction, producing the
larger Δp outliers in the table above. This is the same category of
thing as llama.cpp's flash-attention kernels producing slightly
different logits than naive attention: expected, precedented, and not
fixable by further kernel debugging, since both code paths are
mathematically correct.

### Throughput vs. how full the context already is (256K preset)

Prefill speed for the next 2048 tokens, measured at increasing KV depth
within a single 256K context - i.e. what happens to prefill speed as a
real conversation fills up its context window:

| KV depth (tokens)     | Prefill of next 2048 tokens |
|--------------------------|--------------------------------|
| 16,384                    | 317 t/s                         |
| 131,072                    | 151 t/s                         |
| 253,952 (~full 256K)       | 94.5 t/s                        |

This is a graceful decline, not a cliff - and no GPU timeout/hang was
observed even at the deepest, most expensive point.

## Tuning for your own GPU

Peak VRAM roughly follows:

```
peak_MiB ~= 7384 (non-expert weights) + ~2500-3000 (OS/driver overhead)
          + KV(context) + compute_buffer(context, ubatch)
          + N * 1728  (each GPU-resident expert layer)
```

Measured compute buffer sizes (context, ubatch -> buffer):

| Context | ubatch | Compute buffer |
|---------|--------|------------------|
| 256K    | 512    | 2.17 GiB          |
| 256K    | 2048   | 3.23 GiB          |
| 512K    | 2048   | 5.18 GiB          |
| 1M      | 512    | 2.58 GiB          |
| 1M      | 768    | 3.75 GiB          |
| 1M      | 2048   | 9.31 GiB          |

Measured KV cache sizes (F16): 256K ~1.75 GiB, 512K ~3.46 GiB, 1M ~6.9 GiB.

If you're short on VRAM at your target context: reduce ubatch first
(cheap, only costs prefill speed), then reduce N (costs decode speed,
via more CPU-DDR bandwidth pressure).

## Known limitations

- Not bit-identical to the naive/unfused path at real context depths
  (~4% of tokens pick a different top token in KL-divergence testing).
  Confirmed via direct score inspection to be floating-point rounding
  noise at the top-k=512 selection boundary (different reduction order
  between this kernel and naive's multi-op chain), not a logic bug -
  see Correctness: KL-divergence / perplexity.
- F16 KV cache only - quantized KV is broken upstream without PR #25202,
  which isn't included here. Cache is pretty small on deepseek v4 anyways
- Only tested on a single RTX 5090. Other GPUs/VRAM sizes will need their
  own N/ubatch tuning using the formula above.
- TDR (driver timeout) safety was tested only on this Windows version,
  driver, and GPU. An early version of this kernel (one thread per
  output, no parallel reduction) did cause a GPU timeout/screen-blank
  under load - the current warp-parallel kernel was specifically
  rewritten to fix that and has been stress-tested at full-context depth
  without issue, but if you see driver resets on very different
  hardware, please open an issue.
- No prebuilt binaries. Build from source with `-DGGML_NATIVE=ON` (as
  above) targets your own CPU's instruction set.
