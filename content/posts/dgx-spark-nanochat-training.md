---
title: "Training an LLM from Scratch on My NVIDIA DGX Spark"
date: 2026-02-15
description: "My first experience training a GPT language model end-to-end on the NVIDIA DGX Spark desktop supercomputer using Karpathy's nanochat."
tags: ["dgx-spark", "nvidia", "llm", "nanochat", "deep-learning", "blackwell", "gpu"]
cover:
  image: "/images/dgx-spark-nanochat/metrics_overview.png"
  alt: "DGX Spark Training Metrics Overview"
  caption: "286M parameters, 1.16B tokens, 22.4% MFU, 12h 56m total"
  relative: false
ShowToc: true
TocOpen: true
---

I got an NVIDIA DGX Spark recently, a desktop supercomputer with a Grace Blackwell GPU, and the first thing I wanted to do was obvious: **train an LLM from scratch to test it**.

Not fine-tune someone else's model. Not run inference on a downloaded checkpoint. Train the whole thing — tokenizer, pretraining, supervised fine-tuning, evaluation — from raw text to a chatbot that can answer questions, write code, and count the letters in "strawberry."

Here's how it went.

## The Hardware

| Spec | Details |
|------|---------|
| **GPU** | 1× NVIDIA GB10 (Blackwell architecture) |
| **Memory** | 128 GB unified CPU+GPU |
| **CPU** | 20-core ARM Grace |
| **CUDA Capability** | 12.1 (sm_121a) |
| **Peak BF16** | ~209 TFLOPS |
| **Form Factor** | Desktop (no rack needed!) |

The unified memory is the killer feature. 128 GB shared between CPU and GPU means you can fit models that would never work on a regular consumer GPU. It's a single GPU with a brand-new chip architecture, so software support is a work in progress.

## The Plan: nanochat

I chose [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy, it's the simplest end-to-end LLM training harness that exists. One repo, one complexity dial (`--depth`), and it covers everything: tokenization → pretraining → SFT → evaluation → chat.

## Getting It Running

The GB10 GPU is very new that most of the ML ecosystem doesn't fully support it yet. Here's what I had to fix:

### Problem 1: CUDA 13.0

The GB10's compute capability is `sm_121a`, a chip name that the default `ptxas` bundled with Triton 3.5.0 doesn't recognize. Triton ships CUDA 12.8's assembler, but `sm_121a` needs CUDA 13.0.

**Fix**: Install the CUDA 13.0.2 toolkit and point Triton at it:

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
```

### Problem 2: Flash Attention

nanochat uses Flash Attention 3, which is compiled only for Hopper GPUs (sm90). The GB10 is Blackwell (sm100/sm121a). So, FA3 kernels simply won't load.

**Fix**: Already handled! A [merged PR](https://github.com/karpathy/nanochat/pull/475) added an SDPA (Scaled Dot-Product Attention) fallback for non-Hopper GPUs. The detection logic checks if the GPU's compute major version is 9 (Hopper). if not, it falls back to PyTorch's native SDPA.

### Problem 3: PyTorch Compatibility

PyTorch 2.9+ with CUDA 13.0 is required. The `pyproject.toml` needed updating to point at the `pytorch-cu130` index instead of `pytorch-cu128`.

After all that, I had a working training script: [`runs/speedrun_dgx_spark.sh`](https://github.com/lakshmankolasani/nanochat/blob/master/runs/speedrun_dgx_spark.sh)

## The Model

nanochat's depth parameter auto-configures everything. I chose `depth=12`:

![Model Architecture](/images/dgx-spark-nanochat/architecture.png)

| Parameter | Value |
|-----------|-------|
| **Depth** | 12 (GPT-1 scale) |
| **Model dimension** | 768 |
| **Attention heads** | 6 |
| **Parameters** | 286M |
| **Sequence length** | 1,024 |
| **Vocabulary** | 32,768 (custom BPE) |

Key architectural choices in nanochat:
- **Squared ReLU** activation instead of GELU
- **Rotary Position Embeddings** (RoPE) instead of learned positions
- **QK-Norm** for training stability
- **Value Embeddings** (ResFormer) on alternating layers
- **Logit softcapping** at ±15 to prevent numerical instability
- **No Flash Attention** on DGX Spark — falls back to PyTorch SDPA

## Phase 1: Tokenizer (33 seconds)

First, train a BPE tokenizer from scratch on 2 billion characters of FineWeb-Edu (educational web text):

- **Vocabulary size**: 32,768 ($2^{15}$)
- **9 special tokens**: for chat formatting (`<|bos|>`, `<|user_start|>`, `<|assistant_start|>`, etc.)
- **Training time**: 33 seconds

How does it compare to GPT-2 and GPT-4's tokenizers?

![Tokenizer Comparison](/images/dgx-spark-nanochat/tokenizer_comparison.png)

Our 32K-vocab tokenizer matches GPT-2 (50K vocab) on English text and slightly beats it on scientific content. It's significantly behind GPT-4's 100K-vocab tokenizer on code and non-English text, but that's expected with a vocabulary 3× smaller.

The fun metric here is **compression ratio** (bytes per token). Higher means each token captures more information. On FineWeb-Edu training data, we get **4.72 bytes/token** which is slightly better than GPT-2's 4.67.

## Phase 2: Pretraining (5 hours 16 minutes)

This is the main thing I wanted to test. Training a GPT from scratch on 1.16 billion tokens of web text.

### The Setup

| Config | Value |
|--------|-------|
| **Training tokens** | 1.156 billion |
| **Batch size** | 524,288 tokens |
| **Device batch** | 64 sequences × 1,024 tokens |
| **Gradient accumulation** | 8 steps |
| **Optimizer** | Muon (matrices) + AdamW (embeddings) |
| **Precision** | BF16 |
| **Iterations** | 2,205 |

A few things about the optimizer:

**Muon** is used for all 2D weight matrices.

**AdamW** is used for embeddings and scalar parameters with different learning rates per group (embeddings get 0.3, the LM head gets 0.004).

The **learning rate schedule** is: no warmup → constant for first 50% → linear decay to 0 over the last 50%.

### 22.4% MFU

| Metric | Value |
|--------|-------|
| **MFU** | 22.43% |
| **Throughput** | ~30,800 tok/sec |
| **Peak memory** | 31.5 GB / 121.7 GB (26%) |
| **Total FLOPS** | 8.95 × 10¹⁷ |

The memory usage was surprisingly modest, only 26% of the available 128 GB. There's clearly headroom for larger models (depth 16 or even 20).

### Final Metrics

- **Validation BPB**: 0.906 (bits per byte — tokenizer-independent loss metric)
- **Training time**: 316.5 minutes (~5.3 hours)
- **Estimated cost**: ~$10.55 (@ $2/hr)

## Phase 3: Base Model Evaluation

After pretraining, I evaluated the base model on the **DCLM CORE benchmark**, 21 NLP tasks covering reading comprehension, common sense reasoning, world knowledge, and more.

**CORE Score: 0.128** (0 = random, 1 = perfect)

![CORE Benchmark Results](/images/dgx-spark-nanochat/core_benchmark.png)

The model is clearly above random on most tasks — strongest on CS Algorithms (0.41), ARC-Easy (0.37), and Wikidata QA (0.31). It's *below* random on BoolQ (-0.29), meaning it has a systematic prediction bias on yes/no questions.

### What the Base Model Can Do

Some sample generations from greedy decoding:

> **"The capital of France is"** → "Paris. It is the capital of the country. It is the largest city in"  ✅
>
> **"The planets of the solar system are:"** → "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune," ✅
>
> **"The chemical symbol of gold is"** → "the symbol of the element 'gold' in the periodic table" ❌ (should be "Au")
>
> **"If 5\*x + 3 = 13, then x is"** → "5\*x + 3 = 13" ❌ (just repeats the equation)

The base model has absorbed factual knowledge from the web data but can't reason or solve math. That's what SFT is for.

## Phase 4: Supervised Fine-Tuning (4 hours)

SFT transforms the base model (which just predicts next tokens) into a **chat model** (which follows instructions).

### Training Data (856K conversations)

| Dataset | Size | Purpose |
|---------|------|---------|
| SmolTalk | 460K | General conversation |
| SimpleSpelling | 200K | Spelling tasks |
| MMLU auxiliary | 100K | Multiple-choice reasoning |
| SpellingBee | 80K | Letter counting |
| GSM8K × 2 | 16K | Math with calculator |
| Identity | 2K | "Who are you?" |

The key insight: SFT uses **selective loss masking**. The model only learns to predict *assistant responses* — user messages, system prompts, and tool outputs are excluded from the loss. This teaches the model to *respond* to instructions, not to *generate* instructions.

### Training Details

- **851 iterations** (one full epoch through the data)
- **Weight decay: 0** (prevents catastrophic forgetting of pretrained knowledge)
- **LR schedule**: Constant for 80%, then linear decay to 0
- **Final validation BPB**: 0.443 (much lower than pretraining, the task is easier when you only predict structured assistant responses)

## Phase 5: Chat Model Results

How does the SFT model perform on real benchmarks?

![Chat Evaluation Results](/images/dgx-spark-nanochat/chat_eval.png)

| Task | Accuracy | Random Baseline | Verdict |
|------|----------|----------------|---------|
| **SpellingBee** | **98.05%** | 0% | 🔥 Nearly perfect |
| ARC-Easy | 32.83% | 25% | Slightly above random |
| MMLU | 30.32% | 25% | Slightly above random |
| ARC-Challenge | 26.71% | 25% | Barely above random |
| HumanEval | 7.93% | 0% | Can write some Python |
| GSM8K | 1.36% | 0% | Math is hard |

**ChatCORE Score: 0.212**

The standout result is **SpellingBee at 98%**. "How many 'r's in 'strawberry'?" The model nails this because it was trained on 280K spelling examples. When you train on a specific task, small models can get surprisingly good at it.

The multiple-choice benchmarks hover near the 25% random baseline, which is honestly expected for a 286M-parameter model. For context, GPT-2 (1.5B params, trained on much more data) scores around 0.25 CORE. We're in the right ballpark.

One cool feature: on GSM8K, the model can use a **calculator tool**. It generates `<|python_start|>2+3<|python_end|>`, the inference engine evaluates it, and injects `<|output_start|>5<|output_end|>` back. A tiny model doing tool use!

## The Full Picture

![Time Breakdown](/images/dgx-spark-nanochat/time_breakdown.png)

### Final Summary

| Metric | BASE | SFT |
|--------|------|-----|
| **CORE** | 0.128 | — |
| **ChatCORE** | — | 0.212 |
| Val BPB | 0.906 | 0.443 |
| MFU | 22.4% | 2.4% |
| Training Time | 5h 16m | 4h 0m |

**Total wall clock: 12 hours 56 minutes**, from raw text data to a working chatbot, all on a single desktop machine.

## Reproduce It To Test DGX Spark in less than a Day

Everything is in my fork: [lakshmankolasani/nanochat](https://github.com/lakshmankolasani/nanochat)

```bash
# On a DGX Spark with CUDA 13.0 installed:
git clone https://github.com/lakshmankolasani/nanochat.git
cd nanochat
bash runs/speedrun_dgx_spark.sh
```

The script handles everything: venv setup, dataset download, tokenizer training, pretraining, evaluation, SFT, chat eval, and report generation. Total time: ~13 hours.

## What's Next

- **Deeper models**: With only 26% memory used at depth=12, there's room for depth=16 or even depth=20
- **RL fine-tuning**: nanochat includes `chat_rl.py` for reinforcement learning — I haven't run it yet
- **Flash Attention**: When FA3 or FA2 gets Blackwell support, MFU should improve significantly
- **Multi-epoch training**: The Power Lines scaling laws suggest more data epochs could help at this model size

---

*This post was generated from actual training results on my NVIDIA DGX Spark for testing purpose. The full training code with modifications is available in [my nanochat fork](https://github.com/karpathy/nanochat/compare/master...lakshmankolasani:nanochat:master). This blog was prepared with the help of AI.*
