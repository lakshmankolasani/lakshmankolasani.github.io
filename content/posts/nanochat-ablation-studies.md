---
title: "15 Experiments in 4 Days on a Single GPU"
date: 2026-03-08
description: "15 ablation experiments across 4 groups on a single NVIDIA GB10, testing two experimental sparsity features I added to nanochat: Dynamic Dimensional Layers and Dynamic Dimension Blocks."
tags: ["dgx-spark", "nvidia", "llm", "nanochat", "deep-learning", "ablation", "gpu", "blackwell"]
cover:
  image: "/images/dgx-spark-nanochat/architecture.png"
  alt: "nanochat transformer architecture"
  caption: "15 ablation experiments, 13.7B tokens, ~99 hours on a single GB10"
  relative: false
ShowToc: true
TocOpen: true
---

After [training an LLM from scratch](/posts/dgx-spark-nanochat-training/) on my DGX Spark, I wanted to go deeper. I added two experimental architectural features to nanochat, **DDL** and **DDB**, that haven't been systematically tested. So I built an ablation framework and ran 15 experiments over 4 days on a single GB10.

This post covers what I found.

## What Are DDL and DDB?

Both are experimental features I built on top of [nanochat](https://github.com/karpathy/nanochat) that try to make transformers more efficient by letting the model adapt its own capacity per-token and per-layer.

### DDL (Dynamic Dimensional Layers)

Standard transformers process every residual stream dimension equally for every token. DDL adds a learned gate per block:

```python
mask = sigmoid(Linear(norm(x)))   # (B, T, D) values in (0, 1)
x = x * mask                      # element-wise gating
loss += lambda * mean(mask*(1-mask))  # push gates toward 0 or 1
```

Simple tokens (common words, punctuation) activate fewer dimensions. Complex tokens use more capacity. The entropy regularization term (controlled by `lambda`) pushes gates toward binary decisions.

The key property: **zero inference overhead**. The extra Linear per block is fused by `torch.compile` into the existing computation graph.

### DDB (Dynamic Dimension Blocks)

Instead of every layer having the same width (e.g., 768), DDB lets each layer operate at a different dimension:

```
Layer 0:  768 dims (6 heads)
Layer 1:  768 dims (6 heads)
Layer 2:  1024 dims (8 heads)
Layer 3:  1024 dims (8 heads)
...
Layer 11: 768 dims (6 heads)
```

Projection layers handle the dimension changes between blocks. Attention heads scale proportionally (`num_heads = dim / 128`) to keep `head_dim=128` fixed for hardware efficiency.

## Experiment Setup

All 15 experiments ran on my DGX Spark, a single GB10 with 128 GB unified memory.

| Setting | Value |
|---------|-------|
| GPU | 1x GB10 (Blackwell, SM 12.1) |
| Attention | SDPA (no Flash Attention 3) |
| Optimizer | Muon (weight matrices) + AdamW (embeddings) |
| Base model | depth=12, 768 dim, 6 heads |
| Throughput | ~52-62K tok/sec (~22% BF16 MFU) |

I used two training budgets:
- **ratio=10.5** (10.5 tokens per parameter) for configs that overlap with prior experiments, so I could compare directly
- **ratio=3.0** for new/exploratory configs, as a cheaper screening pass

## DDL Lambda Sweep

**Question**: What entropy regularization weight works best for DDL?

| Config | Lambda | Budget | Tokens | val_bpb |
|--------|--------|--------|--------|---------|
| baseline (no DDL) | -- | 10.5 | 1.16B | 0.9073 |
| ddl_lambda_0.001 | 0.001 | 10.5 | 1.23B | 0.9068 |
| ddl_lambda_0.005 | 0.005 | 3.0 | 351M | 0.9835 |
| **ddl_lambda_0.01** | **0.01** | **10.5** | **1.23B** | **0.9036** |

**Lambda=0.01 wins**, 0.4% better than baseline. This replicates the original finding from the DDL report. Weaker regularization (0.001) barely helps, meaning the model needs strong pressure to push gates toward binary decisions.

Both DDL and baseline run at identical throughput (~52,900 tok/sec). The gating adds zero measurable overhead.

## DDB Dimension Patterns

**Question**: Which block-dimension pattern gives the best BPB?

| Config | Pattern | Budget | Tokens | val_bpb |
|--------|---------|--------|--------|---------|
| baseline | 768 x 12 | 10.5 | 1.16B | 0.9073 |
| hourglass_1024 | narrow-wide-narrow | 3.0 | 428M | 0.9906 |
| hourglass_symmetric | smooth taper | 3.0 | 467M | 0.9839 |
| plateau_1024 | wide middle plateau | 10.5 | 1.75B | 0.8910 |
| **uniform_1024** | **1024 everywhere** | **10.5** | **1.94B** | **0.8868** |

**Just making it wider wins.** uniform_1024 beats baseline by 2.3%. The "clever" shaped patterns (hourglass, plateau) don't outperform the brute-force "make every layer 1024" approach at this scale.

plateau_1024 comes close (0.8910 vs 0.8868) with fewer parameters though, suggesting bookend compression might offer efficiency at larger scales.

## Model Scaling (d10 to d16)

**Question**: How does BPB scale with depth on a single GB10?

nanochat's `--depth` flag is the single complexity dial. Width, heads, learning rate, steps, and data budget are all auto-derived.

| Depth | Dim | Heads | Tokens | val_bpb | Time |
|-------|-----|-------|--------|---------|------|
| 10 | 640 | 5 | 736M | 0.9526 | 2.6h |
| 12 | 768 | 6 | 1.16B | 0.9072 | 5.3h |
| 14 | 896 | 7 | 1.72B | 0.8698 | 11.9h |
| **16** | **1024** | **8** | **2.47B** | **0.8391** | **21.6h** |

Clean log-linear scaling. Each +2 depth yields ~0.03-0.04 BPB improvement. d16 achieves the best result in the entire campaign (0.8391) but takes 21.6 hours, 4x longer than d12 despite only ~1.8x more parameters.

## Combined DDL + DDB

**Question**: Do the improvements compose?

| Config | DDL Lambda | DDB | Budget | val_bpb |
|--------|-----------|-----|--------|---------|
| hourglass + DDL 0.01 | 0.01 | v1 hourglass | 3.0 | 0.9968 |
| hourglass + DDL 0.001 | 0.001 | v1 hourglass | 3.0 | 0.9877 |

Interesting finding: lambda=0.001 beats lambda=0.01 in the combined setting, the **opposite** of the DDL-only result. Stronger gating pressure may conflict with DDB's projection layers. This needs a full-budget run to confirm.

## All Results Ranked

| Config | Group | val_bpb | vs Baseline |
|--------|-------|---------|-------------|
| d16 | scaling | **0.8391** | -7.5% |
| uniform_1024 | DDB | **0.8868** | -2.3% |
| plateau_1024 | DDB | **0.8910** | -1.8% |
| ddl_lambda_0.01 | DDL | **0.9036** | -0.4% |
| ddl_lambda_0.001 | DDL | 0.9068 | -0.1% |
| baseline (d12) | -- | 0.9073 | -- |

Note: d16 and DDB configs have more parameters than baseline d12. DDL's improvement is the only one at iso-parameter count.

## Compute Budget

| Group | Experiments | Time |
|-------|-------------|------|
| DDL ablation | 4 | 20.0h |
| DDB ablation | 5 | 33.8h |
| Model scaling | 4 | 41.3h |
| Combined | 2 | 5.7h |
| **Total** | **15** | **~99h (4.1 days)** |

13.7 billion tokens processed. Zero failures across all 15 experiments.

## Takeaways

**DDL works at iso-parameters.** 0.4% BPB improvement with zero throughput overhead. The per-token adaptive dimension gating is a genuinely useful technique.

**Width beats shape.** At depth=12, "just make it wider" outperforms clever dimension routing patterns. Shape-dependent effects may emerge at larger depths where you can't afford uniform width.

**Scaling is clean on GB10.** d10 to d16 shows textbook scaling behavior, validating nanochat's compute-optimal design on single-GPU hardware.

**Composition is an open question.** DDL + DDB combined results show an unexpected lambda ranking reversal. Whether the improvements truly compose or interfere needs more compute to answer.

**DGX Spark works for ablation research.** 15 experiments, 4 days, 13.7B tokens, no failures. Not fast, but entirely sufficient for systematic architectural exploration on a desktop.

## Reproduce

```bash
git clone https://github.com/lakshmankolasani/nanochat && cd nanochat
uv venv && source .venv/bin/activate
uv pip install -e ".[gpu]"

python -m ablations.runner --list                    # see all experiments
python -m ablations.runner --group ddl_ablation      # run a group
python -m ablations.runner --group model_scaling --dry-run  # preview commands
```