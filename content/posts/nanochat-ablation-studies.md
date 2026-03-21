---
title: "Nanochat Ablation Studies: DDL, DDB, and Scaling on a GB10"
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

After [training an LLM from scratch](/posts/dgx-spark-nanochat-training/) on my DGX Spark, I wanted to go deeper. I added two experimental architectural features to [nanochat](https://github.com/karpathy/nanochat) (Karpathy's minimal GPT training codebase) — **DDL** and **DDB** — that haven't been systematically tested. So I built an ablation framework (not yet public) and ran 15 experiments over 4 days on a single GB10.

This post covers what I found. All results are single-seed runs; directional findings, not definitive conclusions.

## What Are DDL and DDB?

Both are experimental features I built on top of [nanochat](https://github.com/karpathy/nanochat) that try to make transformers more efficient by letting the model adapt its own capacity per-token and per-layer.

### DDL (Dynamic Dimensional Layers)

Standard transformers process every residual stream dimension equally for every token. DDL adds a learned gate per block:

```python
mask = sigmoid(Linear(norm(x)))   # (B, T, D) values in (0, 1)
x = x * mask                      # element-wise gating
loss += lambda * mean(mask*(1-mask))  # push gates toward 0 or 1
```

Simple tokens (common words, punctuation) activate fewer dimensions. Complex tokens use more capacity. Without the regularizer, gates settle near 0.5 and nothing is actually gated — the `mask*(1-mask)` term forces binary on/off decisions so dimensions are truly inactive for simple tokens.

DDL adds a small linear layer per block (~7M params total for d12, a 2.5% increase). In practice, `torch.compile` fuses this into the existing computation graph, resulting in **zero measured throughput overhead** — though this is a compiler optimization, not an architectural guarantee.

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

Learned linear projections handle dimension changes between blocks. The parameter overhead is substantial: hourglass_1024 adds 49M params (+17% vs baseline), plateau_1024 adds 90M (+31%), and uniform_1024 adds 116M (+41%). Attention heads scale proportionally (`num_heads = dim / 128`) to keep `head_dim=128` fixed, aligned with tensor core tile sizes.

## Experiment Setup

All 15 experiments ran on my DGX Spark, a single GB10 with 128 GB unified memory. Trained on [FineWeb-Edu](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle) with a 32K-vocab BPE tokenizer (details in the [training post](/posts/dgx-spark-nanochat-training/)). All results are reported as validation bits-per-byte (BPB) — lower is better, measuring compression quality on a held-out split.

| Setting | Value |
|---------|-------|
| GPU | 1x GB10 (Blackwell, SM 12.1) |
| Attention | SDPA — Scaled Dot-Product Attention (PyTorch native; no Flash Attention 3 on this hardware) |
| Optimizer | [Muon](https://github.com/KellerJordan/modded-nanogpt) (weight matrices) + AdamW (embeddings) |
| Base model | depth=12, 768 dim, 6 heads, 286M params |
| Throughput | ~52-62K tok/sec (~22% BF16 MFU — fraction of theoretical peak compute) |

I used two training budgets, set as tokens-per-parameter ratios. The 10.5 ratio comes from fitting Kaplan-style scaling laws on nanochat, which showed stable ~0.5 exponents for both params and tokens — meaning a fixed ratio gives roughly compute-optimal training (similar in spirit to [Chinchilla](https://arxiv.org/abs/2203.15556), but fit to this codebase):
- **ratio=10.5** for configs that overlap with the [initial training run](/posts/dgx-spark-nanochat-training/), so I could compare directly
- **ratio=3.0** for new/exploratory configs, as a cheaper screening pass — these results are directional, not conclusive

## DDL Lambda Sweep

**Question**: What entropy regularization weight works best for DDL?

| Config | Lambda | Budget | Tokens | val_bpb |
|--------|--------|--------|--------|---------|
| baseline (no DDL) | -- | 10.5 | 1.16B | 0.9073 |
| ddl_lambda_0.001 | 0.001 | 10.5 | 1.23B | 0.9068 |
| **ddl_lambda_0.01** | **0.01** | **10.5** | **1.23B** | **0.9036** |

A fourth config (lambda=0.005) ran at the 3.0 screening budget and scored 0.9835, but is not directly comparable to the 10.5 runs above.

**Lambda=0.01 wins**, 0.4% better than baseline — consistent with the initial DDL experiment in the [training post](/posts/dgx-spark-nanochat-training/). Weaker regularization (0.001) barely helps, meaning the model needs strong pressure to push gates toward binary decisions. Note: this is a single-seed result; 0.4% could be within run-to-run variance.

Both DDL (293M params) and baseline (286M) run at identical throughput (~52,900 tok/sec). The gating adds zero measurable overhead.

## DDB Dimension Patterns

**Question**: Which block-dimension pattern gives the best BPB?

| Config | Pattern | Params | Budget | Tokens | val_bpb |
|--------|---------|--------|--------|--------|---------|
| baseline | 768 x 12 | 286M | 10.5 | 1.16B | 0.9073 |
| hourglass_1024 | narrow-wide-narrow | 336M | 3.0 | 428M | 0.9906 |
| hourglass_symmetric | smooth taper | 357M | 3.0 | 467M | 0.9839 |
| plateau_1024 | wide middle plateau | 376M | 10.5 | 1.75B | 0.8910 |
| **uniform_1024** | **1024 everywhere** | **402M** | **10.5** | **1.94B** | **0.8868** |

**Wider layers win, but this is not an iso-parameter comparison.** uniform_1024 beats baseline by 2.3%, but has 1.4x the parameters and was trained on 1.7x more tokens. This shows that naively scaling width helps, but doesn't isolate the effect of dimension patterns. The hourglass variants used the 3.0 screening budget and are not directly comparable to the 10.5 configs.

plateau_1024 comes close (0.8910 vs 0.8868) with fewer parameters, suggesting bookend compression might offer efficiency at larger scales.

## Model Scaling (d10 to d16)

**Question**: How does BPB scale with depth on a single GB10?

nanochat's `--depth` flag is the single complexity dial. Everything else is auto-derived: `width = depth × 64`, `num_heads = width / 128`, batch size scales as `B ∝ D^0.383` ([Power Lines](https://arxiv.org/abs/2505.13738)), and LR scales as `η ∝ √(B/B_ref)` — all tuned at d12 and transferred via [muP](https://arxiv.org/abs/2203.03466)-style scaling.

| Depth | Dim | Heads | Params | Tokens | val_bpb | Time |
|-------|-----|-------|--------|--------|---------|------|
| 10 | 640 | 5 | 196M | 736M | 0.9526 | 2.6h |
| 12 | 768 | 6 | 286M | 1.16B | 0.9072 | 5.3h |
| 14 | 896 | 7 | 399M | 1.72B | 0.8698 | 11.9h |
| **16** | **1024** | **8** | **537M** | **2.47B** | **0.8391** | **21.6h** |

BPB decreases roughly log-linearly with parameter count, dropping ~0.04 BPB per ~1.4x parameter increase. d16 achieves the best result in the entire campaign (0.8391) but takes 21.6 hours, 4x longer than d12. Note that both parameters and tokens increase with depth, so this is not an isolated depth ablation.

## Combined DDL + DDB

**Question**: Do the improvements compose?

| Config | DDL Lambda | DDB | Budget | val_bpb |
|--------|-----------|-----|--------|---------|
| hourglass + DDL 0.01 | 0.01 | hourglass_1024 | 3.0 | 0.9968 |
| hourglass + DDL 0.001 | 0.001 | hourglass_1024 | 3.0 | 0.9877 |

Lambda=0.001 beats lambda=0.01 here — the **opposite** of the DDL-only result. However, both runs used the 3.0 screening budget, so this reversal could be noise. These were tested with hourglass_1024 (one of the weaker DDB variants) because it was the first DDB pattern implemented; compute budget constraints prevented testing with uniform_1024 or plateau_1024. A full-budget run with a stronger DDB config is needed before drawing conclusions about DDL/DDB interaction.

## All Results Ranked

**Caveat:** These configs vary in parameter count and training tokens. Only DDL is near iso-parameter (293M vs 286M baseline). Ranking by BPB alone overstates architectural contributions vs. raw scale.

| Config | Group | Params | val_bpb | vs Baseline |
|--------|-------|--------|---------|-------------|
| d16 | scaling | 537M | **0.8391** | -7.5% |
| uniform_1024 | DDB | 402M | **0.8868** | -2.3% |
| plateau_1024 | DDB | 376M | **0.8910** | -1.8% |
| ddl_lambda_0.01 | DDL | 293M | **0.9036** | -0.4% |
| ddl_lambda_0.001 | DDL | 293M | 0.9068 | -0.1% |
| baseline (d12) | -- | 286M | 0.9073 | -- |

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

**DDL shows a promising signal at near iso-parameters.** 0.4% BPB improvement (single seed) with zero measured throughput overhead. Worth validating with multiple seeds.

**Wider layers outperformed shaped patterns**, though this comparison is confounded by parameter count and training budget differences. Whether dimension *shape* matters independently of raw width remains open.

**Scaling is smooth from d10 to d16.** Each depth step shows consistent BPB improvement on GB10 single-GPU hardware.

**Composition is an open question.** DDL + DDB combined results show a lambda ranking reversal at the screening budget. Whether the improvements truly compose needs full-budget runs to answer.

**Single-GPU ablation campaigns are practical at this scale.** 15 experiments, 4 days, 13.7B tokens, no failures on a single GB10.

## Caveats

- All results are **single-seed** — small differences (especially DDL's 0.4%) may not be statistically significant
- DDB and scaling comparisons are **not iso-parameter** — larger configs had both more parameters and more training tokens
- Screening-budget (ratio=3.0) results are **directional only** — not directly comparable to full-budget runs
