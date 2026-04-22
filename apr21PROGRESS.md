# Physical Invariants Guided Recovery for Object Tracking Attacks
## Project Progress & Status

---

## 1. Problem Statement

Object trackers in autonomous systems (ground vehicles, drones) must reliably follow a target —
a vehicle, pedestrian, or cyclist — across frames, even under adversarial conditions.  Existing
defenses focus almost exclusively on **detection**: flagging when the tracker is being fooled and
handing control to a human or a rule-based fallback.

Detection alone is insufficient in safety-critical settings for two reasons:

- **Human takeover latency.** Alerting a human driver introduces unavoidable reaction delay,
  unacceptable at highway speeds.
- **Rule-based fallback blindness.** A fallback (e.g., constant-velocity prediction) cannot act
  safely without a correct estimate of the target's current state.

This project studies **recovery**: maintaining or restoring the correct target state *despite an
ongoing attack*, without requiring attack detection as a prerequisite.  The specific failure mode
addressed is **patch-induced tracker drift** in siamese region proposal network (SiamRPN)
trackers.

---

## 2. Threat Model

| Aspect | Detail |
|---|---|
| **System** | Autonomous platform (ground vehicle or drone) running a SiamRPN-based tracker |
| **Attack surface** | A physical adversarial patch or projector-based projection placed in the scene |
| **Attack goal** | Hijack the tracker's predictions — cause it to persistently follow the wrong location |
| **Defense goal** | Produce a usable, approximately correct tracking prediction despite the ongoing attack |
| **Defender knowledge** | Access to the raw camera frame sequence; no prior knowledge of the patch location or attack parameters |

---

## 3. Principled Design

The high-level design is captured in `init_design.png` and reproduced below.

**Title:** *Physical Invariants Guided Recovery for Object Tracking Attacks*

**Core principle:** Physical adversarial attacks have properties that benign scenes do not —
they are spatially localised, exhibit unnatural appearance discontinuities, and produce
response-map anomalies inconsistent with natural motion.  These **physical invariants** can be
exploited to guide recovery, even without detecting the attack explicitly.

The recovery pipeline has two stages:

```
Frame t ──► [ Image Purification ] ──► Tracker ──► { Hypotheses } ──► [ Low-level Feature  ] ──► Output
               (physical attack                                            Extraction + Trusted
                properties)          [ Model Adaptation ]                 Memory
               └──────── Hypothesis Generation ─────────┘   └── Hypothesis Re-ranking ──┘
```

### Stage 1 — Hypothesis Generation

Produces a **diverse set of candidate bounding boxes** that together likely contains the true
target, even though the adversarially corrupted tracker's top-1 output does not.  Two
complementary mechanisms:

- **Image Purification**: exploit physical attack properties (spatial locality, appearance
  discontinuity) to suppress or remove the adversarial influence before the tracker sees the
  frame — e.g., masking, inpainting, or frequency filtering.
- **Model Adaptation**: exploit the contrast between the model's benign behaviour and its
  malicious behaviour to generate hypotheses that span both the adversarial and true-target
  locations — e.g., input perturbation, dropout sampling, or response-map manipulation.

### Stage 2 — Hypothesis Re-ranking

Scores each candidate using signals that are **invariant to the attack**:

- **Low-level Feature Extraction**: features like SIFT keypoint correspondences that the
  adversarial perturbation is not optimised to corrupt.
- **Trusted Memory**: historical knowledge of the true target (template appearance, trajectory,
  geometric priors) that the attack cannot retroactively modify.

---

## 4. Experimental Platform

The codebase uses the **AttrackZone** attack framework (CCS 2022) as an experimental starting
point.  AttrackZone proposes a physical projector-based attack on SiamRPN trackers.  In the
codebase, the **RTAA (Real-time Transferable Adversarial Attack)** algorithm simulates this
physical projector attack digitally: it computes a bounded perturbation (L∞ ≤ 10) injected into
the SiamRPN search-region crop, driving the response-map peak to an adversarial location.

> **Important:** AttrackZone / RTAA is one instantiation of the threat model used for
> experiments.  The defense is intended to generalise to other patch-based tracking attacks;
> RTAA is not the sole target.

The dataset used is **VOT2018** (60 video sequences).

---

## 5. Current Technical Implementation

The current implementation is an **initial instantiation** of the principled design above, using
the SiamRPN + RTAA setup as a testbed.

### 5.1 Hypothesis Generation — Masking (Image Purification)

Inspired by **ObjectSeeker** (a certified defense for object detectors against patch-hiding
attacks), the hypothesis generation step runs SiamRPN on **M = 2N + 4 structured masked
versions** of the adversarial search crop per frame.

- **N horizontal stripe masks** and **N vertical stripe masks** (each zeroing one band of width
  ~crop/N), plus **4 quadrant masks**.  Default: N=8 → 20 masks total.
- **Coverage argument**: the adversarial perturbation is spatially concentrated within the
  search crop.  At least one mask will substantially overlap the perturbed region, suppressing
  the fake response peak and allowing the tracker to respond to the true target instead.
- **Diversity argument**: masks that do not cover the adversarial region produce adversarially-
  induced hypotheses; masks that do cover it produce true-target hypotheses.  The pool spans
  both, giving the re-ranker good discrimination signal.

After generating M raw hypotheses, **DBSCAN clustering** (IoU-based distance) deduplicates
spatially redundant predictions into K diverse cluster representatives, each annotated with a
`vote_count` (how many masks agreed on that spatial location).

### 5.2 Hypothesis Re-ranking — SIFT Correspondence

Each candidate hypothesis is scored by **SIFT local correspondence**: match keypoints between
`prev_frame[prev_pred_bbox]` (the reference region) and `curr_frame[hypothesis_bbox]`.

```
sift_score = 0.1 · match_ratio + 0.45 · inlier_ratio (RANSAC) + 0.45 · descriptor_sim
```

The adversarially-induced hypotheses should score low because the RTAA perturbation is synthetic
and has no real temporal correspondence to the previous target region.

**Known limitation:** The reference region `prev_pred_bbox` is the _tracker's previous
prediction_, which drifts to the adversarial location under sustained attack.  This corresponds
to the **Trusted Memory** component of the principled design being currently inadequate — a
drifted prediction is not trustworthy memory.  All experiments therefore use **n_frames = 1**
(single attack frame) to avoid compounding this drift.

### 5.3 Key Files

| File | Role |
|---|---|
| `experiment_masked_hypothesis.py` | Main experiment: masked generation + SIFT reranking + saves all arrays |
| `experiment_sweep.py` | Automated multi-video, multi-seed sweep with metric computation |
| `plot_masked_scatter.py` | IoU-vs-SIFT scatter, styled after `plot_hypothesis_ranking.py` |
| `plot_masked_hypothesis.py` | Multi-panel analysis: candidate quality, reranking, perturbation coverage |
| `experiment_hypothesis_ranking.py` | Earlier experiment: top-K anchors + SIFT (no masking) |
| `sift_alignment.py` | `SIFTAlignmentDetector` — keypoint extraction, ratio matching, RANSAC |

---

## 6. Evaluation Metrics

Two decoupled metrics designed so that either can be high or low independently:

**Metric 1 — Pool Recall @ τ (generation quality)**
```
MaxPoolIoU(t) = max_{k=1..M} IoU(mask_hyp_k, GT)
PoolRecall@τ  = fraction of runs where MaxPoolIoU ≥ τ   (τ = 0.5)
```
Purely measures whether at least one good candidate exists in the pool, independent of ranking.

**Metric 2 — Ranking Efficiency (reranking quality, conditioned on pool)**
```
RankingEfficiency(t) = IoU(top-SIFT-ranked hyp, GT) / MaxPoolIoU(t)   ∈ [0, 1]
```
Measures how much of the best available IoU is recovered by the SIFT ranker.  Only meaningful
when the pool is non-trivial.  Also computed for **pscore-ranking** as a baseline.

---

## 7. Initial Results (Partial Sweep — 8 / 60 VOT2018 Videos, ~121 Runs)

The automated sweep is still running.  Results cover `ants1`, `ants3`, `bag`, `ball1`, `ball2`,
`basketball`, `birds1`, `blanket`.

### 7.1 Aggregate numbers

| Metric | Value |
|---|---|
| **PoolRecall@0.5** (generation) | **0.322** |
| Mean MaxPoolIoU | 0.377 |
| **RankingEff — SIFT** (reranking) | **0.655 ± ~0.35** |
| **RankingEff — pscore** (baseline) | **0.664 ± ~0.35** |
| Mean TopSIFT IoU | 0.247 |
| Attack pred IoU (corrupted tracker, reference) | 0.188 |

### 7.2 Per-video breakdown

| Video | PoolRecall@0.5 | Mean MaxPoolIoU | RankEff SIFT | RankEff pscore | SIFT advantage |
|---|---|---|---|---|---|
| ants1 | 0.062 | 0.253 | 0.738 | 0.674 | **+0.064** |
| ants3 | 0.222 | 0.338 | 0.774 | 0.570 | **+0.204** |
| bag | 0.250 | 0.342 | 0.614 | 0.580 | **+0.034** |
| ball1 | 0.500 | 0.480 | 0.527 | 0.670 | −0.143 |
| ball2 | 0.667 | 0.532 | 0.539 | 0.430 | **+0.109** |
| basketball | 0.474 | 0.406 | 0.712 | 0.725 | −0.013 |
| birds1 | 0.133 | 0.343 | 0.557 | 0.860 | **−0.303** |
| blanket | 0.250 | 0.320 | 0.729 | 0.819 | −0.090 |

### 7.3 Key conclusions

**C1 — Generation is the primary bottleneck.**
PoolRecall@0.5 = 0.32 means that in ~2 out of 3 frames, no hypothesis in the masked pool has
IoU ≥ 0.5 with the true target.  Even perfect reranking cannot help when the pool contains no
good candidate.  This is the binding constraint.  Whether this is because (a) the masking scheme
does not successfully suppress the adversarial perturbation or (b) the true target simply has no
good-IoU anchor in the search region requires further investigation via the saved
`attack_perturb_cov` and `attack_att_per` arrays.

**C2 — The pipeline still beats the corrupted tracker.**
Mean TopSIFT IoU (0.247) > Mean AttackPred IoU (0.188).  The masked generation + SIFT pipeline
produces a better-than-corrupted output on average, even given the low pool recall.

**C3 — SIFT reranking provides no consistent advantage over pscore.**
RankEff-SIFT (0.655) ≈ RankEff-pscore (0.664) overall; SIFT wins on 4/8 videos, pscore on 4/8.
On `birds1` pscore substantially outperforms SIFT (0.860 vs 0.557), suggesting that in low-
texture or fast-motion scenes SIFT correspondence is unreliable.  The Trusted Memory component
(currently just `prev_pred_bbox`) is too weak to be a reliable reference.

**C4 — High variance throughout.**
Both ranking metrics have σ ≈ 0.35, indicating strong frame-level dependence on target texture,
camera motion, and perturbation structure.

**C5 — Engineering instability.**
~25% of seeds fail with `malloc` / `runtime_error` crashes, likely in the RTAA computation.
Failed runs are excluded rather than counted as failures, which may inflate reported metrics.

---

## 8. Next Steps

### 8.1 Diagnose the generation bottleneck (highest priority)

Before changing the masking scheme, verify the core assumption:

1. **Check the coverage→IoU correlation.** Use the saved `attack_perturb_cov` (fraction of
   perturbation energy in each mask's zeroed region) and `attack_mask_iou_gt` arrays.  If masks
   with higher coverage consistently produce higher IoU hypotheses, the mechanism works and we
   just need more / better masks.  If the correlation is weak, the perturbation may not be
   spatially concentrated as assumed, and a different purification strategy is needed.

2. **Visualise the perturbation structure.** Inspect `attack_att_per` heatmaps (saved in the
   `.npz` files) to understand whether RTAA concentrates energy in a localised region or spreads
   it across the full search crop.

### 8.2 Improve Trusted Memory (Hypothesis Re-ranking)

The current SIFT reference (`prev_pred_bbox`) is the weakest component — it drifts under attack.
This maps directly to the **Trusted Memory** block in the principled design.  Options:

- **Init-frame template anchor**: use the first-frame GT crop as the SIFT reference, which is
  fully drift-immune (but fails if the target's appearance changes significantly).
- **Separate stable tracker**: maintain a second lightweight tracker (e.g., CSRT) that is not
  vulnerable to RTAA's SiamRPN-specific gradient attack; use its output as the trusted reference.
- **Multi-frame memory**: maintain a short buffer of confident, high-SIFT-score past predictions
  as a rolling reference rather than a single previous frame.

### 8.3 Explore Model Adaptation for hypothesis generation

The current implementation only uses Image Purification (masking) for hypothesis generation.
The **Model Adaptation** branch of the design (exploiting benign/malicious behaviour differences)
is unexplored.  Concrete directions:

- **Response-map anomaly detection**: the adversarial peak is unusually sharp and isolated
  compared to benign responses; use this to identify and suppress the adversarial anchor before
  selecting candidates.
- **MC-Dropout sampling**: enable dropout at inference to generate stochastic response maps;
  average or diversify over samples.
- **Gradient-based input purification**: use the gradient of the response w.r.t. the input to
  localise and suppress the adversarial region (different from masking — adaptive rather than
  exhaustive).

### 8.4 Strengthen the low-level feature extractor

SIFT fails on texture-poor and fast-motion targets (birds1, blanket).  Replacements to evaluate:
- **ORB** (faster, comparable robustness)
- **SuperPoint** (learned keypoints, more robust to texture poverty)
- **Dense optical flow** (does not require texture; can score correspondence at every pixel)

### 8.5 Fix engineering crashes

The `malloc` crashes need root-cause analysis — add try/except around the RTAA computation and
count crashed runs as failed (IoU = 0) rather than excluding them, so reported metrics are
conservative rather than optimistic.
