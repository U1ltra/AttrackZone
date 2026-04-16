# Adversarial Tracker Recovery via Low-Level Spatio-Temporal Correspondence

## 1. Problem Statement

Object trackers in autonomous driving must reliably follow a target — a vehicle, pedestrian, or cyclist — across frames, even in adversarial conditions. Existing defenses against adversarial visual attacks on trackers focus almost exclusively on **detection**: flagging when the tracker is being fooled and handing control to a human or a rule-based fallback.

Detection alone is insufficient in safety-critical settings for two reasons:

1. **Human takeover latency.** Alerting a human driver introduces unavoidable reaction delay, which is unacceptable at highway speeds.
2. **Rule-based fallback blindness.** A fallback system (e.g., constant-velocity prediction) cannot act safely without a correct estimate of the target's current state. If the tracker has already drifted to the adversarial location, any plan built on that state is wrong.

This project studies **recovery**: maintaining or restoring the correct target state *despite an ongoing attack*, without requiring attack detection as a prerequisite. The specific failure mode addressed is **patch-induced tracker drift** in siamese region proposal network (SiamRPN) trackers.

---

## 2. Threat Model

### Attacker goal
Cause the tracker to persistently follow a wrong location — a physically placed adversarial patch or a digitally injected perturbation — rather than the true target. This is called a **hijacking attack**.

### Attack surface: SiamRPN internals
SiamRPN operates by cross-correlating a template crop (the target at initialisation) against a search-region crop extracted from each subsequent frame. The attack targets the **search-region crop** fed to the network.

The specific attack used in this codebase is **RTAA (Real-time Transferable Adversarial Attacks)**. At each frame it computes an adversarial perturbation `att_per` that, when added to the search-region tensor `x_crop`, causes the network's response map to peak at a chosen adversarial location rather than the true target:

```
x_adv = x_crop + att_per
response_map(x_adv) → peak at adversarial location, not the target
```

The perturbation is bounded (L∞ ≤ 10 pixel values) and accumulated across frames, making it persistent.

### Digital vs. physical attack
| | Digital | Physical |
|---|---|---|
| Perturbation location | Search-region tensor only | Raw camera pixels (printed patch) |
| SIFT sees it | No — raw frames unchanged | Yes — patch is in the scene |
| Observable signal | Tracking error diverges | Both tracking error and SIFT scores diverge |
| This codebase | Simulates physical by injecting `att_per` back onto raw pixels | Requires video with actual patch present |

### Attacker capabilities
- Can inject bounded pixel perturbations into the search-region crop seen by the tracker
- Cannot modify the tracker's architecture, weights, or the template crop used at initialisation
- Cannot control camera pose or background scene content

### Defender capabilities
- Has access to the raw frame sequence (possibly with injected perturbation visible)
- Can extract the tracker's internal response map and all decoded anchor hypotheses
- Cannot retrain the tracker online

---

## 3. Core Hypothesis

The adversarial attack corrupts the **high-level correlation score** (the SiamRPN response map) — that is exactly what it is designed to do. However, the **low-level spatio-temporal correspondence** between the true target region and adjacent frames remains largely intact.

This is because:
- SIFT keypoints are computed from local gradient structure (scale-space extrema), not from learned semantic features.
- The RTAA perturbation optimises against the SiamRPN cross-correlation objective, not against SIFT descriptors.
- Genuine object regions have natural texture that produces stable, geometrically consistent keypoint matches across frames. An adversarially-induced region does not — it is synthetic noise with no real temporal history.

**Claim:** A SIFT-based inter-frame correspondence score can distinguish the correct target hypothesis from adversarially-induced hypotheses, and can therefore re-rank the SiamRPN response map toward the correct target.

---

## 4. Challenges

### 4.1 The corrupted score problem
The SiamRPN response map is the primary ranking signal for candidate target states. Under attack, this signal is deliberately corrupted — the adversarial hypothesis receives the highest score. Naively taking the top-K anchors by SiamRPN score therefore biases the candidate set toward the adversarial location.

### 4.2 The diversity problem
Even if we avoid the top-1 SiamRPN prediction, the search region covers a bounded spatial area (~271×271 pixels, typically 3–4× the target size). Without a spatial prior, the K candidates selected for SIFT scoring will still cluster around the response map peak.

### 4.3 The texture problem
SIFT requires sufficient local texture to extract keypoints. Small, smooth, or low-contrast targets yield very few keypoints, making the correspondence score unreliable. The score degrades gracefully (returns near-zero for both correct and incorrect hypotheses) but loses discriminative power.

### 4.4 The drift accumulation problem
The attack perturbation accumulates across frames (`att_per` from the previous frame seeds the next). As the tracker drifts to the adversarial location, the "previous predicted bbox" used as the SIFT reference drifts with it. After several frames of drift, the reference itself is corrupt. Recovery should ideally happen within one or two frames of drift onset.

---

## 5. Design

### 5.1 Overview

The recovery pipeline operates as an auxiliary module alongside the standard SiamRPN tracker. At each frame:

1. Run the SiamRPN forward pass normally, obtaining the full anchor response map.
2. Decode **all** anchors (not just the top-1) as a spatial candidate pool.
3. Apply the two-stage SIFT pipeline to select and score the K most plausible candidates.
4. The top-ranked candidate by SIFT score is the recovery output.

The standard tracker's top-1 output is retained as a reference to compare against.

### 5.2 Stage 1 — Spatial Filtering via Global Homography

**Goal:** Narrow the candidate pool to anchors near the true target, without using the corrupted SiamRPN score.

**Method:**

1. Extract SIFT keypoints from the full previous frame and the full current frame.
2. Match keypoints with Lowe's ratio test and estimate a homography `H_bg` via RANSAC. This homography captures the global camera ego-motion (background motion).
3. Warp the previous predicted bounding box `prev_pred_bbox` through `H_bg`:

```
predicted_bbox = H_bg ∘ prev_pred_bbox
```

This gives the expected location of the target in the current frame *under pure camera motion* — independent of the attack.

4. Compute IoU between `predicted_bbox` and every decoded anchor hypothesis. Select the K anchors with the highest IoU.

**Why this works against the attack:** The adversarial perturbation does not shift the raw-frame SIFT keypoints (for the digital simulation, it's injected back onto pixels only within the search region, not globally). The background homography is estimated from the whole frame and is therefore largely unaffected.

**Fallback:** If `H_bg` estimation fails (fewer than `min_matches` good matches), fall back to top-K by SiamRPN pscore. This maintains pipeline continuity at the cost of losing the spatial prior.

### 5.3 Stage 2 — Local Correspondence Scoring

**Goal:** For each of the K spatially-selected candidates, score how strongly it corresponds to the previous target.

**Method:**

For each candidate hypothesis `hyp_i`:

1. Mask the previous frame to `prev_pred_bbox` and extract SIFT keypoints `kp_ref`.
2. Mask the current frame to `hyp_i.bbox` and extract SIFT keypoints `kp_hyp`.
3. Match keypoints between the two masked regions.
4. Compute the correspondence score:

```
sift_score(i) = 0.4 · match_ratio
              + 0.4 · inlier_ratio   (RANSAC geometric consistency)
              + 0.2 · descriptor_sim
```

5. Sort candidates by `sift_score` descending. The top-ranked candidate is the recovery output.

**Why this discriminates adversarial hypotheses:** The RTAA perturbation is synthetic — it has no real temporal correspondence to the previous template region. SIFT keypoints within an adversarially-induced bounding box will either not match the reference keypoints at all, or produce geometrically inconsistent matches (low inlier ratio after RANSAC).

### 5.4 Scoring the reference points

Two additional SIFT scores are computed per attack frame as baselines:

- **GT SIFT score:** `sift_local_score(prev_gt_bbox → curr_gt_bbox)` — upper bound; the correct target region against itself.
- **Pred SIFT score:** `sift_local_score(prev_pred_bbox → curr_pred_bbox)` — the corrupted tracker's own output; expected to be low once drift sets in.

The recovery is considered successful if:

```
top_sift_score ≈ gt_sift_score  >>  pred_sift_score
```

### 5.5 Design diagram

```
Frame t-1                         Frame t
─────────────────────────────     ─────────────────────────────────────────
                                  SiamRPN forward (on x_adv)
prev_frame  ──────────────────►   All decoded anchors (full pool, ~1445)
prev_pred_bbox                         │
     │                                 │
     │   Stage 1: global SIFT          │
     ├──────────────────────────►  H_bg ∘ prev_pred_bbox = predicted_bbox
     │                                 │
     │                            IoU filter → top-K candidates
     │                                 │
     │   Stage 2: local SIFT           │
     └──────────────────────────►  sift_score per candidate
                                       │
                                  sort by sift_score → recovery output
```

---

## 6. Implementation

### Key files

| File | Purpose |
|---|---|
| `net.py` | DaSiamRPN network definition |
| `run_attack.py` | RTAA attack, tracker init/eval, anchor generation |
| `sift_alignment.py` | `SIFTAlignmentDetector` — keypoint extraction, ratio matching, RANSAC inlier ratio |
| `experiment_sift_hypothesis.py` | Validates the geometry-stable / appearance-disrupted hypothesis over full video sequences |
| `experiment_hypothesis_ranking.py` | Main two-stage re-ranking experiment: benign vs. attack simulation, SIFT ranking, log + video output |
| `plot_hypothesis_ranking.py` | Scatter plot of hypothesis SIFT score vs. IoU-with-GT for a single frame |

### Running an experiment

```bash
# Run the two-stage ranking experiment
python experiment_hypothesis_ranking.py --dataset VOT2018 --video car1

# Reproduce with a fixed seed
python experiment_hypothesis_ranking.py --dataset VOT2018 --video car1 --seed 42 --K 10

# Plot the hypothesis scatter for the first simulated frame
python plot_hypothesis_ranking.py out/hypothesis_ranking/log_car1.npz

# Plot a specific frame
python plot_hypothesis_ranking.py out/hypothesis_ranking/log_car1.npz --frame 3
```

### Output artefacts

**`log_<video>.npz`** — per-frame arrays for both benign and attack scenarios:

| Array | Shape | Description |
|---|---|---|
| `attack_hyp_bboxes` | (N, K, 4) | Hypothesis boxes per frame |
| `attack_hyp_sift` | (N, K) | SIFT score per hypothesis |
| `attack_hyp_pscore` | (N, K) | SiamRPN penalised score (corrupted under attack) |
| `attack_gt_sift_scores` | (N,) | GT box SIFT score (upper bound) |
| `attack_pred_sift_scores` | (N,) | Attack prediction SIFT score (lower bound) |

**`video_<video>.mp4`** — annotated video at 5 fps:

| Colour | Meaning |
|---|---|
| Green | Ground-truth box |
| Blue | SiamRPN prediction, benign |
| Red | SiamRPN prediction, attack |
| Cyan → darker | Top-5 two-stage SIFT-ranked hypotheses |

Bottom-left HUD: `Frame | Top SIFT | GT SIFT | Pred SIFT`

---

## 7. Limitations and Open Questions

- **Digital attack SIFT visibility.** The attack perturbs the search-region tensor, not raw pixels. To make it visible to SIFT, the perturbation is resized and pasted back onto the frame. This is an approximation; real physical patches have different spatial structure (higher contrast, confined region).

- **Drift accumulation breaks Stage 1.** Once the tracker has drifted for several frames, `prev_pred_bbox` is already at the wrong location. The H_bg-warped prediction then points to the wrong region. Stage 1 assumes the previous prediction was approximately correct.

- **SIFT fails on low-texture targets.** Smooth objects (plain walls, sky, uniform vehicles) yield too few keypoints for a reliable score. A practical system may need to blend SIFT with a learned feature descriptor in such cases.

- **K selection.** The number of spatially-selected candidates K is a hyperparameter. Too small and the true target may not be in the pool; too large and SIFT computation becomes expensive. K=10 is a practical starting point for a 17×17×5 ≈ 1445-anchor response map.

- **Recovery latency.** The pipeline runs per-frame but has not been profiled for real-time operation. SIFT extraction and matching over K crops is the computational bottleneck; ORB or SuperPoint could replace SIFT for speed.
