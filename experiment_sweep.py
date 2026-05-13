#!/usr/bin/env python3
"""
experiment_sweep.py

Automated multi-run sweep for the masked hypothesis generation experiment.

For each (video, seed) pair, runs experiment_masked_hypothesis.py with
n_frames=1 (single attack frame, avoiding drift accumulation in the SIFT
reference).  Computes two aggregate metrics:

  Metric 1 — Pool Recall @ τ
    MaxPoolIoU = max over M mask hypotheses of IoU(hyp_k, GT)
    PoolRecall@τ = fraction of runs where MaxPoolIoU >= τ
    Measures whether the generation step produces at least one good candidate.

  Metric 2 — Ranking Efficiency  (only computed when MaxPoolIoU > min_iou_threshold)
    TopSIFT_IoU  = IoU of the top-SIFT-ranked mask hypothesis with GT
    RankingEff   = TopSIFT_IoU / MaxPoolIoU  ∈ [0, 1]
    Measures how much of the available best IoU SIFT ranking actually recovers.
    Also computed for pscore-ranking as a baseline.

Results are saved to a .npz (raw per-run data) and a summary .csv.

Usage
-----
  # All videos, 20 seeds each, single frame per run
  python experiment_sweep.py

  # Specific videos
  python experiment_sweep.py --videos car1 racing --n_seeds 10

  # Change hyperparameters
  python experiment_sweep.py --N_masks 4 --iou_threshold 0.3

  # Skip experiment runs and just re-analyse existing logs
  python experiment_sweep.py --analyse_only

  # Vanilla RTAA baseline at the new eps=10 threat model
  python experiment_sweep.py --attack_variant rtaa --eps 10 --out_dir out/sweep_eps10

  # Adaptive attack: RTAA + DoG suppress (zero-gap term only)
  python experiment_sweep.py --attack_variant rtaa_sift --eps 10 \
    --alpha_dog 1000 --gamma_kornia 0 --out_dir out/sweep_eps10

  # Adaptive attack with both terms
  python experiment_sweep.py --attack_variant rtaa_sift --eps 10 \
    --alpha_dog 1000 --gamma_kornia 1.0 --out_dir out/sweep_eps10

"""

import argparse
import json
import os
import random
import subprocess
import sys
from os.path import realpath, dirname, join

import numpy as np

# ---------------------------------------------------------------------------
# Metric computation from a single-frame log
# ---------------------------------------------------------------------------

def compute_metrics(log_path, iou_threshold=0.5):
    """
    Load a single experiment log and compute per-frame metrics.

    Returns a dict with:
      max_pool_iou       float  — best IoU achievable from mask hypothesis pool
      top_sift_iou       float  — IoU of top-SIFT-ranked raw mask hypothesis
      top_pscore_iou     float  — IoU of top-pscore-ranked raw mask hypothesis
      gt_sift_score      float  — SIFT score for GT bbox (upper bound reference)
      pred_sift_score    float  — SIFT score for attack pred bbox (lower bound)
      attack_pred_iou    float  — IoU of attack prediction with GT
      ranking_eff_sift   float  — top_sift_iou / max_pool_iou  (nan if pool too weak)
      ranking_eff_pscore float  — top_pscore_iou / max_pool_iou (nan if pool too weak)
      pool_hit           bool   — max_pool_iou >= iou_threshold
      sift_hit           bool   — top_sift_iou >= iou_threshold
    """
    d = np.load(log_path, allow_pickle=True)

    # Only use frame 0 (n_frames=1 run)
    fi = 0
    gt_bbox   = d['attack_gt_bboxes'][fi]
    pred_bbox = d['attack_pred_bboxes'][fi]

    mask_iou_gt  = d['attack_mask_iou_gt'][fi]    # (M,)
    mask_sift    = d['attack_mask_sift'][fi]       # (M,)
    mask_pscore  = d['attack_mask_pscore'][fi]     # (M,)

    gt_sift   = float(d['attack_gt_sift_scores'][fi])
    pred_sift = float(d['attack_pred_sift_scores'][fi])

    def _iou(a, b):
        ax2, ay2 = a[0]+a[2], a[1]+a[3]
        bx2, by2 = b[0]+b[2], b[1]+b[3]
        iw = max(0., min(ax2,bx2) - max(a[0],b[0]))
        ih = max(0., min(ay2,by2) - max(a[1],b[1]))
        inter = iw * ih
        union = a[2]*a[3] + b[2]*b[3] - inter
        return float(inter/union) if union > 0 else 0.

    # Metric 1: pool quality
    max_pool_iou = float(np.nanmax(mask_iou_gt))

    # Metric 2: reranking — SIFT
    top_sift_idx = int(np.argmax(mask_sift))
    top_sift_iou = float(mask_iou_gt[top_sift_idx])

    # Metric 2 baseline: reranking — pscore
    top_pscore_idx = int(np.argmax(mask_pscore))
    top_pscore_iou = float(mask_iou_gt[top_pscore_idx])

    # Attack prediction IoU
    attack_pred_iou = _iou(pred_bbox, gt_bbox)

    # Ranking efficiency (undefined when pool is too weak)
    if max_pool_iou > 1e-3:
        eff_sift   = top_sift_iou   / max_pool_iou
        eff_pscore = top_pscore_iou / max_pool_iou
    else:
        eff_sift   = float('nan')
        eff_pscore = float('nan')

    return {
        'max_pool_iou':       max_pool_iou,
        'top_sift_iou':       top_sift_iou,
        'top_pscore_iou':     top_pscore_iou,
        'gt_sift_score':      gt_sift,
        'pred_sift_score':    pred_sift,
        'attack_pred_iou':    attack_pred_iou,
        'ranking_eff_sift':   eff_sift,
        'ranking_eff_pscore': eff_pscore,
        'pool_hit':           max_pool_iou >= iou_threshold,
        'sift_hit':           top_sift_iou >= iou_threshold,
    }


# ---------------------------------------------------------------------------
# Dataset listing
# ---------------------------------------------------------------------------

def list_videos(dataset_name, data_dir):
    """Return all video names in the dataset JSON."""
    json_path = join(data_dir, dataset_name + '.json')
    info = json.load(open(json_path))
    return sorted(v['name'] for v in info.values())


# ---------------------------------------------------------------------------
# Run one experiment and return its log path
# ---------------------------------------------------------------------------

def run_experiment(video, seed, out_dir, N_masks, cluster_iou_eps, model,
                   attack_variant, eps, n_iter, alpha_dog, gamma_kornia):
    """
    Call experiment_masked_hypothesis.py as a subprocess.
    Returns the path to the output log, or None on failure.

    The log stem encodes attack_variant and eps so that paired-comparison
    sweeps (vanilla RTAA vs RTAA+SIFT) don't clobber each other's logs.
    """
    variant_tag = (attack_variant if attack_variant == 'rtaa'
                   else f"{attack_variant}_a{int(alpha_dog)}_g{gamma_kornia:.2f}")
    stem     = f"log_masked_{video}_s{seed}_{variant_tag}_eps{int(eps)}"
    log_path = join(out_dir, f"{stem}.npz")
    if os.path.exists(log_path):
        return log_path   # already done — skip

    cmd = [
        sys.executable,
        join(dirname(realpath(__file__)), 'experiment_masked_hypothesis.py'),
        '--video',          video,
        '--seed',           str(seed),
        '--out_dir',        out_dir,
        '--out_stem',       stem,
        '--N_masks',        str(N_masks),
        '--n_frames',       '1',
        '--cluster_iou_eps', str(cluster_iou_eps),
        '--model',          model,
        '--attack_variant', attack_variant,
        '--eps',            str(eps),
        '--n_iter',         str(n_iter),
        '--alpha_dog',      str(alpha_dog),
        '--gamma_kornia',   str(gamma_kornia),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {video} seed={seed}")
        print(result.stderr[-800:])
        return None

    return log_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Sweep masked hypothesis experiment over videos and seeds'
    )
    parser.add_argument('--dataset',    default='VOT2018')
    parser.add_argument('--videos',     nargs='+', default=None,
                        help='Video names to include (default: all in dataset)')
    parser.add_argument('--n_seeds',    type=int, default=20,
                        help='Number of random seeds per video')
    parser.add_argument('--seed_base',  type=int, default=0,
                        help='Seeds are seed_base, seed_base+1, ..., seed_base+n_seeds-1')
    parser.add_argument('--N_masks',    type=int, default=8)
    parser.add_argument('--cluster_iou_eps', type=float, default=0.5)
    parser.add_argument('--iou_threshold',   type=float, default=0.5,
                        help='IoU threshold for PoolRecall and SiftHit (default 0.5)')
    parser.add_argument('--model',      default='SiamRPNvot.model')
    parser.add_argument('--out_dir',    default='out/sweep')
    parser.add_argument('--analyse_only', action='store_true',
                        help='Skip running experiments; just analyse existing logs')
    parser.add_argument('--attack_variant', default='rtaa',
                        choices=['rtaa', 'rtaa_sift'],
                        help='Vanilla RTAA, or RTAA augmented with SIFT-evasion')
    parser.add_argument('--eps',          type=float, default=10.0,
                        help='L_inf perturbation budget (default 10).')
    parser.add_argument('--n_iter',       type=int,   default=5,
                        help='PGD iterations inside the attack loop')
    parser.add_argument('--alpha_dog',    type=float, default=1000.0,
                        help='DoG-suppress loss weight (rtaa_sift only)')
    parser.add_argument('--gamma_kornia', type=float, default=0.0,
                        help='Kornia-SIFT BPDA surrogate weight (rtaa_sift only)')
    args = parser.parse_args()

    data_dir = join(dirname(realpath(__file__)), 'data')
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Video list ---
    if args.videos:
        videos = args.videos
    else:
        videos = list_videos(args.dataset, data_dir)
    seeds = list(range(args.seed_base, args.seed_base + args.n_seeds))

    print(f"Videos: {len(videos)}  Seeds per video: {args.n_seeds}  "
          f"Total runs: {len(videos) * args.n_seeds}")
    print(f"IoU threshold: {args.iou_threshold}  N_masks: {args.N_masks}")

    # --- Run experiments ---
    all_results = []   # list of dicts

    for video in videos:
        video_results = []
        print(f"\n{'─'*60}")
        print(f"Video: {video}")
        for seed in seeds:
            variant_tag = (args.attack_variant if args.attack_variant == 'rtaa'
                           else f"{args.attack_variant}"
                                f"_a{int(args.alpha_dog)}_g{args.gamma_kornia:.2f}")
            expected_stem = (f"log_masked_{video}_s{seed}_"
                             f"{variant_tag}_eps{int(args.eps)}")
            if not args.analyse_only:
                log_path = run_experiment(
                    video, seed, args.out_dir,
                    args.N_masks, args.cluster_iou_eps, args.model,
                    args.attack_variant, args.eps, args.n_iter,
                    args.alpha_dog, args.gamma_kornia,
                )
            else:
                log_path = join(args.out_dir, f"{expected_stem}.npz")

            if log_path is None or not os.path.exists(log_path):
                print(f"  seed={seed}  MISSING")
                continue

            try:
                m = compute_metrics(log_path, iou_threshold=args.iou_threshold)
            except Exception as e:
                print(f"  seed={seed}  ERROR: {e}")
                continue

            m['video'] = video
            m['seed']  = seed
            video_results.append(m)
            all_results.append(m)

            print(f"  seed={seed:5d}  "
                  f"MaxPoolIoU={m['max_pool_iou']:.3f}  "
                  f"TopSIFT_IoU={m['top_sift_iou']:.3f}  "
                  f"RankEff_SIFT={m['ranking_eff_sift']:.3f}  "
                  f"RankEff_pscore={m['ranking_eff_pscore']:.3f}  "
                  f"PoolHit={int(m['pool_hit'])}")

        if video_results:
            _print_video_summary(video, video_results, args.iou_threshold)

    if not all_results:
        print("\nNo results collected.")
        return

    # --- Aggregate across all videos ---
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    _print_aggregate(all_results, args.iou_threshold)

    # --- Save raw results ---
    _save_results(all_results, args.out_dir, args.iou_threshold)


def _print_video_summary(video, results, tau):
    n = len(results)
    pool_recall   = np.mean([r['pool_hit']       for r in results])
    sift_hit      = np.mean([r['sift_hit']       for r in results])
    mean_max_iou  = np.nanmean([r['max_pool_iou']  for r in results])
    mean_top_sift = np.nanmean([r['top_sift_iou']  for r in results])
    mean_pred_iou = np.nanmean([r['attack_pred_iou'] for r in results])

    eff_sift_vals   = [r['ranking_eff_sift']   for r in results
                       if not np.isnan(r['ranking_eff_sift'])]
    eff_pscore_vals = [r['ranking_eff_pscore'] for r in results
                       if not np.isnan(r['ranking_eff_pscore'])]

    print(f"\n  {video}  (n={n} runs)")
    print(f"  Metric 1  PoolRecall@{tau}:      {pool_recall:.3f}  "
          f"(mean MaxPoolIoU={mean_max_iou:.3f})")
    print(f"  Metric 2  RankEff_SIFT:         "
          f"{np.nanmean(eff_sift_vals):.3f} ± {np.nanstd(eff_sift_vals):.3f}"
          f"  (n={len(eff_sift_vals)} valid)")
    print(f"            RankEff_pscore:        "
          f"{np.nanmean(eff_pscore_vals):.3f} ± {np.nanstd(eff_pscore_vals):.3f}")
    print(f"            SiftHit@{tau}:           {sift_hit:.3f}  "
          f"(mean TopSIFT_IoU={mean_top_sift:.3f})")
    print(f"            AttackPred_IoU (ref):  {mean_pred_iou:.3f}")


def _print_aggregate(results, tau):
    n = len(results)
    pool_recall   = np.mean([r['pool_hit']       for r in results])
    sift_hit      = np.mean([r['sift_hit']       for r in results])
    mean_max_iou  = np.nanmean([r['max_pool_iou']  for r in results])
    mean_top_sift = np.nanmean([r['top_sift_iou']  for r in results])
    mean_pred_iou = np.nanmean([r['attack_pred_iou'] for r in results])

    eff_sift_vals   = [r['ranking_eff_sift']   for r in results
                       if not np.isnan(r['ranking_eff_sift'])]
    eff_pscore_vals = [r['ranking_eff_pscore'] for r in results
                       if not np.isnan(r['ranking_eff_pscore'])]

    print(f"  n = {n} (video, seed) pairs\n")
    print(f"  ── Metric 1: Generation quality ──────────────────────────")
    print(f"     PoolRecall@{tau}:         {pool_recall:.3f}  "
          f"({int(pool_recall*n)}/{n} runs had a good candidate)")
    print(f"     Mean MaxPoolIoU:          {mean_max_iou:.3f}")
    print(f"")
    print(f"  ── Metric 2: Reranking quality (conditioned on pool) ──────")
    print(f"     RankingEff (SIFT):        "
          f"{np.nanmean(eff_sift_vals):.3f} ± {np.nanstd(eff_sift_vals):.3f}"
          f"  (n={len(eff_sift_vals)} runs with non-trivial pool)")
    print(f"     RankingEff (pscore, baseline): "
          f"{np.nanmean(eff_pscore_vals):.3f} ± {np.nanstd(eff_pscore_vals):.3f}")
    print(f"     SiftHit@{tau}:             {sift_hit:.3f}  "
          f"(SIFT rank-1 ≥ τ, unconditional)")
    print(f"     Mean TopSIFT_IoU:         {mean_top_sift:.3f}")
    print(f"")
    print(f"  ── Reference ──────────────────────────────────────────────")
    print(f"     Mean AttackPred IoU:      {mean_pred_iou:.3f}  "
          f"(corrupted tracker's output)")


def _save_results(results, out_dir, tau):
    import csv

    # CSV
    csv_path = join(out_dir, 'sweep_results.csv')
    fieldnames = [
        'video', 'seed',
        'max_pool_iou', 'top_sift_iou', 'top_pscore_iou',
        'ranking_eff_sift', 'ranking_eff_pscore',
        'pool_hit', 'sift_hit',
        'attack_pred_iou', 'gt_sift_score', 'pred_sift_score',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV  → {csv_path}")

    # NPZ
    npz_path = join(out_dir, 'sweep_results.npz')
    np.savez(
        npz_path,
        videos              = np.array([r['video']               for r in results]),
        seeds               = np.array([r['seed']                for r in results]),
        max_pool_iou        = np.array([r['max_pool_iou']        for r in results]),
        top_sift_iou        = np.array([r['top_sift_iou']        for r in results]),
        top_pscore_iou      = np.array([r['top_pscore_iou']      for r in results]),
        ranking_eff_sift    = np.array([r['ranking_eff_sift']    for r in results]),
        ranking_eff_pscore  = np.array([r['ranking_eff_pscore']  for r in results]),
        pool_hit            = np.array([r['pool_hit']            for r in results]),
        sift_hit            = np.array([r['sift_hit']            for r in results]),
        attack_pred_iou     = np.array([r['attack_pred_iou']     for r in results]),
        gt_sift_score       = np.array([r['gt_sift_score']       for r in results]),
        pred_sift_score     = np.array([r['pred_sift_score']     for r in results]),
        iou_threshold       = np.array(tau),
    )
    print(f"NPZ  → {npz_path}")


if __name__ == '__main__':
    main()
