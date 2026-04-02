#!/usr/bin/env python3
"""
plot_hypothesis_ranking.py

Scatter plot: hypothesis IoU-with-GT  vs.  two-stage SIFT score.

Each point is one top-K SiamRPN hypothesis from the first simulated attack frame.
The plot tests whether the low-level SIFT score correlates with IoU-with-GT —
i.e. whether it genuinely re-ranks hypotheses toward the correct target.

Three reference points are overlaid:
  ★ red star  — SiamRPN attack prediction (the box the corrupted tracker chose)
  ★ green star — GT box SIFT score (measured against prev GT; IoU-with-GT = 1.0)

Usage
-----
  python plot_hypothesis_ranking.py out/hypothesis_ranking/log_car1.npz
  python plot_hypothesis_ranking.py out/hypothesis_ranking/log_car1.npz --frame 2
  python plot_hypothesis_ranking.py out/hypothesis_ranking/log_car1.npz --out fig.png
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# IoU helper (identical logic to _bbox_iou in experiment_hypothesis_ranking.py)
# ---------------------------------------------------------------------------

def bbox_iou(a, b):
    """IoU between two [x, y, w, h] boxes."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot hypothesis SIFT score vs IoU-with-GT"
    )
    parser.add_argument('npz', help='Path to log_<video>.npz')
    parser.add_argument('--frame', type=int, default=0,
                        help='Which simulated frame index to plot (0 = first, default)')
    parser.add_argument('--out', default=None,
                        help='Output path for the figure (default: same dir as npz)')
    args = parser.parse_args()

    d = np.load(args.npz)

    n_frames = len(d['attack_frame_idxs'])
    fi = args.frame
    assert 0 <= fi < n_frames, f"--frame {fi} out of range [0, {n_frames - 1}]"

    frame_idx     = int(d['attack_frame_idxs'][fi])
    gt_bbox       = d['attack_gt_bboxes'][fi]          # (4,)
    hyp_bboxes    = d['attack_hyp_bboxes'][fi]         # (K, 4)
    hyp_sift      = d['attack_hyp_sift'][fi]           # (K,)
    pred_bbox     = d['attack_pred_bboxes'][fi]        # (4,)
    pred_sift     = float(d['attack_pred_sift_scores'][fi])
    gt_sift       = float(d['attack_gt_sift_scores'][fi])

    K = len(hyp_sift)

    # IoU of each hypothesis with GT
    hyp_iou = np.array([bbox_iou(hyp_bboxes[k], gt_bbox) for k in range(K)])

    # IoU of the attack prediction with GT (reference point)
    pred_iou = bbox_iou(pred_bbox, gt_bbox)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 5))

    # --- Hypothesis scatter ---
    ax.scatter(
        hyp_iou, hyp_sift,
        color='steelblue',
        s=60,
        alpha=0.75,
        zorder=3,
        label=f'Hypotheses (K={K})',
    )

    # --- Attack prediction reference ---
    ax.scatter(
        pred_iou, pred_sift,
        marker='*', s=280, color='red', zorder=5,
        label=f'Attack pred  (IoU={pred_iou:.2f}, SIFT={pred_sift:.2f})',
    )

    # --- GT reference ---
    # GT IoU with itself = 1.0 by definition
    ax.scatter(
        1.0, gt_sift,
        marker='*', s=280, color='limegreen', zorder=5,
        label=f'GT box       (IoU=1.00, SIFT={gt_sift:.2f})',
    )

    # --- Annotations for top-3 SIFT-ranked hypotheses ---
    top3_ids = np.argsort(hyp_sift)[::-1][:3]
    for rank, idx in enumerate(top3_ids):
        ax.annotate(
            f'#{rank + 1}',
            xy=(hyp_iou[idx], hyp_sift[idx]),
            xytext=(4, 4), textcoords='offset points',
            fontsize=13, color='navy',
        )

    ax.set_xlabel('IoU with Ground Truth', fontsize=16)
    ax.set_ylabel('Two-stage SIFT Score', fontsize=16)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        f'Hypothesis SIFT score vs IoU with GT\n'
        f'Frame {frame_idx}  |  {os.path.basename(args.npz)}',
        fontsize=16,
    )
    ax.legend(fontsize=13, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Ideal region: top-right = high IoU AND high SIFT
    ax.axvline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.text(0.76, 0.03, 'high IoU\nlow SIFT', fontsize=12,
            color='gray', ha='center', va='bottom')
    ax.text(0.03, 0.76, 'low IoU\nhigh SIFT', fontsize=12,
            color='gray', ha='left', va='top')
    ax.text(0.76, 0.96, 'high IoU\nhigh SIFT\n(ideal)', fontsize=12,
            color='darkgreen', ha='center', va='top', style='italic')
    
    # enlarge x, y axis number labels and add some padding around the plot
    ax.tick_params(axis='both', which='major', labelsize=13)

    plt.tight_layout(rect=[0, 0, 0.78, 1])

    if args.out:
        out_path = args.out
    else:
        base = os.path.splitext(args.npz)[0]
        out_path = f"{base}_scatter_frame{fi}.png"

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")

    # -----------------------------------------------------------------------
    # Quick text summary
    # -----------------------------------------------------------------------
    print(f"\nFrame {frame_idx}  |  K={K} hypotheses")
    print(f"  Attack pred:  IoU={pred_iou:.3f}  SIFT={pred_sift:.3f}")
    print(f"  GT box:       IoU=1.000  SIFT={gt_sift:.3f}")
    top1_idx = int(np.argmax(hyp_sift))
    print(f"  SIFT rank-1:  IoU={hyp_iou[top1_idx]:.3f}  SIFT={hyp_sift[top1_idx]:.3f}")

    # Pearson correlation between IoU and SIFT score
    if hyp_iou.std() > 0 and hyp_sift.std() > 0:
        r = float(np.corrcoef(hyp_iou, hyp_sift)[0, 1])
        print(f"  Pearson r (IoU ↔ SIFT): {r:.3f}")


if __name__ == '__main__':
    main()
