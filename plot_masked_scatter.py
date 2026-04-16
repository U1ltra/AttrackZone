#!/usr/bin/env python3
"""
plot_masked_scatter.py

Scatter plot for experiment_masked_hypothesis.py output.
Mirrors the style of plot_hypothesis_ranking.py.

X-axis : IoU with ground truth
Y-axis : SIFT local correspondence score

Points
------
  ● small dots    — each of the M masked hypotheses
                    color encodes mask type: blue=h-stripe, orange=v-stripe, green=quadrant
                    alpha encodes perturbation coverage (darker = mask removed more perturbation)
  ● large circles — DBSCAN cluster representatives, sized by vote_count
  ★ red star      — SiamRPN attack prediction (the corrupted tracker's output)
  ★ green star    — GT box SIFT score (IoU = 1.0 by definition)

Annotations
-----------
  #1 #2 #3 — top SIFT-ranked cluster reps (the recovery output candidates)

Usage
-----
  python plot_masked_scatter.py out/masked_hypothesis/log_masked_car1.npz
  python plot_masked_scatter.py out/masked_hypothesis/log_masked_car1.npz --frame 2
  python plot_masked_scatter.py out/masked_hypothesis/log_masked_car1.npz --frame 2 --out fig.png
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def bbox_iou(a, b):
    ax2, ay2 = a[0]+a[2], a[1]+a[3]
    bx2, by2 = b[0]+b[2], b[1]+b[3]
    iw = max(0., min(ax2,bx2) - max(a[0],b[0]))
    ih = max(0., min(ay2,by2) - max(a[1],b[1]))
    inter = iw * ih
    union = a[2]*a[3] + b[2]*b[3] - inter
    return float(inter/union) if union > 0 else 0.


def _mask_color(name):
    name = str(name)
    if name.startswith('h'):   return 'steelblue'
    if name.startswith('v'):   return 'darkorange'
    return 'mediumseagreen'    # quadrant


def main():
    parser = argparse.ArgumentParser(
        description='Scatter plot: IoU-with-GT vs SIFT score for masked hypotheses'
    )
    parser.add_argument('npz', help='Path to log_masked_<video>.npz')
    parser.add_argument('--frame', type=int, default=0,
                        help='Log frame index to plot (0 = first attack frame)')
    parser.add_argument('--out', default=None,
                        help='Output file path (default: <npz_dir>/<stem>_scatter_frame<N>.png)')
    args = parser.parse_args()

    d  = np.load(args.npz, allow_pickle=True)
    fi = args.frame
    n_frames = len(d['attack_frame_idxs'])
    assert 0 <= fi < n_frames, f"--frame {fi} out of range [0, {n_frames-1}]"

    frame_idx    = int(d['attack_frame_idxs'][fi])
    gt_bbox      = d['attack_gt_bboxes'][fi]            # (4,)
    pred_bbox    = d['attack_pred_bboxes'][fi]          # (4,)
    pred_sift    = float(d['attack_pred_sift_scores'][fi])
    gt_sift      = float(d['attack_gt_sift_scores'][fi])
    mask_names   = d['mask_names']                      # (M,)
    M            = len(mask_names)

    # Per-mask data
    mask_bboxes  = d['attack_mask_bboxes'][fi]          # (M, 4)
    mask_sift    = d['attack_mask_sift'][fi]            # (M,)
    mask_iou_gt  = d['attack_mask_iou_gt'][fi]          # (M,)
    perturb_cov  = d['attack_perturb_cov'][fi]          # (M,)

    # Cluster data (NaN-padded)
    cl_bboxes    = d['attack_cluster_bboxes'][fi]       # (K, 4)
    cl_sift      = d['attack_cluster_sift'][fi]         # (K,)
    cl_iou_gt    = d['attack_cluster_iou_gt'][fi]       # (K,)
    cl_votes     = d['attack_cluster_votes'][fi]        # (K,)

    # Valid cluster rows (not NaN-filled)
    cl_valid = ~np.isnan(cl_sift)
    cl_bboxes_v = cl_bboxes[cl_valid]
    cl_sift_v   = cl_sift[cl_valid]
    cl_iou_v    = cl_iou_gt[cl_valid]
    cl_votes_v  = cl_votes[cl_valid].astype(float)

    pred_iou = bbox_iou(pred_bbox, gt_bbox)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 5))

    # --- Masked hypotheses (small dots, color by type, alpha by perturb coverage) ---
    cov_norm = perturb_cov / (perturb_cov.max() + 1e-8)   # [0,1] for alpha scaling

    for mi in range(M):
        color = _mask_color(mask_names[mi])
        alpha = float(np.clip(0.35 + 0.65 * cov_norm[mi], 0.35, 1.0))
        ax.scatter(
            mask_iou_gt[mi], mask_sift[mi],
            color=color, s=55, alpha=alpha, zorder=3,
            edgecolors='none',
        )

    # --- Cluster representatives (larger hollow circles, sized by vote_count) ---
    # These show spatial deduplication: many masks predicting the same location
    # get collapsed into one cluster. Size = how many masks agreed (vote_count).
    # NOTE: reps are chosen by pscore, not SIFT — so #1 rank is annotated on
    # the raw mask dots below, not here.
    if len(cl_sift_v) > 0:
        vote_scale = 120 + 60 * cl_votes_v
        # ax.scatter(
        #     cl_iou_v, cl_sift_v,
        #     s=vote_scale, color='none', edgecolors='black', linewidths=1.5,
        #     zorder=4, alpha=0.85,
        #     label=f'Cluster reps — {int(cl_valid.sum())} clusters\n'
        #           f'(circle size ∝ vote count)',
        # )
        # Label each cluster rep with its vote count only (no rank)
        for ki in range(len(cl_sift_v)):
            ax.annotate(
                f'v={int(cl_votes_v[ki])}',
                xy=(cl_iou_v[ki], cl_sift_v[ki]),
                xytext=(5, 3), textcoords='offset points',
                fontsize=9, color='black',
            )

    # --- Annotate top-3 raw mask hypotheses by SIFT score ---
    # These are the actual recovery candidates: the dots with highest SIFT
    # correspondence to the previous template.
    top3_mask_idx = np.argsort(mask_sift)[::-1][:3]
    for rank, mi in enumerate(top3_mask_idx):
        ax.annotate(
            f'#{rank+1}',
            xy=(mask_iou_gt[mi], mask_sift[mi]),
            xytext=(5, 4), textcoords='offset points',
            fontsize=13, color='navy', fontweight='bold',
        )

    # --- Attack prediction ---
    ax.scatter(
        pred_iou, pred_sift,
        marker='*', s=300, color='red', zorder=6,
        label=f'Attack pred  (IoU={pred_iou:.2f}, SIFT={pred_sift:.2f})',
    )

    # --- GT reference ---
    ax.scatter(
        1.0, gt_sift,
        marker='*', s=300, color='limegreen', zorder=6,
        label=f'GT box       (IoU=1.00, SIFT={gt_sift:.2f})',
    )

    # --- Quadrant guidelines ---
    ax.axvline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.text(0.76, 0.03, 'high IoU\nlow SIFT',   fontsize=12, color='gray',
            ha='center', va='bottom')
    ax.text(0.03, 0.76, 'low IoU\nhigh SIFT',  fontsize=12, color='gray',
            ha='left',   va='top')
    ax.text(0.76, 0.96, 'high IoU\nhigh SIFT\n(ideal)', fontsize=12,
            color='darkgreen', ha='center', va='top', style='italic')

    # --- Mask type legend ---
    type_legend = [
        mpatches.Patch(color='steelblue',      label=f'h-stripe masks ({args.N_masks if hasattr(args,"N_masks") else "N"})'),
        mpatches.Patch(color='darkorange',     label='v-stripe masks'),
        mpatches.Patch(color='mediumseagreen', label='quadrant masks (4)'),
    ]
    # leg1 = ax.legend(handles=type_legend, fontsize=11, loc='lower right',
    #                  title='Mask type  (darker α = higher pert. coverage)',
    #                  title_fontsize=9)
    # put legend outside the plot area on the right
    leg1 = ax.legend(handles=type_legend, fontsize=11, loc='center left', bbox_to_anchor=(1.02, 0.3),
                     title='Mask type  (darker α = higher pert. coverage)', title_fontsize=9)
    ax.add_artist(leg1)
    # ax.legend(fontsize=11, loc='upper left')
    # put legend outside the plot area on the right
    ax.legend(fontsize=11, loc='center left', bbox_to_anchor=(1.02, 0.7))

    ax.set_xlabel('IoU with Ground Truth', fontsize=16)
    ax.set_ylabel('SIFT Correspondence Score', fontsize=16)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        f'Masked hypothesis SIFT score vs IoU with GT\n'
        f'Frame {frame_idx}  |  {os.path.basename(args.npz)}',
        fontsize=16,
    )
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 0.78, 1])

    if args.out:
        out_path = args.out
    else:
        base = os.path.splitext(args.npz)[0]
        out_path = f"{base}_scatter_frame{fi}.png"

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")

    # -------------------------------------------------------------------------
    # Text summary
    # -------------------------------------------------------------------------
    print(f"\nFrame {frame_idx}  |  {M} masked hypotheses  |  {int(cl_valid.sum())} clusters")
    print(f"  Attack pred:     IoU={pred_iou:.3f}  SIFT={pred_sift:.3f}")
    print(f"  GT box:          IoU=1.000  SIFT={gt_sift:.3f}")
    if len(cl_sift_v) > 0:
        print(f"  Cluster rank-1:  IoU={cl_iou_v[0]:.3f}  SIFT={cl_sift_v[0]:.3f}"
              f"  votes={int(cl_votes_v[0])}")
    best_mask_idx = int(np.argmax(mask_iou_gt))
    print(f"  Best mask hyp:   IoU={mask_iou_gt[best_mask_idx]:.3f}"
          f"  SIFT={mask_sift[best_mask_idx]:.3f}"
          f"  mask={mask_names[best_mask_idx]}"
          f"  cov={perturb_cov[best_mask_idx]:.3f}")
    if mask_iou_gt.std() > 0 and mask_sift.std() > 0:
        r = float(np.corrcoef(mask_iou_gt, mask_sift)[0, 1])
        print(f"  Pearson r (IoU ↔ SIFT, all masked hyps): {r:.3f}")


if __name__ == '__main__':
    main()
