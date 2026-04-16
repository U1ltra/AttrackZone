#!/usr/bin/env python3
"""
plot_masked_hypothesis.py

Analysis and visualisation for experiment_masked_hypothesis.py output.

Three analysis panels
─────────────────────
  1. Candidate quality   — IoU-with-GT distribution for (a) all M masked hypotheses
                           and (b) K cluster representatives, per frame.
                           Shows whether at least one good hypothesis exists in the pool.

  2. Reranking quality   — For each frame: scatter of (SIFT score vs IoU-with-GT)
                           for cluster reps.  Ideal: high-IoU rep also has high SIFT.

  3. Perturbation coverage — For each mask, its average perturbation coverage
                            (fraction of att_per energy in the zeroed region)
                            vs. the max IoU-with-GT achieved by that mask across frames.
                            Hypothesis: masks that remove more perturbation produce
                            better hypotheses.

  4. Crop + perturbation heatmap — For a chosen frame: side-by-side of the clean
                                   search crop, the perturbation magnitude heatmap,
                                   and a grid of all mask patterns with their
                                   hypothesis centre overlaid.

Usage
─────
  python plot_masked_hypothesis.py out/masked_hypothesis/log_masked_car1.npz
  python plot_masked_hypothesis.py out/masked_hypothesis/log_masked_car1.npz --frame 2
  python plot_masked_hypothesis.py out/masked_hypothesis/log_masked_car1.npz --frame 2 --show
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path):
    return np.load(path, allow_pickle=True)


def _best_iou_per_frame(mask_iou_gt):
    """Best IoU with GT across all M masks, per frame.  Shape (NF,)."""
    return np.nanmax(mask_iou_gt, axis=1)


def _top_cluster_iou(cluster_iou_gt):
    """IoU with GT of the top SIFT-ranked cluster rep (column 0), per frame."""
    return cluster_iou_gt[:, 0]


# ---------------------------------------------------------------------------
# Panel 1 — Candidate quality
# ---------------------------------------------------------------------------

def plot_candidate_quality(d, ax_mask, ax_cluster, frame_idxs):
    """
    Box-per-frame showing IoU distribution of masked hypotheses and cluster reps.
    """
    mask_iou   = d['attack_mask_iou_gt']        # (NF, M)
    cluster_iou = d['attack_cluster_iou_gt']    # (NF, K)
    NF = mask_iou.shape[0]
    frames = frame_idxs if frame_idxs is not None else list(range(NF))

    # -- masked hyps --
    data_mask = [mask_iou[t][~np.isnan(mask_iou[t])] for t in frames]
    ax_mask.boxplot(data_mask, positions=range(len(frames)), widths=0.5,
                    patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.6),
                    medianprops=dict(color='navy', linewidth=2))
    best_iou = [np.nanmax(mask_iou[t]) for t in frames]
    ax_mask.plot(range(len(frames)), best_iou, 'r^-', markersize=7,
                 label='best mask IoU')
    ax_mask.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
    ax_mask.set_xticks(range(len(frames)))
    ax_mask.set_xticklabels([str(d['attack_frame_idxs'][t]) for t in frames])
    ax_mask.set_xlabel('frame index')
    ax_mask.set_ylabel('IoU with GT')
    ax_mask.set_title('Masked hypothesis IoU distribution (all M masks)')
    ax_mask.set_ylim(-0.05, 1.05)
    ax_mask.legend(fontsize=8)

    # -- cluster reps --
    data_cl = [cluster_iou[t][~np.isnan(cluster_iou[t])] for t in frames]
    ax_cluster.boxplot(data_cl, positions=range(len(frames)), widths=0.5,
                       patch_artist=True,
                       boxprops=dict(facecolor='darkorange', alpha=0.6),
                       medianprops=dict(color='darkred', linewidth=2))
    top_iou = [cluster_iou[t, 0] for t in frames]  # SIFT-ranked #1
    ax_cluster.plot(range(len(frames)), top_iou, 'rs-', markersize=7,
                    label='top SIFT cluster IoU')
    ax_cluster.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
    ax_cluster.set_xticks(range(len(frames)))
    ax_cluster.set_xticklabels([str(d['attack_frame_idxs'][t]) for t in frames])
    ax_cluster.set_xlabel('frame index')
    ax_cluster.set_ylabel('IoU with GT')
    ax_cluster.set_title('Cluster rep IoU (after DBSCAN + SIFT rank)')
    ax_cluster.set_ylim(-0.05, 1.05)
    ax_cluster.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Panel 2 — Reranking quality (scatter per frame)
# ---------------------------------------------------------------------------

def plot_reranking_quality(d, axes, frame_idxs):
    """
    Scatter of SIFT score vs IoU-with-GT for each cluster rep, one subplot per frame.
    Color encodes vote_count.
    """
    cluster_sift   = d['attack_cluster_sift']      # (NF, K)
    cluster_iou    = d['attack_cluster_iou_gt']    # (NF, K)
    cluster_votes  = d['attack_cluster_votes']     # (NF, K)
    cluster_bboxes = d['attack_cluster_bboxes']    # (NF, K, 4) — check for NaN
    gt_sift   = d['attack_gt_sift_scores']
    pred_sift = d['attack_pred_sift_scores']
    NF = cluster_sift.shape[0]
    frames = frame_idxs if frame_idxs is not None else list(range(NF))

    for ax_i, t in enumerate(frames):
        ax = axes[ax_i]
        valid = ~np.isnan(cluster_sift[t]) & ~np.isnan(cluster_iou[t])
        s   = cluster_sift[t][valid]
        iou = cluster_iou[t][valid]
        v   = cluster_votes[t][valid]
        sc  = ax.scatter(s, iou, c=v, cmap='cool', s=60, zorder=3,
                         vmin=1, vmax=max(v.max(), 1) if len(v) > 0 else 1)
        # Mark rank-1 (highest SIFT) separately
        if len(s) > 0:
            ax.scatter(s[0], iou[0], marker='*', s=180, c='red', zorder=5,
                       label='top SIFT')
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(gt_sift[t], color='green', linestyle=':', linewidth=1.2,
                   label=f'GT sift={gt_sift[t]:.2f}')
        ax.axvline(pred_sift[t], color='red', linestyle=':', linewidth=1.2,
                   label=f'Pred sift={pred_sift[t]:.2f}')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('SIFT score')
        ax.set_ylabel('IoU with GT')
        ax.set_title(f'Frame {d["attack_frame_idxs"][t]}')
        ax.legend(fontsize=6)
        plt.colorbar(sc, ax=ax, label='vote_count')


# ---------------------------------------------------------------------------
# Panel 3 — Perturbation coverage vs hypothesis quality
# ---------------------------------------------------------------------------

def plot_perturbation_coverage(d, ax):
    """
    Per-mask scatter: x = mean perturbation coverage (across frames),
                      y = mean best IoU-with-GT for that mask (across frames).
    Annotate each point with the mask name.
    """
    perturb_cov = d['attack_perturb_cov']     # (NF, M)
    mask_iou    = d['attack_mask_iou_gt']     # (NF, M)
    mask_names  = d['mask_names']

    mean_cov = perturb_cov.mean(axis=0)        # (M,)
    mean_iou = np.nanmean(mask_iou, axis=0)    # (M,)
    M = len(mean_cov)

    # Colour by mask type: h-stripe blue, v-stripe orange, quadrant green
    colors = []
    for name in mask_names:
        name = str(name)
        if name.startswith('h'):
            colors.append('steelblue')
        elif name.startswith('v'):
            colors.append('darkorange')
        else:
            colors.append('green')

    ax.scatter(mean_cov, mean_iou, c=colors, s=60, zorder=3)
    for xi, yi, name in zip(mean_cov, mean_iou, mask_names):
        ax.annotate(str(name), (xi, yi), fontsize=6, ha='left', va='bottom')

    ax.set_xlabel('Mean perturbation coverage (fraction of att_per energy removed)')
    ax.set_ylabel('Mean IoU-with-GT of mask hypothesis')
    ax.set_title('Perturbation coverage vs. hypothesis quality per mask')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)

    legend_handles = [
        mpatches.Patch(color='steelblue',  label='h-stripe'),
        mpatches.Patch(color='darkorange', label='v-stripe'),
        mpatches.Patch(color='green',      label='quadrant'),
    ]
    ax.legend(handles=legend_handles, fontsize=8)


# ---------------------------------------------------------------------------
# Panel 4 — Crop + perturbation heatmap for a single frame
# ---------------------------------------------------------------------------

def plot_crop_analysis(d, frame_t, out_dir, video_name):
    """
    For frame index t in the log, produce a figure with:
      Row 0 : clean x_crop | perturbation magnitude heatmap | overlay
      Row 1+ : grid of mask patterns with hypothesis centre marked
    Saved separately as crop_analysis_frame<T>.png
    """
    x_crop     = d['attack_x_crop'][frame_t]    # (H, W, 3) uint8
    att_per    = d['attack_att_per'][frame_t]   # (H, W, 3) float32
    masks_np   = d['masks']                     # (M, H, W) bool
    mask_names = d['mask_names']
    hyp_bboxes = d['attack_mask_bboxes'][frame_t]  # (M, 4)
    hyp_iou    = d['attack_mask_iou_gt'][frame_t]  # (M,)
    frame_idx  = int(d['attack_frame_idxs'][frame_t])
    M          = len(masks_np)
    crop_h, crop_w = x_crop.shape[:2]

    # Perturbation magnitude (mean abs across channels)
    att_mag = np.abs(att_per).mean(axis=2)   # (H, W)

    # Grid layout: top row = 3 panels; remaining rows = mask grid
    ncols_mask = min(M, 5)
    nrows_mask = (M + ncols_mask - 1) // ncols_mask
    nrows_total = 1 + nrows_mask
    fig, axes = plt.subplots(nrows_total, max(3, ncols_mask),
                             figsize=(max(3, ncols_mask)*2.2, nrows_total*2.2))
    fig.suptitle(f'Crop analysis — frame {frame_idx}', fontsize=11)

    # Top row: crop, perturbation heatmap, overlay
    ax_crop   = axes[0, 0]
    ax_heat   = axes[0, 1]
    ax_overlay= axes[0, 2]

    ax_crop.imshow(cv2.cvtColor(x_crop, cv2.COLOR_BGR2RGB))
    ax_crop.set_title('Clean x_crop')
    ax_crop.axis('off')

    im_heat = ax_heat.imshow(att_mag, cmap='hot', interpolation='nearest')
    ax_heat.set_title('|att_per| heatmap')
    ax_heat.axis('off')
    plt.colorbar(im_heat, ax=ax_heat, fraction=0.046, pad=0.04)

    overlay = x_crop.astype(np.float32).copy()
    heat_norm = (att_mag - att_mag.min()) / (att_mag.max() - att_mag.min() + 1e-8)
    heat_rgb = plt.cm.hot(heat_norm)[..., :3] * 255
    alpha = 0.5
    overlay = np.clip(overlay * (1-alpha) + heat_rgb * alpha, 0, 255).astype(np.uint8)
    ax_overlay.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax_overlay.set_title('Crop + pert. overlay')
    ax_overlay.axis('off')

    # Hide unused top-row axes
    for c in range(3, max(3, ncols_mask)):
        axes[0, c].axis('off')

    # Mask grid rows
    for mi in range(M):
        row = 1 + mi // ncols_mask
        col = mi % ncols_mask
        ax  = axes[row, col]

        # Show mask as greyscale pattern
        mask_vis = masks_np[mi].astype(np.float32)   # 1=keep, 0=masked
        ax.imshow(mask_vis, cmap='gray', vmin=0, vmax=1, interpolation='nearest')

        # Draw hypothesis centre
        bx, by, bw, bh = hyp_bboxes[mi]
        # Note: bbox is in FRAME coordinates, not crop coordinates.
        # For visualisation inside the crop, we cannot directly overlay
        # frame-coord bboxes without the frame→crop mapping.
        # Instead, annotate IoU-with-GT in title.
        name = str(mask_names[mi])
        iou  = float(hyp_iou[mi]) if not np.isnan(hyp_iou[mi]) else 0.
        ax.set_title(f'{name}\niou={iou:.2f}', fontsize=6)
        ax.axis('off')

    # Hide unused mask grid axes
    for mi in range(M, nrows_mask * ncols_mask):
        row = 1 + mi // ncols_mask
        col = mi % ncols_mask
        if row < nrows_total and col < max(3, ncols_mask):
            axes[row, col].axis('off')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'crop_analysis_{video_name}_frame{frame_idx}.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Analyse masked hypothesis experiment output'
    )
    parser.add_argument('log', help='Path to log_masked_<video>.npz')
    parser.add_argument('--frame', type=int, default=None,
                        help='Frame index (in log, 0-based) for single-frame panels')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively (requires display)')
    args = parser.parse_args()

    if args.show:
        matplotlib.use('TkAgg')

    d         = _load(args.log)
    NF        = len(d['attack_frame_idxs'])
    out_dir   = os.path.dirname(args.log)
    video_name = os.path.basename(args.log).replace('log_masked_', '').replace('.npz', '')
    all_frames = list(range(NF))
    sel_frame  = args.frame if args.frame is not None else 0

    print(f"Loaded: {args.log}")
    print(f"  Frames: {NF}  Masks: {d['masks'].shape[0]}  "
          f"Crop: {d['masks'].shape[1]}x{d['masks'].shape[2]}")
    best_mask_iou_str    = [f"{np.nanmax(d['attack_mask_iou_gt'][t]):.2f}" for t in all_frames]
    best_cluster_iou_str = [f"{d['attack_cluster_iou_gt'][t,0]:.2f}" for t in all_frames]
    print(f"  attack_mask_iou_gt best per frame:    {best_mask_iou_str}")
    print(f"  attack_cluster_iou_gt rank-1 per frame: {best_cluster_iou_str}")

    # ── Figure 1: candidate + reranking quality ──────────────────────────────
    nf = min(NF, 5)
    fig, axes = plt.subplots(2 + nf, 2, figsize=(12, 4*(2+nf)//2))
    # Row 0-1: candidate quality (boxplots)
    plot_candidate_quality(d, axes[0, 0], axes[0, 1], all_frames)
    # Row 1: perturbation coverage
    plot_perturbation_coverage(d, axes[1, 0])
    axes[1, 1].axis('off')
    # Rows 2+: reranking scatter, one per frame
    scatter_axes = []
    for ti in range(nf):
        row = 2 + ti // 2
        col = ti % 2
        if row < axes.shape[0]:
            scatter_axes.append(axes[row, col])
    # Fill remaining with off
    for r in range(2, axes.shape[0]):
        for c in range(2):
            pass  # drawn by plot_reranking_quality below or turned off

    # Rebuild a cleaner layout for reranking
    fig2, ax2 = plt.subplots(1, nf, figsize=(5*nf, 4.5))
    if nf == 1:
        ax2 = [ax2]
    plot_reranking_quality(d, ax2, list(range(nf)))
    fig2.suptitle(f'Reranking quality — {video_name}', fontsize=12)
    fig2.tight_layout()
    fig2_path = os.path.join(out_dir, f'reranking_{video_name}.png')
    fig2.savefig(fig2_path, dpi=120, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved → {fig2_path}")

    # ── Figure 1 (candidate + coverage) ──────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    plot_candidate_quality(d, axes3[0], axes3[1], all_frames)
    plot_perturbation_coverage(d, axes3[2])
    fig3.suptitle(f'Candidate quality & coverage — {video_name}', fontsize=12)
    fig3.tight_layout()
    fig3_path = os.path.join(out_dir, f'candidate_quality_{video_name}.png')
    fig3.savefig(fig3_path, dpi=120, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved → {fig3_path}")

    # ── Figure 4: crop analysis for selected frame ────────────────────────────
    plot_crop_analysis(d, sel_frame, out_dir, video_name)

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
