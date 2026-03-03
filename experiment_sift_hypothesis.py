#!/usr/bin/env python3
"""
experiment_sift_hypothesis.py

Validates the hypothesis:
    Under adversarial hijacking, GEOMETRY (match_ratio, inlier_ratio,
    kp_stability) within the GT bounding box stays stable, while
    APPEARANCE (descriptor_sim) degrades.

Two experiment modes
--------------------
  benign  – clean DaSiamRPN tracking, SIFT computed on unperturbed frames
  attack  – RTAA-attacked tracking,   SIFT computed on the same frames

Important note on digital vs. physical attacks
-----------------------------------------------
RTAA is a DIGITAL attack: it perturbs the internal search-region tensor
only.  The raw video pixels are unchanged, so SIFT scores computed on those
pixels will look similar in both modes.  For digital attacks the key
observable is tracking_error — the attacked tracker is fooled even though
the visible scene is unmodified.

For PHYSICAL attacks (a printed patch visible in the camera feed), the
frames themselves differ between benign and attacked videos.  In that case:
  • Run --mode benign  on the patch-free video
  • Run --mode attack  on the video with the patch present
  (supply each with its own image directory via the JSON or a custom loader)
The SIFT appearance scores will then genuinely diverge across conditions.

Saved outputs (per run)
-----------------------
  <out_dir>/<mode>_<video>.npz  with arrays indexed by frame t:
    geo_scores[t]     float  0.4·match_ratio + 0.4·inlier_ratio + 0.2·kp_stability
    app_scores[t]     float  descriptor_sim  (appearance proxy)
    match_ratios[t]   float  raw sub-score
    inlier_ratios[t]  float  raw sub-score
    kp_stabilities[t] float  raw sub-score
    track_errors[t]   float  L2 distance between predicted and GT bbox centre
    n_kp_prev[t]      int    keypoints in GT region at frame t-1
    n_kp_curr[t]      int    keypoints in GT region at frame t
    Note: index 0 is NaN (no previous frame to compare against).

Usage
-----
  # Single benign run
  python experiment_sift_hypothesis.py --dataset VOT2018 --video car1 --mode benign

  # Single attack run
  python experiment_sift_hypothesis.py --dataset VOT2018 --video car1 --mode attack

  # Comparison plot from two saved .npz files
  python experiment_sift_hypothesis.py \\
      --compare out/sift_experiment/benign_car1.npz out/sift_experiment/attack_car1.npz \\
      --labels Benign Attack

  # Run all videos in a dataset
  python experiment_sift_hypothesis.py --dataset VOT2018 --mode benign
"""

import argparse
import json
import os
from os.path import realpath, dirname, join

import cv2
import matplotlib
matplotlib.use('Agg')   # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

from net import SiamRPNvot
from run_attack import (
    SiamRPN_init,
    SiamRPN_track as siamrpn_attack_track,
    tracker_eval,
)
from utils import rect_2_cxy_wh, cxy_wh_2_rect, get_subwindow_tracking
from sift_alignment import SIFTAlignmentDetector


# ---------------------------------------------------------------------------
# SIFT ROI scoring
# ---------------------------------------------------------------------------

def roi_sift_scores(detector, frame1, frame2, gt_bbox):
    """
    Compute geometric and appearance sub-scores restricted to the GT bbox.

    All scores are in [0, 1], higher = more stable / more similar.

    Parameters
    ----------
    detector : SIFTAlignmentDetector
    frame1   : BGR uint8 ndarray — previous frame
    frame2   : BGR uint8 ndarray — current frame
    gt_bbox  : array-like [x, y, w, h] in pixel coords (floats OK)

    Returns  (all floats unless noted)
    -------
    geo_score   : 0.4·match_ratio + 0.4·inlier_ratio + 0.2·kp_stability
    app_score   : descriptor_sim
    match_ratio : fraction of keypoints with a good Lowe-test match
    inlier_ratio: RANSAC homography inlier fraction (geometric consistency)
    kp_stability: 1 − normalised |ΔN_keypoints|
    n_kp1       : int  keypoint count in frame1's GT region
    n_kp2       : int  keypoint count in frame2's GT region
    """
    h, w = frame1.shape[:2]
    x, y, bw, bh = (int(v) for v in gt_bbox)
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + bw, w), min(y + bh, h)

    if x2 <= x1 or y2 <= y1:       # degenerate box
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    kp1, desc1 = detector.extract_features(frame1, mask=mask)
    kp2, desc2 = detector.extract_features(frame2, mask=mask)

    good = detector._ratio_match(desc1, desc2)
    n1, n2 = len(kp1), len(kp2)
    n_kps = max(n1, n2, 1)

    match_r  = float(min(len(good) / n_kps, 1.0))
    inlier_r = float(detector._ransac_inlier_ratio(kp1, kp2, good))
    app      = float(detector._descriptor_similarity(good))
    kp_stab  = float(1.0 - np.clip(abs(n2 - n1) / max(n1, 1), 0.0, 1.0))
    geo      = float(np.clip(0.40 * match_r + 0.40 * inlier_r + 0.20 * kp_stab, 0, 1))

    return geo, app, match_r, inlier_r, kp_stab, n1, n2


# ---------------------------------------------------------------------------
# Tracker helpers
# ---------------------------------------------------------------------------

def _center(bbox):
    """Return (cx, cy) of a [x, y, w, h] bbox."""
    return bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0


def center_distance(pred_bbox, gt_bbox):
    """L2 distance between the centres of two [x, y, w, h] bounding boxes."""
    px, py = _center(pred_bbox)
    gx, gy = _center(gt_bbox)
    return float(np.sqrt((px - gx) ** 2 + (py - gy) ** 2))


def track_clean(state, im):
    """
    One step of clean DaSiamRPN tracking — no attack, no defense.

    Replicates the search-crop extraction and tracker_eval call from
    SiamRPN_track in run_attack.py, but without the rtaa_attack step.
    """
    p          = state['p']
    net        = state['net']
    avg_chans  = state['avg_chans']
    window     = state['window']
    target_pos = state['target_pos']
    target_sz  = state['target_sz']

    wc_z    = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z    = target_sz[0] + p.context_amount * sum(target_sz)
    s_z     = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    pad     = (p.instance_size - p.exemplar_size) / 2 / scale_z
    s_x     = s_z + 2 * pad

    x_crop = Variable(
        get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)
        .unsqueeze(0)
    ).cuda()

    # tracker_eval clamps using state['im_w'] / state['im_h'];
    # the f and gt parameters exist in the signature but are unused inside.
    target_pos, target_sz, score = tracker_eval(
        net, x_crop, target_pos, target_sz * scale_z,
        window, scale_z, p, 0, None, state
    )

    state['target_pos'] = target_pos
    state['target_sz']  = target_sz
    state['score']      = score
    return state


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(image_files, gt, net, mode, detector, out_dir, video_name):
    """
    Run the full SIFT experiment on one video sequence and save results.

    Parameters
    ----------
    image_files : list[str]
    gt          : ndarray shape (T, 4)  — GT boxes [x, y, w, h]
    net         : DaSiamRPN network (eval, CUDA)
    mode        : 'benign' | 'attack'
    detector    : SIFTAlignmentDetector
    out_dir     : str
    video_name  : str

    Returns
    -------
    str  path to the saved .npz file
    """
    T = min(len(image_files), len(gt))
    assert T >= 2, "Need at least 2 frames"

    # NaN → "not computed" (frame 0 has no predecessor)
    geo_scores    = np.full(T, np.nan)
    app_scores    = np.full(T, np.nan)
    match_ratios  = np.full(T, np.nan)
    inlier_ratios = np.full(T, np.nan)
    kp_stabilities = np.full(T, np.nan)
    track_errors  = np.full(T, np.nan)
    n_kp_prev     = np.zeros(T, dtype=int)
    n_kp_curr     = np.zeros(T, dtype=int)

    att_per       = 0        # accumulated RTAA perturbation (attack mode)
    final_pos     = None     # retarget destination  (attack mode)
    state         = None
    prev_frame    = None
    pred_locations = []      # tracker's predicted [x,y,w,h] per frame

    for f, image_file in enumerate(tqdm(image_files[:T], desc=f"[{mode}] {video_name}")):
        im = cv2.imread(image_file)
        if im is None:
            print(f"  Warning: could not read {image_file}")
            break

        # ------------------------------------------------------------------
        # Frame 0: initialise tracker
        # ------------------------------------------------------------------
        if f == 0:
            target_pos, target_sz = rect_2_cxy_wh(gt[f])
            state = SiamRPN_init(im, target_pos, target_sz, net)

            if mode == 'attack':
                # Choose retarget direction (mirrors test_hijack_attack.py)
                dx = -200 if (target_pos[0] + target_sz[0] / 2) > im.shape[1] / 2 else 200
                final_pos = [
                    target_pos[0] + dx, target_pos[1],
                    float(target_sz[0]), float(target_sz[1]),
                ]

            init_location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            pred_locations.append(init_location)
            track_errors[0] = center_distance(init_location, gt[0])

            prev_frame = im
            continue

        # ------------------------------------------------------------------
        # Frames 1 … T-1
        # ------------------------------------------------------------------

        # 1. SIFT scores within GT bbox (always on unperturbed frames)
        geo, app, mr, ir, ks, n1, n2 = roi_sift_scores(
            detector, prev_frame, im, gt[f]
        )
        geo_scores[f]     = geo
        app_scores[f]     = app
        match_ratios[f]   = mr
        inlier_ratios[f]  = ir
        kp_stabilities[f] = ks
        n_kp_prev[f]      = n1
        n_kp_curr[f]      = n2

        # 2. Tracking step
        if mode == 'benign':
            state = track_clean(state, im)

        else:   # attack
            # Reset perturbation every 30 frames (mirrors test_hijack_attack.py)
            if f % 30 == 1:
                att_per = 0
            im_bounds = [im.shape[1], im.shape[0]]
            # last_result = previous prediction (not GT), matching the
            # convention in test_hijack_attack.py where regions[f-1] is used
            state, att_per, _ = siamrpn_attack_track(
                state, im, f, pred_locations[f - 1],
                att_per, 0,                     # def_per = 0 (no defense)
                image_save=0, iter=5,
                final_pos=final_pos,
                im_bounds=im_bounds,
            )

        pred_location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        pred_locations.append(pred_location)
        track_errors[f] = center_distance(pred_location, gt[f])

        prev_frame = im

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    save_path = join(out_dir, f"{mode}_{video_name}.npz")
    np.savez(
        save_path,
        geo_scores=geo_scores,
        app_scores=app_scores,
        match_ratios=match_ratios,
        inlier_ratios=inlier_ratios,
        kp_stabilities=kp_stabilities,
        track_errors=track_errors,
        n_kp_prev=n_kp_prev,
        n_kp_curr=n_kp_curr,
    )
    print(f"  Saved → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COLORS  = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
_LSTYLES = ['-', '--', '-.', ':']


def _add_panel(ax, frames, data, ylabel, color, linestyle, label, ylim=(-.05, 1.05)):
    ax.plot(frames, data, color=color, ls=linestyle, lw=1.3, label=label)
    ax.set_ylabel(ylabel, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_single(npz_path, video_name, mode, save_dir):
    """
    3-panel figure: geo score / app score / tracking error for one run.
    An optional 4th panel breaks out the three geometric sub-scores.
    """
    d      = np.load(npz_path)
    frames = np.arange(len(d['geo_scores']))
    color  = _COLORS[0] if mode == 'benign' else _COLORS[1]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"{video_name}  |  mode = {mode}", fontsize=12, fontweight='bold')

    _add_panel(axes[0], frames, d['geo_scores'],
               "Geometric Score\n(match+inlier+kp_stab)", color, '-', 'geo_score')

    _add_panel(axes[1], frames, d['app_scores'],
               "Appearance Score\n(descriptor_sim)", color, '-', 'app_score')

    # Sub-scores breakdown
    axes[2].plot(frames, d['match_ratios'],   color='#1f77b4', lw=1.0, label='match_ratio')
    axes[2].plot(frames, d['inlier_ratios'],  color='#2ca02c', lw=1.0, label='inlier_ratio')
    axes[2].plot(frames, d['kp_stabilities'], color='#9467bd', lw=1.0, label='kp_stability')
    axes[2].set_ylabel("Geometric Sub-scores", fontsize=9)
    axes[2].set_ylim(-.05, 1.05)
    axes[2].legend(fontsize=8, loc='upper right')
    axes[2].grid(True, alpha=0.3)

    _add_panel(axes[3], frames, d['track_errors'],
               "Tracking Error\n(centre dist, px)", color, '-', 'track_error', ylim=None)
    axes[3].set_xlabel("Frame", fontsize=9)

    plt.tight_layout()
    out = join(save_dir, f"plot_{mode}_{video_name}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot → {out}")


def plot_compare(npz_paths, labels, save_path):
    """
    Overlay two (or more) runs in a single 3-panel figure.

    Panel 1: geo_score  — HYPOTHESIS: lines should be close (geometry invariant)
    Panel 2: app_score  — HYPOTHESIS: attack line should be lower (appearance degrades)
    Panel 3: track_error — shows which frames the tracker is fooled
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(
        "Geometry vs. Appearance under Adversarial Hijacking\n"
        "(SIFT scores within GT bounding box)",
        fontsize=12, fontweight='bold'
    )

    for i, (path, label) in enumerate(zip(npz_paths, labels)):
        d      = np.load(path)
        frames = np.arange(len(d['geo_scores']))
        c, ls  = _COLORS[i % len(_COLORS)], _LSTYLES[i % len(_LSTYLES)]
        axes[0].plot(frames, d['geo_scores'],   color=c, ls=ls, lw=1.5, label=label)
        axes[1].plot(frames, d['app_scores'],   color=c, ls=ls, lw=1.5, label=label)
        axes[2].plot(frames, d['track_errors'], color=c, ls=ls, lw=1.5, label=label)

    axes[0].set_ylabel("Geometric Score\n(match_ratio + inlier_ratio + kp_stability)", fontsize=9)
    axes[0].set_ylim(-.05, 1.05)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        "Hypothesis: geometry stays stable — lines should overlap", fontsize=8, style='italic'
    )

    axes[1].set_ylabel("Appearance Score\n(descriptor_sim)", fontsize=9)
    axes[1].set_ylim(-.05, 1.05)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(
        "Hypothesis: appearance degrades under attack — attack line should be lower", fontsize=8, style='italic'
    )

    axes[2].set_ylabel("Tracking Error  (px)", fontsize=9)
    axes[2].set_xlabel("Frame", fontsize=9)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title(
        "High tracking error = tracker is fooled (identifies attacked frames)", fontsize=8, style='italic'
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison plot → {save_path}")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_name, video_filter=None):
    base_path = join(realpath(dirname(__file__)), 'data', dataset_name)
    json_path = join(realpath(dirname(__file__)), 'data', dataset_name + '.json')
    info = json.load(open(json_path))
    for v in info.keys():
        name = info[v]['name']
        if video_filter and name != video_filter:
            continue
        info[v]['image_files'] = [
            join(base_path, name, 'img', f) for f in info[v]['image_files']
        ]
        info[v]['gt']   = np.array(info[v]['gt'])
        info[v]['name'] = v
    return {k: v for k, v in info.items()
            if not video_filter or v['name'] == video_filter}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SIFT alignment experiment: geometry vs. appearance under attack"
    )
    parser.add_argument('--dataset', default='VOT2018',
                        help='Dataset name (must have a matching .json in data/)')
    parser.add_argument('--video', default=None,
                        help='Single video name to process (default: all)')
    parser.add_argument('--mode', choices=['benign', 'attack'], default='benign',
                        help='benign = clean tracker | attack = RTAA-attacked tracker')
    parser.add_argument('--out_dir', default='out/sift_experiment',
                        help='Directory for .npz results and plots')
    parser.add_argument('--model', default='SiamRPNvot.model',
                        help='Filename of the DaSiamRPN .model checkpoint')
    parser.add_argument('--compare', nargs='+', metavar='NPZ',
                        help='Skip experiment; compare two or more saved .npz files')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Legend labels for --compare (default: filenames)')
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Comparison-only mode (no network needed)
    # ---------------------------------------------------------------
    if args.compare:
        labels    = args.labels or [os.path.basename(p) for p in args.compare]
        save_path = join(args.out_dir, 'comparison.png')
        os.makedirs(args.out_dir, exist_ok=True)
        plot_compare(args.compare, labels, save_path)
        return

    # ---------------------------------------------------------------
    # Experiment mode
    # ---------------------------------------------------------------
    net = SiamRPNvot()
    net.load_state_dict(torch.load(join(realpath(dirname(__file__)), args.model)))
    net.eval().cuda()

    detector = SIFTAlignmentDetector()
    dataset  = load_dataset(args.dataset, video_filter=args.video)

    if not dataset:
        print(f"No videos found for dataset={args.dataset} video={args.video}")
        return

    for video in dataset.values():
        print(f"\n{'='*60}")
        print(f"Video: {video['name']}   frames: {len(video['image_files'])}   mode: {args.mode}")
        npz_path = run_experiment(
            image_files=video['image_files'],
            gt=video['gt'],
            net=net,
            mode=args.mode,
            detector=detector,
            out_dir=args.out_dir,
            video_name=video['name'],
        )
        plot_single(npz_path, video['name'], args.mode, args.out_dir)


if __name__ == '__main__':
    main()
