#!/usr/bin/env python3
"""
experiment_sift_attack.py

Minimal experiment for studying the SIFT-signal vulnerability surface under
joint tracker + SIFT-evasion attack.

Compares three attack variants head-to-head, with NO defense in the loop:
  --attack none            : clean tracker (baseline)
  --attack rtaa            : original RTAA (tracker hijack only)
  --attack rtaa_sift_crop  : RTAA + DoG-suppress in 271x271 crop space
                             (legacy parameterisation; high-freq attack signal
                              gets low-pass-filtered by the inject upsample)
  --attack rtaa_sift_frame : RTAA + DoG-suppress at frame (s_x x s_x) resolution
                             with differentiable bilinear downsample for the
                             tracker forward; matches the coord system SIFT
                             actually sees.

Per frame we log:
  - tracker prediction bbox, IoU vs GT
  - SIFT keypoint count inside the R_target ROI for both clean `im` and
    rendered `im_attacked`  ==>  removal_rate = 1 - kp_attacked / kp_clean
  - per-iter L_rtaa / L_dog / L_kornia / L_total traces

Run:
  python experiment_sift_attack.py --video car1 --attack rtaa_sift_frame --eps 16
  python experiment_sift_attack.py --video car1 --attack rtaa_sift_frame \
      --rtaa_weight 0 --eps 16 --n_iter 20 --alpha_dog 1000   # DoG-only ablation
"""

import argparse
import json
import os
import random
from os.path import realpath, dirname, join

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from net import SiamRPNvot
from run_attack import (
    SiamRPN_init,
    rtaa_attack,
    rtaa_sift_attack,
    rtaa_sift_attack_frame,
)
from utils import rect_2_cxy_wh, cxy_wh_2_rect, get_subwindow_tracking
from sift_alignment import SIFTAlignmentDetector
from sift_attack import frame_bbox_to_crop_bbox


# ---------------------------------------------------------------------------
# Search-region helpers
# ---------------------------------------------------------------------------

def _search_region_params(state):
    """(scale_z, s_x_float) — same math as the tracker uses internally."""
    p, tsz = state['p'], state['target_sz']
    wc_z = tsz[1] + p.context_amount * sum(tsz)
    hc_z = tsz[0] + p.context_amount * sum(tsz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    pad = ((p.instance_size - p.exemplar_size) / 2) / scale_z
    return float(scale_z), float(s_z + 2 * pad)


def _extract_x_crop_271(state, im):
    """Network-input crop (3, 271, 271) — the version the tracker sees."""
    p = state['p']
    scale_z, s_x = _search_region_params(state)
    x_crop = get_subwindow_tracking(
        im, state['target_pos'], p.instance_size, round(s_x), state['avg_chans']
    ).unsqueeze(0)
    return Variable(x_crop).cuda(), scale_z, s_x


def _extract_x_crop_frame(state, im, s_x_int):
    """Same crop content as _extract_x_crop_271 but at native frame resolution
    (skips the cv2.resize down to 271 by passing model_sz == original_sz)."""
    crop = get_subwindow_tracking(
        im, state['target_pos'], s_x_int, s_x_int, state['avg_chans']
    ).unsqueeze(0)
    return Variable(crop).cuda()


def frame_bbox_to_crop_bbox_frame_res(bbox_frame, target_pos, s_x_int):
    """Map [x,y,w,h] from frame coords into the s_x_int x s_x_int crop's pixel
    coords, mirroring get_subwindow_tracking's crop origin exactly."""
    c = (s_x_int + 1) / 2
    sx_left = round(target_pos[0] - c)
    sy_top  = round(target_pos[1] - c)
    fx, fy, fw, fh = (float(v) for v in bbox_frame)
    cx = fx - sx_left
    cy = fy - sy_top
    cx0 = max(cx, 0.0); cy0 = max(cy, 0.0)
    cx1 = min(cx + fw, float(s_x_int))
    cy1 = min(cy + fh, float(s_x_int))
    return [cx0, cy0, max(cx1 - cx0, 0.0), max(cy1 - cy0, 0.0)]


# ---------------------------------------------------------------------------
# Network forward — argmax-pscore decode, updates state.
# ---------------------------------------------------------------------------

def run_siamrpn_forward(net, x_271, state, scale_z):
    p          = state['p']
    target_pos = state['target_pos']
    target_sz  = state['target_sz']
    window     = state['window']

    with torch.no_grad():
        d_out, s_out = net(x_271)
    delta = d_out.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(
        s_out.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0
    ).data[1, :].cpu().numpy()

    delta[0] = delta[0] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1] = delta[1] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2] = np.exp(delta[2]) * p.anchor[:, 2]
    delta[3] = np.exp(delta[3]) * p.anchor[:, 3]

    tsz = target_sz * scale_z

    def _chg(r):    return np.maximum(r, 1. / r)
    def _sz(w, h):  pad = (w + h) * .5; return np.sqrt((w + pad) * (h + pad))
    def _sz_wh(wh): pad = (wh[0] + wh[1]) * .5; return np.sqrt((wh[0] + pad) * (wh[1] + pad))

    s_c = _chg(_sz(delta[2], delta[3]) / _sz_wh(tsz))
    r_c = _chg((tsz[0] / tsz[1]) / (delta[2] / delta[3]))
    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore  = penalty * score
    pscore  = pscore * (1 - p.window_influence) + window * p.window_influence

    best = int(np.argmax(pscore))
    t    = delta[:, best] / scale_z
    lr   = penalty[best] * score[best] * p.lr
    x = float(np.clip(t[0] + target_pos[0], 0, state['im_w']))
    y = float(np.clip(t[1] + target_pos[1], 0, state['im_h']))
    w = float(np.clip(target_sz[0] * (1 - lr) + t[2] * lr, 10, state['im_w']))
    h = float(np.clip(target_sz[1] * (1 - lr) + t[3] * lr, 10, state['im_h']))
    state['target_pos'] = np.array([x, y])
    state['target_sz']  = np.array([w, h])
    state['score']      = float(score[best])
    return np.array([x - w / 2, y - h / 2, w, h])


# ---------------------------------------------------------------------------
# Render perturbation into the full frame.
# ---------------------------------------------------------------------------

def _paste_into_frame(im, patch_chw, target_pos, s_x_int):
    """Paste a (3, s_x_int, s_x_int) numpy float patch onto `im` centered at
    target_pos. Patch is ADDED (not replaced) and clipped to [0, 255]."""
    h, w = im.shape[:2]
    cx, cy = int(round(target_pos[0])), int(round(target_pos[1]))
    x1 = cx - s_x_int // 2
    y1 = cy - s_x_int // 2
    sx1, sy1 = max(0, -x1), max(0, -y1)
    sx2 = s_x_int - max(0, x1 + s_x_int - w)
    sy2 = s_x_int - max(0, y1 + s_x_int - h)
    dx1, dy1 = max(0, x1), max(0, y1)
    dx2, dy2 = min(w, x1 + s_x_int), min(h, y1 + s_x_int)
    im_f = im.astype(np.float32).copy()
    if dx2 > dx1 and dy2 > dy1:
        patch_hwc = np.transpose(patch_chw, (1, 2, 0))
        im_f[dy1:dy2, dx1:dx2] += patch_hwc[sy1:sy2, sx1:sx2]
    return np.clip(im_f, 0, 255).astype(np.uint8)


def inject_frame_res(im, delta_frame, target_pos, s_x_int):
    """delta_frame : torch (1, 3, s_x_int, s_x_int) — frame-res perturbation."""
    d = delta_frame.detach().squeeze(0).cpu().numpy().astype(np.float32)
    return _paste_into_frame(im, d, target_pos, s_x_int)


def inject_crop_res(im, att_271, target_pos, s_x_int):
    """att_271 : torch (1, 3, 271, 271) — legacy crop-space perturbation,
    rendered by cv2.resize up to s_x_int. The low-pass step that motivates
    the frame-res variant."""
    a = att_271.detach().squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    a_up = cv2.resize(a, (s_x_int, s_x_int))                # (s_x_int, s_x_int, 3)
    return _paste_into_frame(im, np.transpose(a_up, (2, 0, 1)),
                             target_pos, s_x_int)


# ---------------------------------------------------------------------------
# SIFT keypoint counting inside an ROI.
# ---------------------------------------------------------------------------

def _bbox_to_image_mask(frame_shape, bbox):
    H, W = frame_shape[:2]
    x, y, w, h = (int(round(v)) for v in bbox)
    m = np.zeros((H, W), dtype=np.uint8)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = max(0, min(W, x + w)), max(0, min(H, y + h))
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = 255
    return m


def count_kps_in_roi(detector, frame, bbox):
    mask = _bbox_to_image_mask(frame.shape, bbox)
    if mask.sum() == 0:
        return 0
    return len(detector.extract_features(frame, mask=mask)[0])


def build_kp_sparse_mask(detector, frame, r_target_frame_bbox,
                         target_pos, s_x_int, half_side=4):
    """Build a {0,1} spatial-sparsity mask for the gradient attack.

    Detects SIFT keypoints inside `r_target_frame_bbox` in the *clean* frame
    once, then turns on a (2*half_side)x(2*half_side) square around each kp
    (in s_x_int crop-relative coords). Pass this to `rtaa_sift_attack_frame`
    via `attack_mask_frame=` to force the gradient PGD to spend its budget
    only on the spatial support where keypoints actually exist -- the
    spatial-sparsity prior that Amerini's smoothing attack uses implicitly.

    Returns
    -------
    mask_t       : (1, 1, s_x_int, s_x_int) float cuda tensor (broadcasts
                   across the 3 colour channels of delta in the PGD loop)
    n_kps        : how many kps the mask is built around
    area_frac    : fraction of crop pixels with mask=1 (for logging)
    """
    full = _bbox_to_image_mask(frame.shape, r_target_frame_bbox)
    if full.sum() == 0:
        kps = []
    else:
        kps, _ = detector.extract_features(frame, mask=full)

    c = (s_x_int + 1) / 2
    sx_left = round(target_pos[0] - c)
    sy_top  = round(target_pos[1] - c)

    m = np.zeros((s_x_int, s_x_int), dtype=np.float32)
    for kp in kps:
        cx = int(round(kp.pt[0] - sx_left))
        cy = int(round(kp.pt[1] - sy_top))
        x0 = max(cx - half_side, 0); y0 = max(cy - half_side, 0)
        x1 = min(cx + half_side, s_x_int); y1 = min(cy + half_side, s_x_int)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 1.0

    mask_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).cuda()
    return mask_t, len(kps), float(m.sum() / m.size)


# ---------------------------------------------------------------------------
# Amerini-style iterative keypoint-targeted smoothing attack.
# ---------------------------------------------------------------------------
# Faithful baseline for the "vanilla keypoint attack" from
#   Amerini et al., "Counter-forensics of SIFT-based copy-move detection by
#   means of keypoint classification", EURASIP J. Image Video Process. 2013.
# This is the single weakest of their three attacks (smoothing-only, class-
# unaware) but the conceptually cleanest: no patch database, no per-kp
# optimization. Acts as a CEILING reference for "how many kps can be removed
# in R_target if we drop the L_inf budget and only optimize for kp removal."
#
# Differences vs the gradient SIFT attack (rtaa_sift_frame, rtaa_weight=0):
#   - per-keypoint (8x8 patch around each kp), not per-ROI uniform
#   - iterative: re-detect kps each iter so the attack handles its own
#     "smoothing creates new kps" failure mode (paper sec 4.3)
#   - no L_inf budget — only image-quality preservation (small kernel)
#   - no tracker hijack — pure SIFT suppression; the tracker forward on the
#     attacked frame is purely diagnostic
# ---------------------------------------------------------------------------

def amerini_smoothing_attack(frame_bgr, roi_bbox, detector,
                             sigma=0.7, gaussian_ksize=3,
                             patch_half=4, max_iter=40,
                             target_removal=1.0):
    """One-shot Amerini-style smoothing attack on the kps inside `roi_bbox`.

    Pseudocode (mirrors Algorithm 1 of the paper, single-attack variant):
        attacked = frame
        kps_init = SIFT(attacked) inside roi_bbox
        while iter < max_iter and removal_rate < target_removal:
            kps      = SIFT(attacked) inside roi_bbox
            blurred  = GaussianBlur(attacked, ksize, sigma)
            for kp in kps:
                paste blurred[patch around kp.xy] back into attacked
            iter += 1
        return attacked, ...

    Returns
    -------
    attacked     : uint8 HxWx3 BGR — modified frame
    n_iters_done : how many outer iterations were actually run
    kp_final     : kps inside roi_bbox at the end
    kp_init      : kps inside roi_bbox at the start (for removal_rate)
    """
    if gaussian_ksize % 2 == 0:
        gaussian_ksize += 1
    H, W = frame_bgr.shape[:2]

    def _kps_in_roi(frame):
        mask = _bbox_to_image_mask(frame.shape, roi_bbox)
        if mask.sum() == 0:
            return []
        kps, _ = detector.extract_features(frame, mask=mask)
        return kps

    kps_init = _kps_in_roi(frame_bgr)
    n_init   = max(len(kps_init), 1)
    kp_stop  = int(n_init * (1.0 - target_removal))

    attacked = frame_bgr.copy()
    n_iters  = 0
    for it in range(max_iter):
        kps = _kps_in_roi(attacked)
        if len(kps) <= kp_stop:
            break
        blurred = cv2.GaussianBlur(attacked, (gaussian_ksize, gaussian_ksize),
                                   sigma)
        for kp in kps:
            x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
            x0 = max(x - patch_half, 0)
            y0 = max(y - patch_half, 0)
            x1 = min(x + patch_half, W)
            y1 = min(y + patch_half, H)
            if x1 > x0 and y1 > y0:
                attacked[y0:y1, x0:x1] = blurred[y0:y1, x0:x1]
        n_iters = it + 1

    final_kps = _kps_in_roi(attacked)
    return attacked, n_iters, len(final_kps), len(kps_init)


# ---------------------------------------------------------------------------
# Dataset loading.
# ---------------------------------------------------------------------------

def load_video(dataset_name, video_name):
    json_path = join(realpath(dirname(__file__)), 'data', dataset_name + '.json')
    info = json.load(open(json_path))
    for v in info.values():
        if v['name'] == video_name:
            v['image_files'] = [
                join(realpath(dirname(__file__)), 'data', dataset_name, v['name'],
                     'img', f) for f in v['image_files']
            ]
            v['gt'] = np.array(v['gt'])
            return v
    raise ValueError(f"Video '{video_name}' not found in {dataset_name}")


def _bbox_iou(a, b):
    ax1, ay1 = a[0], a[1]; ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]; bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0., min(ax2, bx2) - max(ax1, bx1))
    ih = max(0., min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.


# ---------------------------------------------------------------------------
# Benign simulation (baseline).
# ---------------------------------------------------------------------------

def simulate_benign(net, image_files, gt, init_frame, sim_frames):
    im0 = cv2.imread(image_files[init_frame])
    tp, tsz = rect_2_cxy_wh(gt[init_frame])
    state = SiamRPN_init(im0, tp, tsz, net)

    log = []
    for f in tqdm(sim_frames, desc="Benign"):
        im = cv2.imread(image_files[f])
        x_271, scale_z, _ = _extract_x_crop_271(state, im)
        pred = run_siamrpn_forward(net, x_271, state, scale_z)
        log.append({
            'frame_idx': f,
            'pred_bbox': pred.copy(),
            'gt_bbox':   gt[f].copy(),
            'iou_gt':    _bbox_iou(pred, gt[f]),
        })
    return log


# ---------------------------------------------------------------------------
# Attack simulation.
# ---------------------------------------------------------------------------

def simulate_attack(net, detector, image_files, gt, init_frame, sim_frames, args):
    im0 = cv2.imread(image_files[init_frame])
    tp, tsz = rect_2_cxy_wh(gt[init_frame])
    state = SiamRPN_init(im0, tp, tsz, net)
    prev_pred_bbox = gt[init_frame].copy()  # seeds R_target on first attack frame

    # Hijack target (fixed across the window)
    cx0 = state['target_pos'][0]
    dx  = -200 if cx0 > im0.shape[1] / 2 else 200
    final_pos = [state['target_pos'][0] + dx, state['target_pos'][1],
                 float(state['target_sz'][0]), float(state['target_sz'][1])]

    log = []
    for i, f in enumerate(tqdm(sim_frames, desc="Attack")):
        im = cv2.imread(image_files[f])
        im_bounds = [im.shape[1], im.shape[0]]
        p = state['p']
        target_pos = state['target_pos']
        target_sz  = state['target_sz']
        scale_z, s_x = _search_region_params(state)
        s_x_int = int(round(s_x))

        x_271, _, _ = _extract_x_crop_271(state, im)

        # Choose the SIFT suppress ROI
        r_target_frame = gt[f] if args.roi_source == 'gt' else prev_pred_bbox

        loss_log = {}

        if args.attack == 'none':
            x_for_tracker = x_271
            delta_frame_t = None

        elif args.attack == 'rtaa':
            x_for_tracker = rtaa_attack(
                net, x_271, x_271, prev_pred_bbox,
                target_pos, target_sz, scale_z, p,
                eps=args.eps, iteration=args.n_iter,
                final_pos=final_pos, im_bounds=im_bounds,
            )
            delta_frame_t = None

        elif args.attack == 'rtaa_sift_crop':
            r_target_271 = frame_bbox_to_crop_bbox(
                r_target_frame, target_pos, s_x, p.instance_size
            )
            x_for_tracker = rtaa_sift_attack(
                net, x_271, x_271, prev_pred_bbox,
                target_pos, target_sz, scale_z, p,
                r_target_crop_bbox=r_target_271,
                eps=args.eps, iteration=args.n_iter,
                final_pos=final_pos, im_bounds=im_bounds,
                alpha_dog=args.alpha_dog, gamma_kornia=args.gamma_kornia,
                dog_contrast=args.dog_contrast,
                loss_log=loss_log,
            )
            delta_frame_t = None

        elif args.attack == 'rtaa_sift_frame':
            x_frame_clean = _extract_x_crop_frame(state, im, s_x_int)
            r_target_crop = frame_bbox_to_crop_bbox_frame_res(
                r_target_frame, target_pos, s_x_int
            )
            sparse_mask_t = None
            if args.sparse_mask:
                sparse_mask_t, n_kp_for_mask, area_frac = build_kp_sparse_mask(
                    detector, im, r_target_frame, target_pos, s_x_int,
                    half_side=args.sparse_half_side,
                )
                loss_log['sparse_n_kps']     = [float(n_kp_for_mask)]
                loss_log['sparse_area_frac'] = [float(area_frac)]
            delta_frame_t, x_for_tracker = rtaa_sift_attack_frame(
                net, x_frame_clean, prev_pred_bbox,
                target_pos, target_sz, scale_z, p,
                r_target_crop_bbox=r_target_crop,
                eps=args.eps, iteration=args.n_iter,
                final_pos=final_pos, im_bounds=im_bounds,
                attack_mask_frame=sparse_mask_t,
                alpha_dog=args.alpha_dog, gamma_kornia=args.gamma_kornia,
                dog_contrast=args.dog_contrast,
                rtaa_weight=args.rtaa_weight,
                loss_log=loss_log,
            )

        elif args.attack == 'amerini_smoothing':
            # Modify the frame directly via iterative per-kp smoothing, then
            # re-extract the tracker's 271 crop from the modified frame so
            # the forward pass sees what SIFT also sees.
            im_attacked_pre, n_amerini, kp_amer_final, kp_amer_init = (
                amerini_smoothing_attack(
                    im, r_target_frame, detector,
                    sigma=args.amerini_sigma,
                    gaussian_ksize=args.amerini_ksize,
                    patch_half=args.amerini_patch_half,
                    max_iter=args.amerini_max_iter,
                    target_removal=args.amerini_target_removal,
                )
            )
            x_for_tracker = Variable(get_subwindow_tracking(
                im_attacked_pre, target_pos, p.instance_size,
                round(s_x), state['avg_chans']
            ).unsqueeze(0)).cuda()
            delta_frame_t = None
            loss_log['amerini_iters']    = [float(n_amerini)]
            loss_log['amerini_kp_init']  = [float(kp_amer_init)]
            loss_log['amerini_kp_final'] = [float(kp_amer_final)]

        else:
            raise ValueError(f"unknown --attack {args.attack}")

        # Render the attacked frame
        if args.attack == 'none':
            im_attacked = im.copy()
        elif args.attack == 'amerini_smoothing':
            im_attacked = im_attacked_pre
        elif args.attack == 'rtaa_sift_frame':
            im_attacked = inject_frame_res(im, delta_frame_t, target_pos, s_x_int)
        else:
            im_attacked = inject_crop_res(im, x_for_tracker - x_271,
                                          target_pos, s_x_int)

        # ─── Pixel-magnitude diagnostics, comparable to PGD's --eps ───
        # Computed uniformly over im_attacked - im for ANY attack variant
        # (skip 'none' since the delta is identically zero). Useful for
        # comparing the gradient and amerini attacks on the same axis as
        # the --eps L_inf budget.
        #
        #   perturbation_linf    = max |Δ| over all pixels/channels.
        #                          For PGD this should approach --eps when the
        #                          budget is fully spent; for amerini this is
        #                          the "implied eps" the smoothing happens to
        #                          use (usually much smaller).
        #   perturbation_l1_mean = mean |Δ| over pixels where any channel
        #                          changed. Per-edit magnitude inside the
        #                          attacked support.
        #   perturbation_frac    = fraction of frame pixels touched
        #                          (any channel changed). PGD on the dense
        #                          variants is ~100% within the s_x crop;
        #                          amerini and --sparse_mask runs are << 1.
        if args.attack != 'none':
            diff_int     = im_attacked.astype(np.int16) - im.astype(np.int16)
            abs_diff     = np.abs(diff_int)
            touched_mask = abs_diff.sum(axis=-1) > 0
            n_touched    = int(touched_mask.sum())
            loss_log['perturbation_linf']    = [float(abs_diff.max())]
            loss_log['perturbation_l1_mean'] = [
                float(abs_diff[touched_mask].mean()) if n_touched > 0 else 0.0
            ]
            loss_log['perturbation_frac']    = [
                float(n_touched) / float(touched_mask.size)
            ]

        pred_bbox = run_siamrpn_forward(net, x_for_tracker, state, scale_z)

        # SIFT removal rate inside R_target (the ROI we tried to suppress)
        kp_clean    = count_kps_in_roi(detector, im,          r_target_frame)
        kp_attacked = count_kps_in_roi(detector, im_attacked, r_target_frame)
        removal_rate = 1.0 - kp_attacked / max(kp_clean, 1)

        # Also count keypoints in the actual GT box (independent measurement)
        kp_gt_clean    = count_kps_in_roi(detector, im,          gt[f])
        kp_gt_attacked = count_kps_in_roi(detector, im_attacked, gt[f])
        removal_rate_gt = 1.0 - kp_gt_attacked / max(kp_gt_clean, 1)

        log.append({
            'frame_idx':       f,
            'pred_bbox':       pred_bbox.copy(),
            'gt_bbox':         gt[f].copy(),
            'r_target_bbox':   np.asarray(r_target_frame, dtype=np.float32),
            'iou_gt':          _bbox_iou(pred_bbox, gt[f]),
            'kp_clean':        int(kp_clean),
            'kp_attacked':     int(kp_attacked),
            'removal_rate':    float(removal_rate),
            'kp_gt_clean':     int(kp_gt_clean),
            'kp_gt_attacked':  int(kp_gt_attacked),
            'removal_rate_gt': float(removal_rate_gt),
            's_x_int':         int(s_x_int),
            'loss_log':        loss_log,
            'im_attacked':     im_attacked,        # kept for video rendering
        })

        prev_pred_bbox = pred_bbox.copy()

    return log


# ---------------------------------------------------------------------------
# Side-by-side video.
# ---------------------------------------------------------------------------

def _draw_bbox(frame, bbox, color, label=None, thickness=2):
    x, y, bw, bh = (int(round(v)) for v in bbox)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, thickness)
    if label:
        cv2.putText(frame, label, (x, max(y - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _draw_kps_in_roi(frame, detector, roi_bbox, color):
    mask = _bbox_to_image_mask(frame.shape, roi_bbox)
    if mask.sum() == 0:
        return
    kps, _ = detector.extract_features(frame, mask=mask)
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(frame, (x, y), 2, color, -1)


def render_video(detector, image_files, benign_log, attack_log, out_path,
                 attack_label):
    if not attack_log:
        return
    h, w = cv2.imread(image_files[attack_log[0]['frame_idx']]).shape[:2]
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (2 * w, h)
    )
    for b, a in zip(benign_log, attack_log):
        clean = cv2.imread(image_files[a['frame_idx']])
        attacked = a['im_attacked'].copy()
        clean_vis = clean.copy()

        _draw_kps_in_roi(clean_vis, detector, a['r_target_bbox'], (255, 255, 0))
        _draw_kps_in_roi(attacked,  detector, a['r_target_bbox'], (255, 255, 0))

        for img in (clean_vis, attacked):
            _draw_bbox(img, a['gt_bbox'],       (0, 200, 0), 'GT')
            _draw_bbox(img, a['r_target_bbox'], (200, 200, 0), 'R_target', 1)
        _draw_bbox(clean_vis, b['pred_bbox'], (200, 80, 0), 'Benign')
        _draw_bbox(attacked,  a['pred_bbox'], (0, 0, 220),  'Attack')

        hud = [
            f"Frame: {a['frame_idx']}   ({attack_label})",
            f"kp_clean (R)={a['kp_clean']}  kp_atk (R)={a['kp_attacked']}  "
            f"removal={a['removal_rate']:.2f}",
            f"kp_clean (GT)={a['kp_gt_clean']}  kp_atk (GT)={a['kp_gt_attacked']}  "
            f"removal_gt={a['removal_rate_gt']:.2f}",
            f"benign IoU={b['iou_gt']:.2f}   attack IoU={a['iou_gt']:.2f}",
        ]
        for i, line in enumerate(hud):
            y_pos = h - 8 - (len(hud) - 1 - i) * 20
            cv2.putText(attacked, line, (8, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(attacked, line, (8, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        side = np.concatenate([clean_vis, attacked], axis=1)
        writer.write(side)
    writer.release()


# ---------------------------------------------------------------------------
# Save + summary.
# ---------------------------------------------------------------------------

def save_log(out_path, benign_log, attack_log, args):
    NF = len(attack_log)
    def _stack(key):
        arr = np.full((NF, args.n_iter), np.nan, dtype=np.float32)
        for i, e in enumerate(attack_log):
            vals = e['loss_log'].get(key, []) if e['loss_log'] else []
            arr[i, :len(vals)] = vals
        return arr

    def _stack_scalar(key):
        """Per-frame scalar (loss_log[key] is a 1-element list)."""
        arr = np.full((NF,), np.nan, dtype=np.float32)
        for i, e in enumerate(attack_log):
            vals = e['loss_log'].get(key, []) if e['loss_log'] else []
            if vals:
                arr[i] = float(vals[0])
        return arr

    np.savez(
        out_path,
        attack_variant   = np.array(args.attack),
        eps              = np.array(args.eps, dtype=np.float32),
        n_iter           = np.array(args.n_iter, dtype=np.int32),
        alpha_dog        = np.array(args.alpha_dog, dtype=np.float32),
        gamma_kornia     = np.array(args.gamma_kornia, dtype=np.float32),
        dog_contrast     = np.array(args.dog_contrast, dtype=np.float32),
        rtaa_weight      = np.array(args.rtaa_weight, dtype=np.float32),
        roi_source       = np.array(args.roi_source),

        benign_frame_idxs  = np.array([e['frame_idx'] for e in benign_log]),
        benign_pred_bboxes = np.array([e['pred_bbox'] for e in benign_log]),
        benign_gt_bboxes   = np.array([e['gt_bbox']   for e in benign_log]),
        benign_iou_gt      = np.array([e['iou_gt']    for e in benign_log], dtype=np.float32),

        attack_frame_idxs       = np.array([e['frame_idx']        for e in attack_log]),
        attack_pred_bboxes      = np.array([e['pred_bbox']         for e in attack_log]),
        attack_gt_bboxes        = np.array([e['gt_bbox']           for e in attack_log]),
        attack_r_target_bboxes  = np.array([e['r_target_bbox']     for e in attack_log]),
        attack_iou_gt           = np.array([e['iou_gt']            for e in attack_log], dtype=np.float32),
        attack_kp_clean         = np.array([e['kp_clean']          for e in attack_log], dtype=np.int32),
        attack_kp_attacked      = np.array([e['kp_attacked']       for e in attack_log], dtype=np.int32),
        attack_removal_rate     = np.array([e['removal_rate']      for e in attack_log], dtype=np.float32),
        attack_kp_gt_clean      = np.array([e['kp_gt_clean']       for e in attack_log], dtype=np.int32),
        attack_kp_gt_attacked   = np.array([e['kp_gt_attacked']    for e in attack_log], dtype=np.int32),
        attack_removal_rate_gt  = np.array([e['removal_rate_gt']   for e in attack_log], dtype=np.float32),
        attack_s_x_int          = np.array([e['s_x_int']           for e in attack_log], dtype=np.int32),

        loss_rtaa   = _stack('L_rtaa'),
        loss_dog    = _stack('L_dog'),
        loss_kornia = _stack('L_kornia'),
        loss_total  = _stack('L_total'),

        # Per-frame pixel-magnitude diagnostics. Generic — populated for any
        # attack variant except 'none', so amerini and PGD variants can be
        # compared on the same axis as --eps.
        perturbation_linf    = _stack_scalar('perturbation_linf'),
        perturbation_l1_mean = _stack_scalar('perturbation_l1_mean'),
        perturbation_frac    = _stack_scalar('perturbation_frac'),

        # Per-frame Amerini-only diagnostics (NaN for other variants)
        amerini_iters    = _stack_scalar('amerini_iters'),
        amerini_kp_init  = _stack_scalar('amerini_kp_init'),
        amerini_kp_final = _stack_scalar('amerini_kp_final'),

        # Per-frame sparse-mask diagnostics (NaN unless --sparse_mask)
        sparse_n_kps     = _stack_scalar('sparse_n_kps'),
        sparse_area_frac = _stack_scalar('sparse_area_frac'),
    )


def print_summary(benign_log, attack_log, args):
    b_iou = np.mean([e['iou_gt'] for e in benign_log])
    a_iou = np.mean([e['iou_gt'] for e in attack_log])
    rr    = np.mean([e['removal_rate']    for e in attack_log])
    rr_gt = np.mean([e['removal_rate_gt'] for e in attack_log])

    print(f"\n=== Summary [{args.attack}] ===")
    print(f"  benign  mean IoU vs GT : {b_iou:.3f}")
    print(f"  attack  mean IoU vs GT : {a_iou:.3f}   "
          f"(lower => tracker hijacked more)")
    print(f"  removal_rate (R_target): {rr:.3f}     "
          f"(higher => SIFT signal suppressed)")
    print(f"  removal_rate (GT box)  : {rr_gt:.3f}")
    ll0 = attack_log[0]['loss_log'] if attack_log else {}
    if 'L_rtaa' in ll0 and ll0['L_rtaa']:
        rtaa0, rtaa1 = ll0['L_rtaa'][0],  ll0['L_rtaa'][-1]
        dog0,  dog1  = ll0['L_dog'][0],   ll0['L_dog'][-1]
        print(f"  frame 0 L_rtaa: {rtaa0:+.3f} -> {rtaa1:+.3f}   "
              f"L_dog: {dog0:.4f} -> {dog1:.4f}")
    if 'perturbation_linf' in ll0:
        mean_linf = np.mean([e['loss_log']['perturbation_linf'][0]
                             for e in attack_log])
        max_linf  = np.max ([e['loss_log']['perturbation_linf'][0]
                             for e in attack_log])
        mean_l1   = np.mean([e['loss_log']['perturbation_l1_mean'][0]
                             for e in attack_log])
        mean_cov  = np.mean([e['loss_log']['perturbation_frac'][0]
                             for e in attack_log])
        print(f"  perturbation magnitude (== implied L_inf eps):")
        print(f"    max |delta| per frame  : mean={mean_linf:.1f}  max={max_linf:.1f}")
        print(f"    mean |delta| in touched pixels: {mean_l1:.2f}")
        print(f"    fraction of frame pixels touched: {mean_cov * 100:.2f}%")
    if 'amerini_iters' in ll0:
        mean_iters = np.mean([e['loss_log']['amerini_iters'][0]
                              for e in attack_log])
        print(f"  amerini outer iters used (mean): "
              f"{mean_iters:.1f} / {args.amerini_max_iter}")
    if 'sparse_n_kps' in ll0:
        mean_n_kps = np.mean([e['loss_log']['sparse_n_kps'][0]     for e in attack_log])
        mean_area  = np.mean([e['loss_log']['sparse_area_frac'][0] for e in attack_log])
        print(f"  sparse mask: {mean_n_kps:.1f} kps (mean), "
              f"{mean_area * 100:.1f}% of crop area perturbable")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SIFT-signal vulnerability surface experiment"
    )
    parser.add_argument('--dataset',  default='VOT2018')
    parser.add_argument('--video',    required=True)
    parser.add_argument('--model',    default='SiamRPNvot.model')
    parser.add_argument('--out_dir',  default='out/sift_attack')
    parser.add_argument('--out_stem', default=None)
    parser.add_argument('--n_frames', type=int, default=10)
    parser.add_argument('--seed',     type=int, default=None)

    parser.add_argument('--attack', default='rtaa_sift_frame',
                        choices=['none', 'rtaa', 'rtaa_sift_crop',
                                 'rtaa_sift_frame', 'amerini_smoothing'])
    parser.add_argument('--eps',          type=float, default=16.0,
                        help='L_inf perturbation budget in pixel units')
    parser.add_argument('--n_iter',       type=int,   default=10)
    parser.add_argument('--alpha_dog',    type=float, default=1000.0)
    parser.add_argument('--gamma_kornia', type=float, default=0.0)
    parser.add_argument('--dog_contrast', type=float, default=0.04)
    parser.add_argument('--rtaa_weight',  type=float, default=1.0,
                        help='Weight on L_RTAA; set 0 to ablate DoG-only '
                             '(only honoured by --attack rtaa_sift_frame)')
    parser.add_argument('--roi_source', default='prev_pred',
                        choices=['gt', 'prev_pred'],
                        help='Where R_target (SIFT suppress region) comes from. '
                             'gt = oracle attacker; prev_pred = causal attacker.')
    parser.add_argument('--sparse_mask', action='store_true',
                        help='Constrain the rtaa_sift_frame perturbation to '
                             '(2*sparse_half_side)x(2*sparse_half_side) squares '
                             'around each SIFT kp detected in R_target on the '
                             'clean frame -- Amerini-style spatial sparsity '
                             'applied to the gradient attack. Only honoured by '
                             '--attack rtaa_sift_frame.')
    parser.add_argument('--sparse_half_side', type=int, default=4,
                        help='Half-side of the per-kp square in pixels '
                             '(default 4 matches Amerini 8x8 patches).')
    # --- Amerini smoothing-attack knobs (only honoured by amerini_smoothing) ---
    parser.add_argument('--amerini_sigma',          type=float, default=0.7,
                        help='Gaussian std for the per-kp smoothing (paper: 0.7)')
    parser.add_argument('--amerini_ksize',          type=int,   default=3,
                        help='Gaussian kernel size (paper: 3)')
    parser.add_argument('--amerini_patch_half',     type=int,   default=4,
                        help='Half-side of the per-kp modification patch '
                             '(paper: 4, giving 8x8 patches)')
    parser.add_argument('--amerini_max_iter',       type=int,   default=40,
                        help='Outer iter cap (paper: 40)')
    parser.add_argument('--amerini_target_removal', type=float, default=1.0,
                        help='Early-stop once removal_rate >= this. '
                             '1.0 = run until no kps left (or hit max_iter).')
    parser.add_argument('--no_video', action='store_true')
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 99999)
        print(f"Seed: {args.seed}  (pass --seed {args.seed} to reproduce)")

    # --- Model + detector ---
    net = SiamRPNvot()
    net.load_state_dict(torch.load(join(realpath(dirname(__file__)), args.model)))
    net.eval().cuda()
    detector = SIFTAlignmentDetector()

    # --- Video selection ---
    video       = load_video(args.dataset, args.video)
    image_files = video['image_files']
    gt          = video['gt']
    T           = min(len(image_files), len(gt))
    N           = args.n_frames
    assert T >= N + 2, f"Video too short ({T} frames); need at least {N + 2}"
    rng         = random.Random(args.seed)
    init_frame  = rng.randint(0, T - N - 1)
    sim_frames  = list(range(init_frame + 1, init_frame + 1 + N))
    print(f"Video: {args.video}  T={T}  init={init_frame}  "
          f"sim=[{sim_frames[0]},{sim_frames[-1]}]")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Run benign + attack ---
    benign_log = simulate_benign(net, image_files, gt, init_frame, sim_frames)
    attack_log = simulate_attack(net, detector, image_files, gt,
                                  init_frame, sim_frames, args)

    # --- Save + summary ---
    stem = args.out_stem or f"log_{args.video}_{args.attack}"
    log_path = join(args.out_dir, f"{stem}.npz")
    save_log(log_path, benign_log, attack_log, args)
    print(f"\nLog  -> {log_path}")

    if not args.no_video:
        video_path = join(args.out_dir, f"{stem}.mp4")
        render_video(detector, image_files, benign_log, attack_log,
                     video_path, args.attack)
        print(f"Video -> {video_path}")

    print_summary(benign_log, attack_log, args)


if __name__ == '__main__':
    main()
