#!/usr/bin/env python3
"""
experiment_hypothesis_ranking.py

Two-stage SIFT hypothesis ranking experiment.

For a randomly selected 10-frame window in a specified video:
  1. Initialises SiamRPN at the chosen frame (no perturbation yet).
  2. Runs a BENIGN simulation (10 frames, no attack).
  3. Re-initialises and runs an ATTACK simulation (10 frames, RTAA).

In both simulations the top-K SiamRPN hypotheses (raw response-map candidates)
are scored by the two-stage SIFT pipeline discussed in design:

  Stage 1 — full-frame SIFT matching → global homography estimate H_bg
             (camera ego-motion prior; logged but not used to filter candidates
              since the digital RTAA attack does not shift raw pixels globally)

  Stage 2 — local SIFT correspondence:
             match keypoints in frame_{t-1}[prev_pred_bbox] against keypoints
             in frame_t[hypothesis_bbox].  This directly tests "does this
             candidate region have natural temporal continuity with the last
             known target?"  Adversarially-induced hypotheses score near zero
             because the RTAA perturbation has no real correspondence to the
             previous template.

In the attack simulation, a SIFT score for the ground-truth box is also computed
as a baseline reference (how well does the GT region correspond to the previous
GT region despite the attack?).

Outputs
-------
  <out_dir>/log_<video>.npz    — per-frame hypothesis scores for both scenarios
  <out_dir>/video_<video>.mp4  — annotated video (5 fps)

Video annotation legend
-----------------------
  Green          — ground-truth box
  Blue           — SiamRPN benign prediction
  Red            — SiamRPN attack prediction
  Cyan → yellow  — top-5 attack hypotheses ranked by two-stage SIFT score
                   (rank 1 = brightest/cyan, rank 5 = darker)
  Bottom-left HUD: frame index | top SIFT score | GT SIFT score

Usage
-----
  python experiment_hypothesis_ranking.py --dataset VOT2018 --video car1
  python experiment_hypothesis_ranking.py --dataset VOT2018 --video car1 --K 10 --seed 42
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
)
from utils import rect_2_cxy_wh, cxy_wh_2_rect, get_subwindow_tracking
from sift_alignment import SIFTAlignmentDetector


# ---------------------------------------------------------------------------
# SiamRPN: single forward pass returning top-1 + top-K hypotheses
# ---------------------------------------------------------------------------

def _search_region_params(state):
    """Return (scale_z, s_x) — the scale factor and search-region size in frame pixels."""
    p = state['p']
    target_sz = state['target_sz']
    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    pad = ((p.instance_size - p.exemplar_size) / 2) / scale_z
    s_x = s_z + 2 * pad
    return float(scale_z), float(s_x)


def _extract_x_crop(state, im):
    """Crop the search region from `im` as a (1, 3, H, W) CUDA tensor."""
    p = state['p']
    scale_z, s_x = _search_region_params(state)
    x_crop = get_subwindow_tracking(
        im, state['target_pos'], p.instance_size, round(s_x), state['avg_chans']
    ).unsqueeze(0)
    return Variable(x_crop).cuda(), scale_z, s_x


def run_siamrpn_forward(net, x_crop, state, scale_z):
    """
    One SiamRPN forward pass.  Updates state with the top-1 result and returns
    ALL decoded anchor hypotheses as a spatial candidate pool.

    The pool is intentionally not filtered by pscore here — under adversarial
    attack the pscore is corrupted and would bias candidate selection toward the
    adversarial location.  Downstream, Stage 1 (H_bg IoU filter) selects the K
    spatially-plausible candidates before Stage 2 (SIFT) ranks them.

    All bounding boxes are [x, y, w, h] in frame pixel coordinates.

    Parameters
    ----------
    net      : SiamRPNvot (eval, CUDA)
    x_crop   : (1, 3, H, W) CUDA tensor — may be clean or adversarially perturbed
    state    : tracker state dict (mutated in-place with the top-1 result)
    scale_z  : float — search-region scale factor

    Returns
    -------
    pred_bbox   : np.ndarray [x, y, w, h] — top-1 prediction in frame coords
                  (selected by pscore — this is the tracker's reported output)
    hypotheses  : list of ALL anchor dicts with keys
                    bbox           [x,y,w,h]
                    siamrpn_score  raw softmax score
                    pscore         penalised + windowed score (for logging only)
    """
    p = state['p']
    target_pos = state['target_pos']
    target_sz = state['target_sz']   # frame coords
    window = state['window']

    delta, score = net(x_crop)

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score_raw = F.softmax(
        score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0
    ).data[1, :].cpu().numpy()

    # Decode anchor offsets (search-region crop coordinates)
    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    # Size and aspect-ratio penalty (same as tracker_eval in run_attack.py)
    target_sz_scaled = target_sz * scale_z   # search-region scale for penalty

    def _change(r):
        return np.maximum(r, 1.0 / r)

    def _sz(w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def _sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        return np.sqrt((wh[0] + pad) * (wh[1] + pad))

    s_c = _change(_sz(delta[2, :], delta[3, :]) / _sz_wh(target_sz_scaled))
    r_c = _change((target_sz_scaled[0] / target_sz_scaled[1]) / (delta[2, :] / delta[3, :]))
    penalty = np.exp(-(r_c * s_c - 1.0) * p.penalty_k)
    pscore = penalty * score_raw
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence

    def _decode(idx):
        """Decode anchor idx → [x, y, w, h] clamped frame-pixel bbox."""
        target = delta[:, idx] / scale_z          # frame coords
        lr = penalty[idx] * score_raw[idx] * p.lr
        res_x = target[0] + target_pos[0]
        res_y = target[1] + target_pos[1]
        res_w = target_sz[0] * (1 - lr) + target[2] * lr
        res_h = target_sz[1] * (1 - lr) + target[3] * lr
        res_x = float(max(0, min(state['im_w'], res_x)))
        res_y = float(max(0, min(state['im_h'], res_y)))
        res_w = float(max(10, min(state['im_w'], res_w)))
        res_h = float(max(10, min(state['im_h'], res_h)))
        return np.array([res_x - res_w / 2, res_y - res_h / 2, res_w, res_h])

    # --- Top-1: update tracker state (mirrors tracker_eval behaviour) ---
    best_id = int(np.argmax(pscore))
    best_bbox = _decode(best_id)
    state['target_pos'] = np.array([best_bbox[0] + best_bbox[2] / 2,
                                    best_bbox[1] + best_bbox[3] / 2])
    state['target_sz'] = np.array([best_bbox[2], best_bbox[3]])
    state['score'] = float(score_raw[best_id])

    # --- Full anchor pool (all anchors, no pscore filtering) ---
    # Decoding all ~1445 anchors is fast (numpy); SIFT scoring happens only on
    # the K candidates selected by Stage 1, so this does not add significant cost.
    n_anchors = len(score_raw)
    hypotheses = []
    for idx in range(n_anchors):
        hypotheses.append({
            'bbox': _decode(idx),
            'siamrpn_score': float(score_raw[idx]),
            'pscore': float(pscore[idx]),
        })

    return best_bbox, hypotheses


# ---------------------------------------------------------------------------
# SIFT two-stage scoring
# ---------------------------------------------------------------------------

def stage1_global_homography(detector, prev_frame, curr_frame):
    """
    Stage 1: estimate global camera-motion homography H from full-frame SIFT.

    Returns H (3×3 ndarray) or None.  Used as a spatial prior to predict where
    the correct target "should" appear under pure ego-motion, independent of
    the adversarial perturbation.
    """
    kp1, desc1 = detector.extract_features(prev_frame)
    kp2, desc2 = detector.extract_features(curr_frame)
    good = detector._ratio_match(desc1, desc2)
    if len(good) < detector.min_matches:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H


def sift_local_score(detector, prev_frame, curr_frame, ref_bbox, hyp_bbox):
    """
    Stage 2: SIFT correspondence score between ref_bbox in prev_frame and
    hyp_bbox in curr_frame.

    The question answered: "does the content inside this candidate box have
    natural temporal continuity with the previous target?"

    Adversarially-induced hypotheses fail this test because RTAA perturbations
    are synthetic noise — they carry no real keypoint correspondence to the
    previous template region.

    Score = 0.4·match_ratio + 0.4·inlier_ratio + 0.2·descriptor_sim ∈ [0, 1]
    Higher = stronger correspondence = more likely to be the real target.
    """
    h, w = prev_frame.shape[:2]

    def _mask(bbox):
        m = np.zeros((h, w), dtype=np.uint8)
        x, y, bw, bh = (int(v) for v in bbox)
        m[max(y, 0):min(y + bh, h), max(x, 0):min(x + bw, w)] = 255
        return m

    kp_ref, desc_ref = detector.extract_features(prev_frame, mask=_mask(ref_bbox))
    kp_hyp, desc_hyp = detector.extract_features(curr_frame, mask=_mask(hyp_bbox))

    good = detector._ratio_match(desc_ref, desc_hyp)
    n = max(len(kp_ref), len(kp_hyp), 1)
    match_r = float(min(len(good) / n, 1.0))
    inlier_r = float(detector._ransac_inlier_ratio(kp_ref, kp_hyp, good))
    desc_s = float(detector._descriptor_similarity(good))
    return float(np.clip(0.4 * match_r + 0.4 * inlier_r + 0.2 * desc_s, 0.0, 1.0))


def _warp_bbox(bbox, H):
    """
    Warp a [x, y, w, h] bounding box through homography H.

    All four corners are transformed and the axis-aligned bounding rectangle
    of the warped corners is returned as [x, y, w, h].
    """
    x, y, w, h = bbox
    corners = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    warped = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
    x1, y1 = warped.min(axis=0)
    x2, y2 = warped.max(axis=0)
    return np.array([x1, y1, x2 - x1, y2 - y1])


def _bbox_iou(a, b):
    """IoU between two [x, y, w, h] bounding boxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def score_and_rank(detector, prev_frame, curr_frame, prev_pred_bbox, all_hypotheses, K=10):
    """
    Two-stage scoring pipeline.

    Stage 1 — spatial filtering via H_bg
        Estimate the global camera-motion homography H_bg from full-frame SIFT.
        Warp prev_pred_bbox through H_bg to obtain predicted_bbox: where the
        target would land under pure camera motion.  Select the K hypotheses
        from the full anchor pool with the highest IoU against predicted_bbox.
        This bypasses the corrupted pscore and avoids wasting SIFT computation
        on anchors that are spatially far from any plausible target location.

        Fallback: if H_bg estimation fails (too few global matches), fall back
        to top-K by pscore to keep the pipeline running.

    Stage 2 — appearance correspondence via local SIFT
        For each of the K spatial candidates, match SIFT keypoints between
        prev_frame[prev_pred_bbox] and curr_frame[hypothesis_bbox].
        Score = 0.4·match_ratio + 0.4·inlier_ratio + 0.2·descriptor_sim.
        Adversarially-induced candidates score near zero because the RTAA
        perturbation has no real temporal correspondence to the previous template.

    Returns
    -------
    candidates : list of K dicts, sorted by sift_score descending, each with
                   bbox, siamrpn_score, pscore, sift_score, iou_with_predicted
    H_bg       : 3×3 ndarray or None
    predicted_bbox : [x,y,w,h] warped bbox or None (for logging / visualisation)
    """
    # --- Stage 1: H_bg + IoU-based candidate selection ---
    H_bg = stage1_global_homography(detector, prev_frame, curr_frame)
    predicted_bbox = None

    if H_bg is not None:
        predicted_bbox = _warp_bbox(prev_pred_bbox, H_bg)
        ious = [_bbox_iou(h['bbox'], predicted_bbox) for h in all_hypotheses]
        top_k_ids = np.argsort(ious)[::-1][:K]
        candidates = []
        for idx in top_k_ids:
            h = dict(all_hypotheses[idx])   # copy so we can add sift_score
            h['iou_with_predicted'] = float(ious[idx])
            candidates.append(h)
    else:
        # H_bg failed — fall back to pscore ranking
        raise RuntimeError("Global homography estimation failed; cannot select candidates by spatial prior.")
        sorted_hyps = sorted(all_hypotheses, key=lambda h: h['pscore'], reverse=True)[:K]
        candidates = [dict(h) for h in sorted_hyps]
        for c in candidates:
            c['iou_with_predicted'] = float('nan')

    # --- Stage 2: local SIFT correspondence ---
    for c in candidates:
        c['sift_score'] = sift_local_score(
            detector, prev_frame, curr_frame, prev_pred_bbox, c['bbox']
        )
    candidates.sort(key=lambda h: h['sift_score'], reverse=True)

    return candidates, H_bg, predicted_bbox


# ---------------------------------------------------------------------------
# Perturbation injection: search-region tensor → raw frame pixels
# ---------------------------------------------------------------------------

def inject_perturbation(im, att_per_tensor, target_pos, s_x):
    """
    Project the RTAA search-region perturbation back onto the raw frame pixels.

    This makes the digital perturbation "visible" to SIFT, simulating the
    visual effect of a physical adversarial patch confined to the search region.

    att_per_tensor : torch Tensor (1, 3, instance_size, instance_size)
                     = x_adv - x_crop  (in [0,255] pixel scale)
    target_pos     : (cx, cy) search-region centre in frame coords
    s_x            : search-region size in frame pixels (before resampling)

    Returns a uint8 BGR frame with the perturbation pasted into the search area.
    """
    att_np = att_per_tensor.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    s_x_int = int(round(s_x))
    att_resized = cv2.resize(att_np.astype(np.float32), (s_x_int, s_x_int))

    cx, cy = int(round(target_pos[0])), int(round(target_pos[1]))
    x1 = cx - s_x_int // 2
    y1 = cy - s_x_int // 2

    h, w = im.shape[:2]
    im_f = im.astype(np.float32).copy()

    # Clip source/dest to valid frame boundaries
    src_x1 = max(0, -x1);   src_y1 = max(0, -y1)
    src_x2 = s_x_int - max(0, x1 + s_x_int - w)
    src_y2 = s_x_int - max(0, y1 + s_x_int - h)
    dst_x1 = max(0, x1);    dst_y1 = max(0, y1)
    dst_x2 = min(w, x1 + s_x_int)
    dst_y2 = min(h, y1 + s_x_int)

    if dst_x2 > dst_x1 and dst_y2 > dst_y1:
        im_f[dst_y1:dst_y2, dst_x1:dst_x2] += att_resized[src_y1:src_y2, src_x1:src_x2]

    return np.clip(im_f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Video rendering
# ---------------------------------------------------------------------------

# Colour gradient for hypothesis ranks 1→5 (BGR)
_RANK_COLORS = [
    (255, 255,   0),   # rank 1: cyan
    (255, 200,   0),   # rank 2
    (200, 140,   0),   # rank 3
    (100,  80,   0),   # rank 4
    ( 60,  30,   0),   # rank 5
]


def _draw_bbox(frame, bbox, color, thickness=2, label=None):
    x, y, bw, bh = (int(v) for v in bbox)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, thickness)
    if label:
        cv2.putText(frame, label, (x, max(y - 4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)


def render_frame(clean_frame, gt_bbox, benign_pred, attack_pred,
                 top5_attack_hyps, frame_num, top_sift_score, gt_sift_score):
    """
    Compose one annotated output frame.

    Drawing order (back to front so important boxes are not obscured):
      1. Top-5 SIFT-ranked attack hypotheses (thinner boxes)
      2. GT box (green)
      3. Benign prediction (blue)
      4. Attack prediction (red)
      5. Bottom-left HUD text
    """
    vis = clean_frame.copy()

    # Top-5 two-stage ranked hypotheses
    for rank, hyp in enumerate(top5_attack_hyps[:5]):
        color = _RANK_COLORS[rank]
        label = f"#{rank + 1} sift={hyp['sift_score']:.2f}"
        _draw_bbox(vis, hyp['bbox'], color, thickness=1, label=label)

    _draw_bbox(vis, gt_bbox,     (0, 200, 0),   thickness=2, label="GT")
    _draw_bbox(vis, benign_pred, (200, 80, 0),  thickness=2, label="Benign")
    _draw_bbox(vis, attack_pred, (0,  0, 220),  thickness=2, label="Attack")

    # Bottom-left HUD
    h = vis.shape[0]
    lines = [
        f"Frame: {frame_num}",
        f"Top SIFT: {top_sift_score:.3f}",
        f"GT SIFT:  {gt_sift_score:.3f}",
    ]
    for i, line in enumerate(lines):
        y_pos = h - 8 - (len(lines) - 1 - i) * 20
        # Shadow for readability on any background
        cv2.putText(vis, line, (8, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(vis, line, (8, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_video(dataset_name, video_name):
    json_path = join(realpath(dirname(__file__)), 'data', dataset_name + '.json')
    info = json.load(open(json_path))

    for v in info.values():
        if v['name'] == video_name:
            v['image_files'] = [
                join(realpath(dirname(__file__)), 'data', dataset_name, v['name'], 'img', f)
                for f in v['image_files']
            ]
            v['gt'] = np.array(v['gt'])
            return v
    raise ValueError(f"Video '{video_name}' not found in {dataset_name}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(args):
    # --- Model ---
    net = SiamRPNvot()
    net.load_state_dict(torch.load(join(realpath(dirname(__file__)), args.model)))
    net.eval().cuda()

    detector = SIFTAlignmentDetector()

    video = load_video(args.dataset, args.video)
    image_files = video['image_files']
    gt = video['gt']
    T = min(len(image_files), len(gt))
    N = args.n_frames

    assert T >= N + 2, f"Video too short ({T} frames); need at least {N + 2}"

    # --- Random start frame ---
    # init_frame = initialisation only (no tracking step)
    # sim_frames = the N frames we actually analyse
    rng = random.Random(args.seed)
    init_frame = rng.randint(0, T - N - 1)
    sim_frames = list(range(init_frame + 1, init_frame + 1 + N))
    print(f"Video: {args.video}  total={T}  init_frame={init_frame}  "
          f"sim=[{sim_frames[0]}, {sim_frames[-1]}]  seed={args.seed}")

    os.makedirs(args.out_dir, exist_ok=True)

    def init_tracker():
        im0 = cv2.imread(image_files[init_frame])
        assert im0 is not None, f"Cannot read {image_files[init_frame]}"
        tp, tsz = rect_2_cxy_wh(gt[init_frame])
        state = SiamRPN_init(im0, tp, tsz, net)
        init_bbox = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        return state, im0, init_bbox

    # -----------------------------------------------------------------------
    # BENIGN simulation
    # -----------------------------------------------------------------------
    print("\n--- Benign simulation ---")
    benign_log = []
    state, prev_frame, prev_pred_bbox = init_tracker()

    for f in tqdm(sim_frames, desc="Benign"):
        im = cv2.imread(image_files[f])
        assert im is not None

        x_crop, scale_z, _ = _extract_x_crop(state, im)

        pred_bbox, hypotheses = run_siamrpn_forward(
            net, x_crop, state, scale_z
        )

        ranked_hyps, _, _ = score_and_rank(
            detector, prev_frame, im, prev_pred_bbox, hypotheses, K=args.K
        )

        benign_log.append({
            'frame_idx': f,
            'pred_bbox': pred_bbox.copy(),
            'gt_bbox': gt[f].copy(),
            'hypotheses': [
                {k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in h.items()}
                for h in ranked_hyps
            ],
        })

        prev_frame = im
        prev_pred_bbox = pred_bbox

    # -----------------------------------------------------------------------
    # ATTACK simulation
    # -----------------------------------------------------------------------
    print("\n--- Attack simulation ---")
    attack_log = []
    state, prev_frame, prev_pred_bbox = init_tracker()
    prev_gt_bbox = gt[init_frame].copy()

    # Retarget direction (matches test_hijack_attack.py convention)
    cx0 = state['target_pos'][0]
    dx = -200 if cx0 > cv2.imread(image_files[init_frame]).shape[1] / 2 else 200
    final_pos = [state['target_pos'][0] + dx, state['target_pos'][1],
                 float(state['target_sz'][0]), float(state['target_sz'][1])]

    att_per = 0   # accumulated RTAA perturbation (int 0 = uninitialised)

    for i, f in enumerate(tqdm(sim_frames, desc="Attack")):
        im = cv2.imread(image_files[f])
        assert im is not None
        im_bounds = [im.shape[1], im.shape[0]]

        # Reset perturbation every 30 frames (mirrors test_hijack_attack.py)
        if i % 30 == 0:
            att_per = 0

        p = state['p']
        target_pos = state['target_pos']
        target_sz = state['target_sz']
        scale_z, s_x = _search_region_params(state)

        x_crop = Variable(
            get_subwindow_tracking(
                im, target_pos, p.instance_size, round(s_x), state['avg_chans']
            ).unsqueeze(0)
        ).cuda()

        # Apply accumulated perturbation then re-attack
        if isinstance(att_per, int):
            x_crop_init = x_crop.clone()
        else:
            att_np = att_per.cpu().detach().numpy()
            att_np = np.resize(att_np, (1, x_crop.shape[1], x_crop.shape[2], x_crop.shape[3]))
            x_crop_init = torch.clamp(x_crop + torch.from_numpy(att_np).cuda(), 0, 255)

        x_adv = rtaa_attack(
            net, x_crop_init, x_crop, prev_pred_bbox,
            target_pos, target_sz, scale_z, p,
            iteration=5, final_pos=final_pos, im_bounds=im_bounds,
        )
        att_per = x_adv - x_crop

        # Inject perturbation into raw frame so SIFT sees the attack signal
        im_attacked = inject_perturbation(im, att_per, target_pos, s_x)

        # Forward on attacked crop; updates state top-1
        pred_bbox, hypotheses = run_siamrpn_forward(
            net, x_adv, state, scale_z
        )

        # Two-stage SIFT ranking on attacked frames
        ranked_hyps, _, _ = score_and_rank(
            detector, prev_frame, im_attacked, prev_pred_bbox, hypotheses, K=args.K
        )

        # GT baseline: how well does GT_t correspond to GT_{t-1} under attack?
        gt_sift = sift_local_score(
            detector, prev_frame, im_attacked, prev_gt_bbox, gt[f]
        )

        attack_log.append({
            'frame_idx': f,
            'pred_bbox': pred_bbox.copy(),
            'gt_bbox': gt[f].copy(),
            'gt_sift_score': gt_sift,
            'hypotheses': [
                {k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in h.items()}
                for h in ranked_hyps
            ],
        })

        prev_frame = im_attacked
        prev_pred_bbox = pred_bbox
        prev_gt_bbox = gt[f].copy()

    # -----------------------------------------------------------------------
    # Save log
    # -----------------------------------------------------------------------
    def _flatten(log, with_gt_sift=False):
        out = {
            'frame_idxs':  np.array([e['frame_idx'] for e in log]),
            'pred_bboxes': np.array([e['pred_bbox'] for e in log]),
            'gt_bboxes':   np.array([e['gt_bbox']   for e in log]),
            'hyp_bboxes':  np.array([[h['bbox']         for h in e['hypotheses']] for e in log]),
            'hyp_sift':    np.array([[h['sift_score']   for h in e['hypotheses']] for e in log]),
            'hyp_pscore':  np.array([[h['pscore']       for h in e['hypotheses']] for e in log]),
            'hyp_rawscore':np.array([[h['siamrpn_score'] for h in e['hypotheses']] for e in log]),
        }
        if with_gt_sift:
            out['gt_sift_scores'] = np.array([e['gt_sift_score'] for e in log])
        return out

    b = _flatten(benign_log, with_gt_sift=False)
    a = _flatten(attack_log, with_gt_sift=True)

    log_path = join(args.out_dir, f"log_{args.video}.npz")
    np.savez(
        log_path,
        # metadata
        init_frame=np.array(init_frame),
        seed=np.array(args.seed),
        # benign
        benign_frame_idxs=b['frame_idxs'],
        benign_pred_bboxes=b['pred_bboxes'],
        benign_gt_bboxes=b['gt_bboxes'],
        benign_hyp_bboxes=b['hyp_bboxes'],
        benign_hyp_sift=b['hyp_sift'],
        benign_hyp_pscore=b['hyp_pscore'],
        benign_hyp_rawscore=b['hyp_rawscore'],
        # attack
        attack_frame_idxs=a['frame_idxs'],
        attack_pred_bboxes=a['pred_bboxes'],
        attack_gt_bboxes=a['gt_bboxes'],
        attack_hyp_bboxes=a['hyp_bboxes'],
        attack_hyp_sift=a['hyp_sift'],
        attack_hyp_pscore=a['hyp_pscore'],
        attack_hyp_rawscore=a['hyp_rawscore'],
        attack_gt_sift_scores=a['gt_sift_scores'],
    )
    print(f"\nLog  → {log_path}")

    # -----------------------------------------------------------------------
    # Render output video
    # -----------------------------------------------------------------------
    first_im = cv2.imread(image_files[sim_frames[0]])
    h_v, w_v = first_im.shape[:2]
    video_path = join(args.out_dir, f"video_{args.video}.mp4")
    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (w_v, h_v)
    )

    for b_e, a_e in zip(benign_log, attack_log):
        clean = cv2.imread(image_files[b_e['frame_idx']])
        top5 = a_e['hypotheses'][:5]
        top_sift = top5[0]['sift_score'] if top5 else 0.0

        vis = render_frame(
            clean_frame=clean,
            gt_bbox=b_e['gt_bbox'],
            benign_pred=b_e['pred_bbox'],
            attack_pred=a_e['pred_bbox'],
            top5_attack_hyps=top5,
            frame_num=b_e['frame_idx'],
            top_sift_score=top_sift,
            gt_sift_score=a_e['gt_sift_score'],
        )
        writer.write(vis)

    writer.release()
    print(f"Video → {video_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage SIFT hypothesis ranking experiment"
    )
    parser.add_argument('--dataset',  default='VOT2018',
                        help='Dataset name (matching .json in data/)')
    parser.add_argument('--video',    required=True,
                        help='Video sequence name')
    parser.add_argument('--model',    default='SiamRPNvot.model')
    parser.add_argument('--out_dir',  default='out/hypothesis_ranking')
    parser.add_argument('--K',        type=int, default=20,
                        help='Number of top-K hypotheses extracted from response map')
    parser.add_argument('--n_frames', type=int, default=5,
                        help='Number of simulation frames after init')
    parser.add_argument('--seed',     type=int, default=None,
                        help='RNG seed for start-frame selection (random if omitted)')
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 99999)
        print(f"Seed: {args.seed}  (pass --seed {args.seed} to reproduce)")

    run(args)


if __name__ == '__main__':
    main()
