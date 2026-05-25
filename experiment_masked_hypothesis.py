#!/usr/bin/env python3
"""
experiment_masked_hypothesis.py

ObjectSeeker-inspired masked hypothesis generation for adversarial tracker recovery.

Instead of drawing candidates from a single corrupted SiamRPN response map, this
experiment runs SiamRPN on M = 2*N + 4 structured masked versions of the adversarial
search crop per frame.  Each mask zeroes out one horizontal stripe, vertical stripe,
or image quadrant.  Masks that overlap the adversarial perturbation's concentrated
region suppress the fake response peak, allowing the tracker to respond to the true
target instead.

Pipeline per attack frame
─────────────────────────
  Stage 0  — For each of M masks: apply to x_adv → run SiamRPN → top-1 hypothesis
  Stage 0.5 — DBSCAN-cluster the M hypotheses by IoU → K diverse representatives
               each carries a vote_count (masks that agreed on that location)
  Stage 2  — SIFT local correspondence score for each cluster rep vs. prev template
              sort by SIFT score → top-ranked rep = recovery candidate

Saved arrays (npz)
──────────────────
  masks                   (M, H_crop, W_crop) bool   — static mask patterns
  mask_names              (M,) str

  attack_x_crop           (N, H_crop, W_crop, 3) uint8   — clean search crop
  attack_att_per          (N, H_crop, W_crop, 3) float32 — perturbation in crop coords

  attack_mask_bboxes      (N, M, 4)  — top-1 bbox from each masked run
  attack_mask_pscore      (N, M)     — pscore from each masked run
  attack_mask_rawscore    (N, M)     — raw siamrpn score
  attack_mask_iou_gt      (N, M)     — IoU of each mask hypothesis with GT
  attack_mask_sift        (N, M)     — SIFT score: mask hyp vs prev template
  attack_perturb_cov      (N, M)     — fraction of perturbation energy in masked region

  attack_cluster_bboxes   (N, K, 4)  — DBSCAN cluster reps (NaN-padded to K)
  attack_cluster_sift     (N, K)
  attack_cluster_votes    (N, K)
  attack_cluster_iou_gt   (N, K)
  attack_cluster_pscore   (N, K)

  attack_gt_sift_scores   (N,)       — GT box SIFT baseline
  attack_pred_sift_scores (N,)       — attack-pred SIFT baseline

  benign_pred_bboxes      (N, 4)
  benign_gt_bboxes        (N, 4)

Usage
─────
  python experiment_masked_hypothesis.py --video car1
  python experiment_masked_hypothesis.py --video car1 --N_masks 8 --seed 42
  python experiment_masked_hypothesis.py --video car1 --N_masks 4 --K_clusters 12
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
from sklearn.cluster import DBSCAN
from torch.autograd import Variable
from tqdm import tqdm

from net import SiamRPNvot
from run_attack import SiamRPN_init, rtaa_attack, rtaa_sift_attack
from utils import rect_2_cxy_wh, cxy_wh_2_rect, get_subwindow_tracking
from sift_alignment import SIFTAlignmentDetector
from sift_attack import frame_bbox_to_crop_bbox


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def generate_masks(crop_h, crop_w, N):
    """
    Return 2*N + 4 structured binary masks for a crop of size (crop_h, crop_w).

    Stripe masks (2*N total):
      - N horizontal: each zeroes one row-band of height ~crop_h/N
      - N vertical:   each zeroes one col-band of width  ~crop_w/N
    Quadrant masks (4): each zeroes one quarter of the crop.

    Coverage guarantee: any contiguous region taller than crop_h/N will be
    zeroed by at least one horizontal stripe mask, and similarly for width.

    Returns: list of (mask_np: np.ndarray bool (H,W), name: str)
    True = keep pixel, False = zero it out.
    """
    masks = []

    for i in range(N):
        r0 = int(round(i * crop_h / N))
        r1 = int(round((i + 1) * crop_h / N))
        m = np.ones((crop_h, crop_w), dtype=bool)
        m[r0:r1, :] = False
        masks.append((m, f"h{i}"))

    for j in range(N):
        c0 = int(round(j * crop_w / N))
        c1 = int(round((j + 1) * crop_w / N))
        m = np.ones((crop_h, crop_w), dtype=bool)
        m[:, c0:c1] = False
        masks.append((m, f"v{j}"))

    hmid, wmid = crop_h // 2, crop_w // 2
    for r0, r1, c0, c1, name in [
        (0,    hmid,   0,    wmid,   "q_tl"),
        (0,    hmid,   wmid, crop_w, "q_tr"),
        (hmid, crop_h, 0,    wmid,   "q_bl"),
        (hmid, crop_h, wmid, crop_w, "q_br"),
    ]:
        m = np.ones((crop_h, crop_w), dtype=bool)
        m[r0:r1, c0:c1] = False
        masks.append((m, name))

    return masks


# ---------------------------------------------------------------------------
# SiamRPN forward pass helpers
# ---------------------------------------------------------------------------

def _search_region_params(state):
    p, tsz = state['p'], state['target_sz']
    wc_z = tsz[1] + p.context_amount * sum(tsz)
    hc_z = tsz[0] + p.context_amount * sum(tsz)
    s_z  = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    pad = ((p.instance_size - p.exemplar_size) / 2) / scale_z
    return float(scale_z), float(s_z + 2 * pad)


def _extract_x_crop(state, im):
    p = state['p']
    scale_z, s_x = _search_region_params(state)
    x_crop = get_subwindow_tracking(
        im, state['target_pos'], p.instance_size, round(s_x), state['avg_chans']
    ).unsqueeze(0)
    return Variable(x_crop).cuda(), scale_z, s_x


def _decode_anchors(delta_raw, score_raw, state, scale_z):
    """
    Decode raw network outputs to anchor bboxes + penalty scores.

    Returns: (pscore array, penalty array, decode_fn)
    decode_fn(idx) -> np.ndarray [x, y, w, h] in frame pixel coords.
    Does NOT modify state.
    """
    p          = state['p']
    target_pos = state['target_pos']
    target_sz  = state['target_sz']
    window     = state['window']

    delta = delta_raw.copy()
    delta[0] = delta[0] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1] = delta[1] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2] = np.exp(delta[2]) * p.anchor[:, 2]
    delta[3] = np.exp(delta[3]) * p.anchor[:, 3]

    tsz = target_sz * scale_z

    def _chg(r):  return np.maximum(r, 1. / r)
    def _sz(w, h): pad = (w + h) * .5; return np.sqrt((w + pad) * (h + pad))
    def _sz_wh(wh): pad = (wh[0]+wh[1])*.5; return np.sqrt((wh[0]+pad)*(wh[1]+pad))

    s_c = _chg(_sz(delta[2], delta[3]) / _sz_wh(tsz))
    r_c = _chg((tsz[0] / tsz[1]) / (delta[2] / delta[3]))
    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore  = penalty * score_raw
    pscore  = pscore * (1 - p.window_influence) + window * p.window_influence

    def decode(idx):
        t  = delta[:, idx] / scale_z
        lr = penalty[idx] * score_raw[idx] * p.lr
        x  = float(np.clip(t[0] + target_pos[0], 0, state['im_w']))
        y  = float(np.clip(t[1] + target_pos[1], 0, state['im_h']))
        w  = float(np.clip(target_sz[0] * (1-lr) + t[2] * lr, 10, state['im_w']))
        h  = float(np.clip(target_sz[1] * (1-lr) + t[3] * lr, 10, state['im_h']))
        return np.array([x - w/2, y - h/2, w, h])

    return pscore, penalty, score_raw, decode


def _net_outputs(net, x_input):
    """Run net forward, return (delta CHW numpy, score_raw 1D numpy)."""
    with torch.no_grad():
        delta_t, score_t = net(x_input)
    delta = delta_t.permute(1,2,3,0).contiguous().view(4,-1).data.cpu().numpy()
    score_raw = F.softmax(
        score_t.permute(1,2,3,0).contiguous().view(2,-1), dim=0
    ).data[1,:].cpu().numpy()
    return delta, score_raw


def forward_top1_no_update(net, x_input, state, scale_z):
    """
    Run one SiamRPN forward pass.  Does NOT update tracker state.

    Returns (bbox [x,y,w,h], pscore: float, raw_score: float).
    Used for masked hypothesis generation so the main tracker state is
    only advanced once (by run_siamrpn_forward on the unmasked x_adv).
    """
    delta, score_raw = _net_outputs(net, x_input)
    pscore, _, score_raw_full, decode = _decode_anchors(delta, score_raw, state, scale_z)
    best = int(np.argmax(pscore))
    return decode(best), float(pscore[best]), float(score_raw_full[best])


def run_siamrpn_forward(net, x_crop, state, scale_z):
    """
    One SiamRPN forward pass.  Updates state with the top-1 result.
    Returns (pred_bbox [x,y,w,h], all_hypotheses list).
    """
    delta, score_raw = _net_outputs(net, x_crop)
    pscore, penalty, score_raw_full, decode = _decode_anchors(delta, score_raw, state, scale_z)

    best = int(np.argmax(pscore))
    best_bbox = decode(best)
    state['target_pos'] = np.array([best_bbox[0]+best_bbox[2]/2, best_bbox[1]+best_bbox[3]/2])
    state['target_sz']  = np.array([best_bbox[2], best_bbox[3]])
    state['score']      = float(score_raw_full[best])

    n = len(score_raw)
    hypotheses = [{'bbox': decode(i), 'siamrpn_score': float(score_raw_full[i]),
                   'pscore': float(pscore[i])} for i in range(n)]
    return best_bbox, hypotheses


# ---------------------------------------------------------------------------
# Stage 0: masked hypothesis generation
# ---------------------------------------------------------------------------

def run_masked_hypotheses(net, x_adv, state, scale_z, masks):
    """
    For each mask, zero out that region of x_adv and run a forward pass.

    State is NOT modified.  Returns a list of M dicts:
      name, bbox [x,y,w,h], pscore, siamrpn_score
    """
    crop_h, crop_w = x_adv.shape[2], x_adv.shape[3]
    x_base = x_adv.detach()
    results = []
    for mask_np, name in masks:
        mask_t = torch.from_numpy(mask_np.astype(np.float32)).cuda()
        mask_t = mask_t.view(1, 1, crop_h, crop_w)   # broadcast over batch + channels
        bbox, pscore, raw = forward_top1_no_update(net, x_base * mask_t, state, scale_z)
        results.append({'name': name, 'bbox': bbox, 'pscore': pscore, 'siamrpn_score': raw})
    return results


# ---------------------------------------------------------------------------
# Stage 0.5: DBSCAN clustering
# ---------------------------------------------------------------------------

def _bbox_iou(a, b):
    ax1, ay1 = a[0], a[1];  ax2, ay2 = a[0]+a[2], a[1]+a[3]
    bx1, by1 = b[0], b[1];  bx2, by2 = b[0]+b[2], b[1]+b[3]
    iw = max(0., min(ax2, bx2) - max(ax1, bx1))
    ih = max(0., min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2]*a[3] + b[2]*b[3] - inter
    return float(inter / union) if union > 0 else 0.


def _target_pseudo_box(prev_defense_bbox, target_pos, target_sz,
                       final_pos, im_bounds):
    """Reconstruct the pseudo box rtaa_attack uses as its attract target.

    Mirrors the formula at run_attack.py rtaa_attack so analysis can read pscore
    on exactly the anchor set the attack optimizes over. Returns box in frame
    coords as [x, y, w, h].

    Note: when final_pos preserves the original size (final_pos[2:4] ==
    target_sz), rate_wd evaluates to 0, collapsing the pseudo box to zero area.
    That matches rtaa_attack's behavior on the first attack frame, where the
    pseudo-attract loss term is effectively disabled.
    """
    rate_xy1 = (final_pos[0] - target_pos[0]) / im_bounds[0]
    rate_xy2 = (final_pos[1] - target_pos[1]) / im_bounds[1]
    max_change = max(abs(final_pos[2] - target_sz[0]),
                     abs(final_pos[3] - target_sz[1]))
    if max_change == abs(final_pos[2] - target_sz[0]):
        rate_wd = (final_pos[2] - target_sz[0]) / im_bounds[0]
    else:
        rate_wd = (final_pos[3] - target_sz[1]) / im_bounds[1]
    g = prev_defense_bbox
    return np.array([
        g[0] + rate_xy1 * g[2],
        g[1] + rate_xy2 * g[3],
        g[2] * rate_wd,
        g[3] * rate_wd,
    ], dtype=np.float32)


def _bbox_to_image_mask(frame_shape, bbox):
    """Build a uint8 mask (255 inside bbox, 0 outside) for SIFT detectAndCompute."""
    H, W = frame_shape[:2]
    x, y, w, h = [int(round(v)) for v in bbox]
    m = np.zeros((H, W), dtype=np.uint8)
    x0, y0 = max(0, x),       max(0, y)
    x1, y1 = max(0, min(W, x + w)), max(0, min(H, y + h))
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = 255
    return m


def cluster_hypotheses(hyp_list, iou_eps=0.5):
    """
    Cluster masked hypotheses by IoU distance using DBSCAN.

    Two boxes are placed in the same cluster when IoU >= iou_eps.
    Each cluster is represented by its highest-pscore member.
    Returns a list of cluster dicts (original keys + 'vote_count'), sorted by
    vote_count desc then pscore desc.
    """
    n = len(hyp_list)
    if n == 0:
        return []

    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            d = 1. - _bbox_iou(hyp_list[i]['bbox'], hyp_list[j]['bbox'])
            dist[i, j] = dist[j, i] = d
    # Clip to [0, 1]: floating-point rounding can produce tiny negatives
    # when IoU is exactly 1.0, which DBSCAN's precomputed-metric check rejects.
    np.clip(dist, 0., 1., out=dist)

    labels = DBSCAN(eps=1. - iou_eps, min_samples=1,
                    metric='precomputed').fit_predict(dist)

    clusters = []
    for c in range(int(labels.max()) + 1):
        members = [hyp_list[i] for i in range(n) if labels[i] == c]
        rep = dict(max(members, key=lambda h: h['pscore']))
        rep['vote_count'] = len(members)
        clusters.append(rep)

    clusters.sort(key=lambda h: (h['vote_count'], h['pscore']), reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Perturbation coverage analysis
# ---------------------------------------------------------------------------

def compute_perturbation_coverages(att_per_chw, masks):
    """
    For each mask, compute the fraction of total perturbation energy that falls
    in the masked-out (zeroed) region.

    Higher value → this mask covers a more perturbation-intensive region,
    meaning the corresponding hypothesis was generated without that perturbation.

    att_per_chw : (3, H, W) float32 — perturbation in crop coordinates
    masks       : list of (mask_np bool (H,W), name)
    Returns     : list of M floats in [0, 1]
    """
    mag   = np.abs(att_per_chw).mean(axis=0)   # (H, W) mean L1 across channels
    total = float(mag.sum()) + 1e-8
    return [float(mag[~m].sum() / total) for m, _ in masks]


# ---------------------------------------------------------------------------
# SIFT scoring (mirrors experiment_hypothesis_ranking.py)
# ---------------------------------------------------------------------------

def sift_local_score(detector, prev_frame, curr_frame, ref_bbox, hyp_bbox):
    """
    SIFT correspondence score between ref_bbox in prev_frame and hyp_bbox in curr_frame.

    Score = 0.1·match_ratio + 0.45·inlier_ratio + 0.45·descriptor_sim ∈ [0,1]
    Higher = stronger temporal correspondence = more likely to be the real target.
    """
    h, w = prev_frame.shape[:2]

    def _mask(bbox):
        m = np.zeros((h, w), dtype=np.uint8)
        x, y, bw, bh = (int(v) for v in bbox)
        m[max(y,0):min(y+bh,h), max(x,0):min(x+bw,w)] = 255
        return m

    kp_r, desc_r = detector.extract_features(prev_frame, mask=_mask(ref_bbox))
    kp_h, desc_h = detector.extract_features(curr_frame, mask=_mask(hyp_bbox))
    good = detector._ratio_match(desc_r, desc_h)
    n = max(len(kp_r), len(kp_h), 1)
    match_r  = float(min(len(good) / n, 1.))
    inlier_r = float(detector._ransac_inlier_ratio(kp_r, kp_h, good))
    desc_s   = float(detector._descriptor_similarity(good))
    return float(np.clip(0.1*match_r + 0.45*inlier_r + 0.45*desc_s, 0., 1.))


# ---------------------------------------------------------------------------
# Segmentation-mask cropping (mirrors get_subwindow_tracking geometry)
# ---------------------------------------------------------------------------

def crop_mask_to_search(mask_full, target_pos, original_sz, model_sz):
    """
    Crop a full-image binary mask around `target_pos` with side `original_sz`
    and resize to (model_sz, model_sz), matching the geometry of
    `get_subwindow_tracking`.  Out-of-image regions are padded with 0
    (non-perturbable).  NEAREST interpolation keeps the mask binary.

    Returns: torch float tensor of shape (1, 1, model_sz, model_sz) on CUDA,
    ready to broadcast across the (1, 3, model_sz, model_sz) search crop.
    """
    sz = original_sz
    c  = (sz + 1) / 2
    x_min = round(target_pos[0] - c)
    x_max = x_min + sz - 1
    y_min = round(target_pos[1] - c)
    y_max = y_min + sz - 1
    H, W = mask_full.shape[:2]
    left_pad   = int(max(0., -x_min))
    top_pad    = int(max(0., -y_min))
    right_pad  = int(max(0., x_max - W + 1))
    bottom_pad = int(max(0., y_max - H + 1))
    x_min += left_pad; x_max += left_pad
    y_min += top_pad;  y_max += top_pad
    if left_pad or top_pad or right_pad or bottom_pad:
        m_pad = np.zeros((H + top_pad + bottom_pad,
                          W + left_pad + right_pad), dtype=np.float32)
        m_pad[top_pad:top_pad+H, left_pad:left_pad+W] = mask_full.astype(np.float32)
        crop = m_pad[int(y_min):int(y_max+1), int(x_min):int(x_max+1)]
    else:
        crop = mask_full[int(y_min):int(y_max+1),
                         int(x_min):int(x_max+1)].astype(np.float32)
    crop_resized = cv2.resize(crop, (model_sz, model_sz),
                              interpolation=cv2.INTER_NEAREST)
    return (torch.from_numpy(crop_resized.astype(np.float32))
            .unsqueeze(0).unsqueeze(0).cuda())


# ---------------------------------------------------------------------------
# Perturbation injection into raw frame
# ---------------------------------------------------------------------------

def inject_perturbation(im, att_per_tensor, target_pos, s_x,
                        attack_mask=None, clip_negatives=False):
    """
    Render the search-crop perturbation onto the full image inside the search
    region centered at `target_pos`.

    `clip_negatives`: if True, zero out negative perturbation values before
    rendering.  Matches test_hijack_attack.py's "physical patch" approximation
    (a printed sticker can only add light, not subtract).  Visualization-only:
    the tracker's own forward pass still uses the signed perturbation.

    `attack_mask`: if given, the same search-crop binary mask used by
    rtaa_attack to constrain the optimization.  Resized to (s_x, s_x) with
    NEAREST interpolation and applied to the perturbation before compositing,
    so the rendered image cannot drift outside the masked region.
    """
    att_np   = att_per_tensor.cpu().detach().squeeze(0).permute(1,2,0).numpy()
    if clip_negatives:
        att_np = np.where(att_np < 0, 0, att_np)
    s_x_int  = int(round(s_x))
    att_res  = cv2.resize(att_np.astype(np.float32), (s_x_int, s_x_int))
    if attack_mask is not None:
        mask_np  = attack_mask.detach().squeeze(0).squeeze(0).cpu().numpy()
        mask_res = cv2.resize(mask_np.astype(np.float32), (s_x_int, s_x_int),
                              interpolation=cv2.INTER_NEAREST)
        att_res  = att_res * mask_res[:, :, np.newaxis]
    cx, cy   = int(round(target_pos[0])), int(round(target_pos[1]))
    x1, y1   = cx - s_x_int//2, cy - s_x_int//2
    h, w     = im.shape[:2]
    im_f     = im.astype(np.float32).copy()
    sx1, sy1 = max(0,-x1),        max(0,-y1)
    sx2, sy2 = s_x_int-max(0,x1+s_x_int-w), s_x_int-max(0,y1+s_x_int-h)
    dx1, dy1 = max(0,x1),         max(0,y1)
    dx2, dy2 = min(w,x1+s_x_int), min(h,y1+s_x_int)
    if dx2>dx1 and dy2>dy1:
        im_f[dy1:dy2,dx1:dx2] += att_res[sy1:sy2,sx1:sx2]
    return np.clip(im_f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Video rendering
# ---------------------------------------------------------------------------

_RANK_COLORS = [
    (255, 255,   0),
    (255, 200,   0),
    (200, 140,   0),
    (100,  80,   0),
    ( 60,  30,   0),
]


def _draw_bbox(frame, bbox, color, thickness=2, label=None):
    x, y, bw, bh = (int(v) for v in bbox)
    cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, thickness)
    if label:
        cv2.putText(frame, label, (x, max(y-4, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


def render_frame(clean_frame, gt_bbox, benign_pred, attack_pred,
                 masked_hyps, top_clusters,
                 frame_num, top_sift, gt_sift, pred_sift):
    """
    Compose one annotated output frame.

      Gray dots     — centres of all M masked hypotheses
      Cyan→yellow   — top-5 cluster representatives ranked by SIFT score
                      label shows rank, SIFT score, vote count
      Green         — ground-truth box
      Blue          — benign SiamRPN prediction
      Red           — attack SiamRPN prediction
      Bottom HUD    — frame / top-cluster SIFT / GT SIFT / pred SIFT
    """
    vis = clean_frame.copy()

    # All masked hypothesis centroids (gray dots)
    for mh in masked_hyps:
        cx = int(mh['bbox'][0] + mh['bbox'][2] / 2)
        cy = int(mh['bbox'][1] + mh['bbox'][3] / 2)
        cv2.circle(vis, (cx, cy), 3, (160, 160, 160), -1)

    # Top-5 cluster reps
    for rank, cl in enumerate(top_clusters[:5]):
        color = _RANK_COLORS[rank]
        label = f"#{rank+1} s={cl.get('sift_score',0):.2f} v={cl['vote_count']}"
        _draw_bbox(vis, cl['bbox'], color, thickness=1, label=label)

    _draw_bbox(vis, gt_bbox,     (0, 200,  0), thickness=2, label="GT")
    _draw_bbox(vis, benign_pred, (200, 80, 0), thickness=2, label="Benign")
    _draw_bbox(vis, attack_pred, (0,   0, 220), thickness=2, label="Attack")

    h = vis.shape[0]
    lines = [
        f"Frame: {frame_num}",
        f"TopCluster SIFT: {top_sift:.3f}",
        f"GT SIFT:         {gt_sift:.3f}",
        f"Pred SIFT:       {pred_sift:.3f}",
    ]
    for i, line in enumerate(lines):
        y_pos = h - 8 - (len(lines)-1-i) * 20
        cv2.putText(vis, line, (8, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20,20,20), 2, cv2.LINE_AA)
        cv2.putText(vis, line, (8, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
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

    # --- Optional segmentation (loaded once, reused per frame) ---
    seg_model = None
    if args.use_segmentation:
        from pixellib.semantic import semantic_segmentation
        from segment import segmentation_attack_mask as _seg_fn
        seg_model = semantic_segmentation()
        seg_model.load_ade20k_model(args.seg_model)
        print(f"Segmentation: loaded {args.seg_model}; attack will be mask-constrained")

    video       = load_video(args.dataset, args.video)
    image_files = video['image_files']
    gt          = video['gt']
    T           = min(len(image_files), len(gt))
    N           = args.n_frames
    assert T >= N + 2, f"Video too short ({T} frames); need at least {N+2}"

    rng        = random.Random(args.seed)
    init_frame = rng.randint(0, T - N - 1)
    sim_frames = list(range(init_frame + 1, init_frame + 1 + N))
    print(f"Video: {args.video}  total={T}  init={init_frame}  "
          f"sim=[{sim_frames[0]},{sim_frames[-1]}]  seed={args.seed}")

    os.makedirs(args.out_dir, exist_ok=True)

    def init_tracker():
        im0   = cv2.imread(image_files[init_frame])
        tp, tsz = rect_2_cxy_wh(gt[init_frame])
        state = SiamRPN_init(im0, tp, tsz, net)
        return state, im0, cxy_wh_2_rect(state['target_pos'], state['target_sz'])

    # Determine crop size (fixed by network architecture)
    _s, _im, _ = init_tracker()
    _xc, _, _  = _extract_x_crop(_s, _im)
    crop_h, crop_w = _xc.shape[2], _xc.shape[3]
    del _s, _im, _xc

    # Build masks (static for all frames)
    masks     = generate_masks(crop_h, crop_w, args.N_masks)
    M         = len(masks)
    K         = args.K_clusters
    mask_names_list = [name for _, name in masks]
    if args.inject_gt_hypothesis:
        # GT injection appends an extra hypothesis with no underlying mask.
        # mask_names is M+1 long; masks_np stays (M, H, W).
        mask_names_list.append('gt_inj')
    masks_np  = np.array([m for m, _ in masks])   # (M, H, W) bool

    print(f"Masks: {M} total  "
          f"({args.N_masks} h-stripes + {args.N_masks} v-stripes + 4 quadrants)  "
          f"crop={crop_h}x{crop_w}")

    # -----------------------------------------------------------------------
    # Benign simulation (baseline — no masked hypotheses needed)
    # -----------------------------------------------------------------------
    print("\n--- Benign simulation ---")
    benign_log = []
    state, prev_frame, prev_pred_bbox = init_tracker()

    for f in tqdm(sim_frames, desc="Benign"):
        im           = cv2.imread(image_files[f])
        x_crop, scale_z, _ = _extract_x_crop(state, im)
        pred_bbox, _ = run_siamrpn_forward(net, x_crop, state, scale_z)
        benign_log.append({'frame_idx': f,
                           'pred_bbox': pred_bbox.copy(),
                           'gt_bbox':   gt[f].copy()})
        prev_frame    = im
        prev_pred_bbox = pred_bbox

    # -----------------------------------------------------------------------
    # Attack simulation
    # -----------------------------------------------------------------------
    print("\n--- Attack simulation ---")
    attack_log = []
    state, prev_frame, prev_defense_bbox = init_tracker()
    # prev_defense_bbox = the DEFENSE's previous-iter top-1 (clusters[0]['bbox']
    # after SIFT rerank). On the init frame the defense has not run yet, so
    # we seed it with the init GT — same value as init_tracker's third return.
    prev_gt_bbox = gt[init_frame].copy()

    cx0 = state['target_pos'][0]
    dx  = -200 if cx0 > cv2.imread(image_files[init_frame]).shape[1] / 2 else 200
    final_pos = [state['target_pos'][0] + dx, state['target_pos'][1],
                 float(state['target_sz'][0]), float(state['target_sz'][1])]

    att_per = 0   # accumulated RTAA perturbation (int 0 = uninitialised)

    for i, f in enumerate(tqdm(sim_frames, desc="Attack")):
        im        = cv2.imread(image_files[f])
        im_bounds = [im.shape[1], im.shape[0]]

        if i % 30 == 0:
            att_per = 0

        p          = state['p']
        target_pos = state['target_pos']
        target_sz  = state['target_sz']
        scale_z, s_x = _search_region_params(state)

        x_crop = Variable(
            get_subwindow_tracking(
                im, target_pos, p.instance_size, round(s_x), state['avg_chans']
            ).unsqueeze(0)
        ).cuda()

        if isinstance(att_per, int):
            x_crop_init = x_crop.clone()
        else:
            att_np = att_per.cpu().detach().numpy()
            att_np = np.resize(att_np, (1, x_crop.shape[1], x_crop.shape[2], x_crop.shape[3]))
            x_crop_init = torch.clamp(x_crop + torch.from_numpy(att_np).cuda(), 0, 255)

        # --- Optional: segmentation mask in search-crop coords ---
        attack_mask_t = None
        seg_util = float('nan')
        if seg_model is not None:
            mask_full, seg_util = _seg_fn(seg_model, image_files[f])
            attack_mask_t = crop_mask_to_search(
                mask_full, target_pos, round(s_x), p.instance_size
            )

        sift_loss_log = {} if args.attack_variant == 'rtaa_sift' else None

        if args.attack_variant == 'rtaa_sift':
            r_target_crop = frame_bbox_to_crop_bbox(
                prev_defense_bbox, target_pos, s_x, p.instance_size
            )
            x_adv = rtaa_sift_attack(
                net, x_crop_init, x_crop, prev_defense_bbox,
                target_pos, target_sz, scale_z, p,
                r_target_crop_bbox=r_target_crop,
                eps=args.eps, iteration=args.n_iter,
                final_pos=final_pos, im_bounds=im_bounds,
                attack_mask=attack_mask_t,
                alpha_dog=args.alpha_dog, gamma_kornia=args.gamma_kornia,
                dog_contrast=args.dog_contrast,
                loss_log=sift_loss_log,
            )
        else:
            x_adv = rtaa_attack(
                net, x_crop_init, x_crop, prev_defense_bbox,
                target_pos, target_sz, scale_z, p,
                eps=args.eps, iteration=args.n_iter,
                final_pos=final_pos, im_bounds=im_bounds,
                attack_mask=attack_mask_t,
            )
        att_per = x_adv - x_crop

        # --- Crop-space arrays for analysis ---
        # x_crop: clean crop (HWC uint8 for saving)
        x_crop_np_hwc = x_crop.detach().squeeze(0).permute(1,2,0).cpu().numpy()
        # att_per: perturbation in crop space
        att_per_np_hwc = att_per.detach().squeeze(0).permute(1,2,0).cpu().numpy()
        att_per_np_chw = att_per.detach().squeeze(0).cpu().numpy()  # CHW for coverage

        # --- Stage 0: masked hypotheses (BEFORE state update) ---
        masked_hyps = run_masked_hypotheses(net, x_adv, state, scale_z, masks)

        # --- Optional GT injection (oracle diagnostic) ---
        # Appends GT bbox to the hypothesis pool. Lets us isolate SIFT-rerank
        # behavior: when GT is guaranteed to be in the pool, max_pool_iou=1.0
        # so ranking_eff_sift = top_sift_iou. If SIFT picks GT under benign
        # but not under attack, the attack broke the SIFT signal specifically.
        if args.inject_gt_hypothesis:
            masked_hyps.append({
                'name':          'gt_inj',
                'bbox':          np.array(gt[f], dtype=np.float32),
                'pscore':        1.0,
                'siamrpn_score': 1.0,
            })

        # --- Main forward pass (updates state to adversarial top-1) ---
        im_attacked = inject_perturbation(
            im, att_per, target_pos, s_x,
            attack_mask=attack_mask_t,
            clip_negatives=args.clip_negatives,
        )
        pred_bbox, all_hyps = run_siamrpn_forward(net, x_adv, state, scale_z)

        # === Attack-loss diagnostics in evaluation (pscore) space =============
        # Read pscore on the *exact* anchor subsets rtaa_attack optimizes over,
        # using the same IoU threshold (0.1) as truth_suppress_iou_thresh /
        # pseudo_iou_thresh in run_attack.py. Tracker is argmax-driven, so the
        # attack hijacks the top-1 iff pscore_pseudo_max > pscore_truth_max.
        pseudo_box = _target_pseudo_box(
            prev_defense_bbox, target_pos, target_sz, final_pos, im_bounds
        )
        pseudo_valid = (pseudo_box[2] > 0) and (pseudo_box[3] > 0)
        truth_pscores  = []
        pseudo_pscores = []
        for h in all_hyps:
            if _bbox_iou(h['bbox'], prev_defense_bbox) > 0.1:
                truth_pscores.append(h['pscore'])
            if pseudo_valid and _bbox_iou(h['bbox'], pseudo_box) > 0.1:
                pseudo_pscores.append(h['pscore'])
        pscore_truth_max  = float(max(truth_pscores))  if truth_pscores  else float('nan')
        pscore_pseudo_max = float(max(pseudo_pscores)) if pseudo_pscores else float('nan')
        pred_pscore       = float(max(h['pscore'] for h in all_hyps))

        # === Removal rate: SIFT keypoint count in prev_defense_bbox ROI =======
        # Clean `im` vs perturbed `im_attacked` — directly probes the upstream
        # signal the DoG-suppress / Kornia-suppress losses are designed to wipe.
        roi_mask     = _bbox_to_image_mask(im.shape, prev_defense_bbox)
        kp_clean     = len(detector.extract_features(im,          mask=roi_mask)[0])
        kp_attacked  = len(detector.extract_features(im_attacked, mask=roi_mask)[0])
        removal_rate = 1.0 - kp_attacked / max(kp_clean, 1)

        # --- Stage 0.5: cluster masked hypotheses ---
        clusters = cluster_hypotheses(masked_hyps, iou_eps=args.cluster_iou_eps)

        # --- Perturbation coverage per mask ---
        # NaN-pad the injected-GT entry: it has no mask, so coverage is undefined.
        perturb_cov = compute_perturbation_coverages(att_per_np_chw, masks)
        if args.inject_gt_hypothesis:
            perturb_cov.append(float('nan'))

        # --- Annotate each masked hypothesis with IoU-GT and SIFT ---
        # SIFT reference = prev_defense_bbox (defense's previous-iter top-1),
        # consistent with what a real white-box-aware defense would use.
        for mh in masked_hyps:
            mh['iou_gt']    = _bbox_iou(mh['bbox'], gt[f])
            mh['sift_score'] = sift_local_score(
                detector, prev_frame, im_attacked, prev_defense_bbox, mh['bbox']
            )

        # --- Annotate each cluster rep with SIFT and IoU-GT ---
        for cl in clusters:
            cl['sift_score'] = sift_local_score(
                detector, prev_frame, im_attacked, prev_defense_bbox, cl['bbox']
            )
            cl['iou_gt'] = _bbox_iou(cl['bbox'], gt[f])

        clusters.sort(key=lambda c: c['sift_score'], reverse=True)

        # --- Baselines ---
        gt_sift   = sift_local_score(detector, prev_frame, im_attacked, prev_gt_bbox,   gt[f])
        pred_sift = sift_local_score(detector, prev_frame, im_attacked, prev_defense_bbox, pred_bbox)

        # Segmentation mask in crop coords (None when --use_segmentation is off)
        seg_mask_np = (attack_mask_t.detach().squeeze(0).squeeze(0).cpu().numpy()
                       if attack_mask_t is not None else None)

        attack_log.append({
            'frame_idx':     f,
            'pred_bbox':     pred_bbox.copy(),
            'gt_bbox':       gt[f].copy(),
            'gt_sift_score': gt_sift,
            'pred_sift_score': pred_sift,
            # crop-space tensors
            'x_crop_np':     x_crop_np_hwc.astype(np.uint8),
            'att_per_np':    att_per_np_hwc.astype(np.float32),
            # per-mask data
            'masked_hyps':   masked_hyps,
            'perturb_cov':   perturb_cov,
            # cluster data
            'clusters':      clusters,
            # segmentation (NaN-filled when not in use)
            'seg_mask':      seg_mask_np,
            'seg_util':      float(seg_util),
            # SIFT-attack diagnostics (only populated when --attack_variant rtaa_sift)
            'sift_loss_log': sift_loss_log,
            # attack-loss diagnostics (in evaluation space, on x_adv / im_attacked)
            'pseudo_box':        pseudo_box.copy(),
            'pred_pscore':       pred_pscore,
            'pscore_truth_max':  pscore_truth_max,
            'pscore_pseudo_max': pscore_pseudo_max,
            'kp_clean':          int(kp_clean),
            'kp_attacked':       int(kp_attacked),
            'removal_rate':      float(removal_rate),
        })

        prev_frame = im_attacked
        # Rollforward R_target to the DEFENSE's top-1, not the tracker's
        # (hijacked) pscore top-1. clusters is non-empty: M masked hyps + 4
        # quadrant hyps guarantee >= 1 cluster.
        prev_defense_bbox = np.asarray(clusters[0]['bbox'], dtype=np.float32)
        prev_gt_bbox = gt[f].copy()

    # -----------------------------------------------------------------------
    # Flatten logs → numpy arrays and save .npz
    # -----------------------------------------------------------------------

    NF = len(attack_log)

    def _pad_clusters(cl_list):
        """Pad cluster list to exactly K rows with NaN/0 fill."""
        bboxes  = np.full((K, 4), np.nan, dtype=np.float32)
        sift    = np.full((K,),   np.nan, dtype=np.float32)
        votes   = np.zeros((K,),          dtype=np.int32)
        iou_gt  = np.full((K,),   np.nan, dtype=np.float32)
        pscores = np.full((K,),   np.nan, dtype=np.float32)
        for ki, cl in enumerate(cl_list[:K]):
            bboxes[ki]  = cl['bbox']
            sift[ki]    = cl.get('sift_score', np.nan)
            votes[ki]   = cl.get('vote_count', 0)
            iou_gt[ki]  = cl.get('iou_gt', np.nan)
            pscores[ki] = cl.get('pscore', np.nan)
        return bboxes, sift, votes, iou_gt, pscores

    # crop arrays: (NF, H_crop, W_crop, 3)
    attack_x_crop  = np.stack([e['x_crop_np']  for e in attack_log])
    attack_att_per = np.stack([e['att_per_np']  for e in attack_log])

    # per-mask arrays: (NF, M, ...)
    attack_mask_bboxes  = np.array([[mh['bbox']          for mh in e['masked_hyps']] for e in attack_log])
    attack_mask_pscore  = np.array([[mh['pscore']         for mh in e['masked_hyps']] for e in attack_log])
    attack_mask_raw     = np.array([[mh['siamrpn_score']  for mh in e['masked_hyps']] for e in attack_log])
    attack_mask_iou_gt  = np.array([[mh['iou_gt']         for mh in e['masked_hyps']] for e in attack_log])
    attack_mask_sift    = np.array([[mh['sift_score']      for mh in e['masked_hyps']] for e in attack_log])
    attack_perturb_cov  = np.array([e['perturb_cov'] for e in attack_log])   # (NF, M)

    # cluster arrays: (NF, K, ...)
    cl_bboxes  = np.full((NF, K, 4), np.nan, dtype=np.float32)
    cl_sift    = np.full((NF, K),    np.nan, dtype=np.float32)
    cl_votes   = np.zeros((NF, K),           dtype=np.int32)
    cl_iou_gt  = np.full((NF, K),    np.nan, dtype=np.float32)
    cl_pscores = np.full((NF, K),    np.nan, dtype=np.float32)
    for ti, e in enumerate(attack_log):
        b, s, v, ig, ps = _pad_clusters(e['clusters'])
        cl_bboxes[ti]  = b
        cl_sift[ti]    = s
        cl_votes[ti]   = v
        cl_iou_gt[ti]  = ig
        cl_pscores[ti] = ps

    # segmentation arrays (only populated when --use_segmentation; NaN otherwise)
    if seg_model is not None:
        attack_seg_masks = np.stack(
            [e['seg_mask'].astype(np.float32) for e in attack_log]
        )                                                       # (NF, H_crop, W_crop)
    else:
        attack_seg_masks = np.full((NF, crop_h, crop_w), np.nan, dtype=np.float32)
    attack_seg_util = np.array([e['seg_util'] for e in attack_log], dtype=np.float32)

    # SIFT-attack per-iter loss traces (NF, n_iter) — NaN when variant != rtaa_sift
    if args.attack_variant == 'rtaa_sift':
        def _stack(key):
            arr = np.full((NF, args.n_iter), np.nan, dtype=np.float32)
            for ti, e in enumerate(attack_log):
                vals = e['sift_loss_log'].get(key, [])
                arr[ti, :len(vals)] = vals
            return arr
        sift_loss_rtaa   = _stack('L_rtaa')
        sift_loss_dog    = _stack('L_dog')
        sift_loss_kornia = _stack('L_kornia')
        sift_loss_total  = _stack('L_total')
    else:
        sift_loss_rtaa = sift_loss_dog = sift_loss_kornia = sift_loss_total = (
            np.full((NF, args.n_iter), np.nan, dtype=np.float32)
        )

    log_stem = getattr(args, 'out_stem', None) or f"log_masked_{args.video}"
    log_path = join(args.out_dir, f"{log_stem}.npz")
    np.savez(
        log_path,
        # --- metadata ---
        init_frame  = np.array(init_frame),
        seed        = np.array(args.seed),
        N_masks     = np.array(args.N_masks),
        mask_names  = np.array(mask_names_list),
        # --- static masks ---
        masks       = masks_np,                       # (M, H_crop, W_crop) bool
        # --- benign baseline ---
        benign_frame_idxs  = np.array([e['frame_idx']  for e in benign_log]),
        benign_pred_bboxes = np.array([e['pred_bbox']   for e in benign_log]),
        benign_gt_bboxes   = np.array([e['gt_bbox']     for e in benign_log]),
        # --- attack baselines ---
        attack_frame_idxs    = np.array([e['frame_idx']       for e in attack_log]),
        attack_pred_bboxes   = np.array([e['pred_bbox']        for e in attack_log]),
        attack_gt_bboxes     = np.array([e['gt_bbox']          for e in attack_log]),
        attack_gt_sift_scores   = np.array([e['gt_sift_score']   for e in attack_log]),
        attack_pred_sift_scores = np.array([e['pred_sift_score'] for e in attack_log]),
        # --- crop-space data ---
        attack_x_crop  = attack_x_crop,               # (NF, H, W, 3) uint8
        attack_att_per = attack_att_per,               # (NF, H, W, 3) float32
        # --- per-mask data ---
        attack_mask_bboxes  = attack_mask_bboxes,      # (NF, M, 4)
        attack_mask_pscore  = attack_mask_pscore,      # (NF, M)
        attack_mask_rawscore= attack_mask_raw,         # (NF, M)
        attack_mask_iou_gt  = attack_mask_iou_gt,      # (NF, M)
        attack_mask_sift    = attack_mask_sift,        # (NF, M)
        attack_perturb_cov  = attack_perturb_cov,      # (NF, M)
        # --- cluster data (NaN-padded to K rows) ---
        attack_cluster_bboxes = cl_bboxes,             # (NF, K, 4)
        attack_cluster_sift   = cl_sift,               # (NF, K)
        attack_cluster_votes  = cl_votes,              # (NF, K)
        attack_cluster_iou_gt = cl_iou_gt,             # (NF, K)
        attack_cluster_pscore = cl_pscores,            # (NF, K)
        # --- segmentation (NaN-filled when --use_segmentation is off) ---
        use_segmentation  = np.array(bool(seg_model is not None)),
        clip_negatives    = np.array(bool(args.clip_negatives)),
        attack_seg_masks  = attack_seg_masks,          # (NF, H_crop, W_crop)
        attack_seg_util   = attack_seg_util,           # (NF,) fraction of kosher pixels
        # --- attack-variant metadata + SIFT-attack diagnostics ---
        attack_variant    = np.array(args.attack_variant),
        eps               = np.array(args.eps, dtype=np.float32),
        n_iter            = np.array(args.n_iter, dtype=np.int32),
        alpha_dog         = np.array(args.alpha_dog, dtype=np.float32),
        gamma_kornia      = np.array(args.gamma_kornia, dtype=np.float32),
        dog_contrast      = np.array(args.dog_contrast, dtype=np.float32),
        inject_gt_hypothesis = np.array(bool(args.inject_gt_hypothesis)),
        sift_loss_rtaa    = sift_loss_rtaa,
        sift_loss_dog     = sift_loss_dog,
        sift_loss_kornia  = sift_loss_kornia,
        sift_loss_total   = sift_loss_total,
        # --- attack-loss diagnostics (pscore-space + SIFT removal rate) ---
        attack_pseudo_bboxes      = np.array([e['pseudo_box']        for e in attack_log], dtype=np.float32),
        attack_pred_pscore        = np.array([e['pred_pscore']       for e in attack_log], dtype=np.float32),
        attack_pscore_truth_max   = np.array([e['pscore_truth_max']  for e in attack_log], dtype=np.float32),
        attack_pscore_pseudo_max  = np.array([e['pscore_pseudo_max'] for e in attack_log], dtype=np.float32),
        attack_kp_clean           = np.array([e['kp_clean']          for e in attack_log], dtype=np.int32),
        attack_kp_attacked        = np.array([e['kp_attacked']       for e in attack_log], dtype=np.int32),
        attack_removal_rate       = np.array([e['removal_rate']      for e in attack_log], dtype=np.float32),
    )
    print(f"\nLog  → {log_path}")

    # -----------------------------------------------------------------------
    # Render annotated video
    # -----------------------------------------------------------------------
    first_im = cv2.imread(image_files[sim_frames[0]])
    h_v, w_v = first_im.shape[:2]
    video_path = join(args.out_dir, f"video_masked_{args.video}.mp4")
    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (w_v, h_v)
    )
    for b_e, a_e in zip(benign_log, attack_log):
        clean     = cv2.imread(image_files[b_e['frame_idx']])
        clusters  = a_e['clusters']    # sorted by sift_score
        top_sift  = clusters[0].get('sift_score', 0.) if clusters else 0.
        vis = render_frame(
            clean_frame   = clean,
            gt_bbox       = b_e['gt_bbox'],
            benign_pred   = b_e['pred_bbox'],
            attack_pred   = a_e['pred_bbox'],
            masked_hyps   = a_e['masked_hyps'],
            top_clusters  = clusters,
            frame_num     = b_e['frame_idx'],
            top_sift      = top_sift,
            gt_sift       = a_e['gt_sift_score'],
            pred_sift     = a_e['pred_sift_score'],
        )
        writer.write(vis)
    writer.release()
    print(f"Video → {video_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Masked hypothesis generation for adversarial tracker recovery"
    )
    parser.add_argument('--dataset',   default='VOT2018')
    parser.add_argument('--video',     required=True)
    parser.add_argument('--model',     default='SiamRPNvot.model')
    parser.add_argument('--out_dir',   default='out/masked_hypothesis')
    parser.add_argument('--out_stem',  default=None,
                        help='Log filename stem (default: log_masked_<video>)')
    parser.add_argument('--N_masks',   type=int, default=8,
                        help='N horizontal + N vertical stripe masks (default 8, giving 16+4=20 total)')
    parser.add_argument('--K_clusters', type=int, default=20,
                        help='Max cluster representatives to save (NaN-padded)')
    parser.add_argument('--cluster_iou_eps', type=float, default=0.5,
                        help='IoU threshold for DBSCAN clustering (higher = finer clusters)')
    parser.add_argument('--n_frames',  type=int, default=5,
                        help='Number of simulation frames after init')
    parser.add_argument('--seed',      type=int, default=None)
    parser.add_argument('--clip_negatives', action='store_true',
                        help='Zero out negative perturbation values in the rendered '
                             'image (matches test_hijack_attack.py; render-only — '
                             'the tracker still sees the signed perturbation)')
    parser.add_argument('--use_segmentation', action='store_true',
                        help='Constrain the RTAA optimization to ADE20K-segmented '
                             '"kosher" regions (walls, buildings, roads, signs, ...). '
                             'Affects both the attack and the rendered image.')
    parser.add_argument('--seg_model', default='deeplabv3_xception65_ade20k.h5',
                        help='Path to ADE20K segmentation model weights '
                             '(required when --use_segmentation is set)')
    parser.add_argument('--attack_variant', default='rtaa',
                        choices=['rtaa', 'rtaa_sift'],
                        help='Vanilla RTAA, or RTAA augmented with SIFT-evasion '
                             'loss terms (DoG-suppress + optional Kornia surrogate)')
    parser.add_argument('--eps', type=float, default=10.0,
                        help='L_inf perturbation budget on the search crop '
                             '(default 10 — the published threat model). '
                             'Set higher only for budget studies.')
    parser.add_argument('--n_iter', type=int, default=5,
                        help='PGD iterations inside the attack loop')
    parser.add_argument('--alpha_dog', type=float, default=1000.0,
                        help='Weight on the zero-gap DoG-suppress loss term '
                             '(rtaa_sift variant only)')
    parser.add_argument('--gamma_kornia', type=float, default=0.0,
                        help='Weight on the BPDA-surrogate Kornia-SIFT '
                             'suppress loss term (rtaa_sift variant only; '
                             'default 0 — enable only if DoG alone is insufficient)')
    parser.add_argument('--dog_contrast', type=float, default=0.04,
                        help='Contrast threshold below which |DoG| is unpenalised '
                             '(matches OpenCV cv2.SIFT_create contrastThreshold)')
    parser.add_argument('--inject_gt_hypothesis', action='store_true',
                        help='Append GT bbox to the masked-hypothesis pool as an '
                             'oracle diagnostic. With GT guaranteed in the pool, '
                             'max_pool_iou = 1.0, so ranking_eff_sift = top_sift_iou. '
                             'Isolates whether the SIFT reranker picks GT under '
                             'attack vs benign.')
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 99999)
        print(f"Seed: {args.seed}  (pass --seed {args.seed} to reproduce)")

    run(args)


if __name__ == '__main__':
    main()
