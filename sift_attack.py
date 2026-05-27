"""
sift_attack.py

Adaptive-attack loss terms targeting the SIFT-based hypothesis re-ranker
in sift_alignment.py. The defense itself stays on OpenCV SIFT; these
losses provide gradient signal for the attacker.

Two complementary suppress terms, both on the search crop x_adv
(shape (1, 3, H, W), BGR float in [0, 255]):

  L_DoG_suppress    — penalises |DoG(x_adv)| above OpenCV's contrast
                      threshold inside R_target. Zero surrogate gap:
                      DoG = G(sigma1) - G(sigma2) is identical math
                      under any library (Gaussian blur is deterministic).
                      This is the workhorse — it attacks the upstream
                      signal that classical SIFT's threshold gating
                      relies on.

  L_kornia_suppress — Kornia-SIFT response inside R_target. BPDA
                      surrogate (Kornia's soft-NMS forward differs from
                      OpenCV's hard NMS; calibration showed ~52% kp
                      co-location at 4 px). Use only when DoG suppress
                      alone is insufficient; transfer to OpenCV is
                      empirical and reported separately.

The bbox R_target arrives in frame pixel coords and is mapped to the
search-crop coordinate system inside each loss via
`frame_bbox_to_crop_bbox`.
"""

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Frame <-> crop bbox mapping
# ---------------------------------------------------------------------------

def frame_bbox_to_crop_bbox(bbox_frame, target_pos, s_x, model_sz):
    """
    Map a [x, y, w, h] bbox from frame pixel coords to search-crop coords.

    The search crop is centered at `target_pos` with side `s_x`, then
    resized to `model_sz` (typically 271). Crop-coords are therefore:
        crop_x = (frame_x - (target_pos[0] - s_x/2)) * (model_sz / s_x)

    Out-of-crop bboxes are clipped. Returns [cx, cy, cw, ch] in crop pixels.
    """
    fx, fy, fw, fh = (float(v) for v in bbox_frame)
    sx_left = float(target_pos[0]) - float(s_x) / 2
    sy_top  = float(target_pos[1]) - float(s_x) / 2
    scale   = float(model_sz) / float(s_x)
    cx = (fx - sx_left) * scale
    cy = (fy - sy_top)  * scale
    cw = fw * scale
    ch = fh * scale
    cx0 = max(cx, 0.0); cy0 = max(cy, 0.0)
    cx1 = min(cx + cw, float(model_sz))
    cy1 = min(cy + ch, float(model_sz))
    return [cx0, cy0, max(cx1 - cx0, 0.0), max(cy1 - cy0, 0.0)]


def make_crop_mask(crop_h, crop_w, crop_bbox, device, soft_edge=2.0):
    """
    (1, 1, H, W) float mask, 1 inside crop_bbox, 0 outside, with a
    sigmoid-falloff edge of `soft_edge` pixels so gradients near the
    bbox boundary are non-zero.
    """
    cx, cy, cw, ch = crop_bbox
    if cw <= 0 or ch <= 0:
        return torch.zeros(1, 1, crop_h, crop_w, device=device)
    ys = torch.arange(crop_h, device=device, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(crop_w, device=device, dtype=torch.float32).view(1, -1)
    dx = torch.minimum(xs - cx, (cx + cw) - xs)
    dy = torch.minimum(ys - cy, (cy + ch) - ys)
    d = torch.minimum(dx, dy)
    m = torch.sigmoid(d / max(soft_edge, 1e-3))
    return m.view(1, 1, crop_h, crop_w)


# ---------------------------------------------------------------------------
# DoG response on the search crop — zero-gap with OpenCV
# ---------------------------------------------------------------------------

def _gaussian_kernel1d(sigma, device):
    ksize = max(int(round(6 * sigma)) | 1, 3)
    half = ksize // 2
    x = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _gaussian_blur(x, sigma):
    k = _gaussian_kernel1d(sigma, x.device)
    ksize = k.numel(); pad = ksize // 2
    k_row = k.view(1, 1, 1, ksize)
    k_col = k.view(1, 1, ksize, 1)
    x = F.conv2d(x, k_row, padding=(0, pad))
    x = F.conv2d(x, k_col, padding=(pad, 0))
    return x


def dog_response_crop(x_adv, sigma1=1.0, sigma2=1.6):
    """
    Differentiable DoG response on a (1, 3, H, W) BGR float crop in [0, 255].

    Returns (1, 1, H, W) DoG response = G(sigma1)·gray − G(sigma2)·gray on
    a normalised [0, 1] grayscale image.

    sigma2=1.6 mirrors the defense's `cv2.SIFT_create(sigma=1.6)`. The
    response at a pixel is the same value the defense's SIFT pyramid
    sees at that scale, modulo floating-point.
    """
    # BGR -> gray (OpenCV order; x_crop comes from cv2.imread via im_to_torch)
    gray = (x_adv[:, 0:1] * 0.114
            + x_adv[:, 1:2] * 0.587
            + x_adv[:, 2:3] * 0.299) / 255.0
    return _gaussian_blur(gray, sigma1) - _gaussian_blur(gray, sigma2)


def sift_dog_suppress_loss(x_adv, r_target_crop_bbox,
                           contrast=0.04, soft_edge=2.0):
    """
    Mean over R_target of relu(|DoG| - contrast).

    Drives the DoG response below the contrast threshold (0.04 matches the
    defense's `cv2.SIFT_create(contrastThreshold=0.04)`), so OpenCV's
    keypoint detector finds nothing matchable inside R_target.

    Returns a scalar. Zero surrogate gap.
    """
    R = dog_response_crop(x_adv).abs()
    _, _, H, W = R.shape
    mask = make_crop_mask(H, W, r_target_crop_bbox, x_adv.device, soft_edge)
    excess = F.relu(R - contrast) * mask
    excess = R * mask
    return excess.sum() / mask.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Kornia SIFT suppress — BPDA surrogate (gamma term)
# ---------------------------------------------------------------------------

_KORNIA_SIFT = None


def _get_kornia_sift(num_features=500):
    global _KORNIA_SIFT
    if _KORNIA_SIFT is None:
        import kornia.feature as KF
        _KORNIA_SIFT = (KF.SIFTFeature(num_features=num_features, rootsift=False)
                        .cuda().eval())
    return _KORNIA_SIFT


def sift_kornia_suppress_loss(x_adv, r_target_crop_bbox,
                              soft_edge=2.0, num_features=500):
    """
    Mean Kornia-SIFT response weighted by "fraction inside R_target".

    BPDA surrogate: minimising this minimises Kornia's response at kps
    that land inside R_target. Transfer to OpenCV is empirical (the
    surrogate gap is non-trivial — calibration showed ~52% kp co-loc).
    Pair with the DoG-suppress term (zero-gap) rather than relying on
    this alone.
    """
    sift = _get_kornia_sift(num_features)
    gray = (x_adv[:, 0:1] * 0.114
            + x_adv[:, 1:2] * 0.587
            + x_adv[:, 2:3] * 0.299) / 255.0
    lafs, resps, _ = sift(gray)
    kp_xy = lafs[0, :, :, 2]      # (N, 2) in crop coords
    resp  = resps[0]               # (N,)
    cx, cy, cw, ch = r_target_crop_bbox
    if cw <= 0 or ch <= 0:
        return torch.zeros((), device=x_adv.device)
    dx = torch.minimum(kp_xy[:, 0] - cx, (cx + cw) - kp_xy[:, 0])
    dy = torch.minimum(kp_xy[:, 1] - cy, (cy + ch) - kp_xy[:, 1])
    d  = torch.minimum(dx, dy)
    in_mask = torch.sigmoid(d / max(soft_edge, 1e-3))
    return (resp.abs() * in_mask).sum() / in_mask.sum().clamp(min=1.0)
