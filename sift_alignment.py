"""
SIFT-based alignment score for adversarial perturbation detection.

Core idea: between two adjacent frames, genuine scene motion produces
geometrically consistent SIFT correspondences. An adversarial patch
disrupts local texture, breaking this consistency — especially in the
region of the tracked target.

Usage:
    detector = SIFTAlignmentDetector()
    result = detector.compute_alignment_score(frame1, frame2, bbox=[x, y, w, h])

    if result.is_adversarial:
        adapt_strength = result.adaptation_strength  # in [0, 1]
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """
    Full breakdown of the inter-frame alignment score.

    alignment_score : float in [0, 1]
        1 = perfectly aligned (benign), 0 = completely disrupted (adversarial).
        This is the primary scalar you should threshold.

    Sub-scores (all in [0, 1], higher = more benign):
        match_ratio          -- fraction of keypoints that found a good match
        inlier_ratio         -- fraction of matches that are geometrically consistent (RANSAC)
        descriptor_sim       -- 1 - normalised mean descriptor distance
        kp_stability         -- 1 - normalised |ΔN_keypoints| between frames

    ROI diagnostics (populated only when bbox is provided):
        roi_alignment_score  -- alignment score inside the tracked bounding box
        bg_alignment_score   -- alignment score outside the bounding box
        roi_bg_gap           -- bg_score - roi_score  (positive → ROI is disrupted)

    Decision outputs:
        is_adversarial       -- True when alignment_score < adversarial_threshold
        adaptation_strength  -- continuous value in [0, 1] for how aggressively to adapt;
                                derived from alignment_score via a centred sigmoid so that
                                lower alignment → stronger adaptation.
    """
    alignment_score: float
    match_ratio: float
    inlier_ratio: float
    descriptor_sim: float
    kp_stability: float
    roi_alignment_score: Optional[float]
    bg_alignment_score: Optional[float]
    roi_bg_gap: Optional[float]
    is_adversarial: bool
    adaptation_strength: float

    def __str__(self) -> str:
        lines = [
            f"AlignmentResult:",
            f"  alignment_score    = {self.alignment_score:.4f}",
            f"  match_ratio        = {self.match_ratio:.4f}",
            f"  inlier_ratio       = {self.inlier_ratio:.4f}",
            f"  descriptor_sim     = {self.descriptor_sim:.4f}",
            f"  kp_stability       = {self.kp_stability:.4f}",
        ]
        if self.roi_alignment_score is not None:
            lines += [
                f"  roi_alignment      = {self.roi_alignment_score:.4f}",
                f"  bg_alignment       = {self.bg_alignment_score:.4f}",
                f"  roi_bg_gap         = {self.roi_bg_gap:.4f}",
            ]
        lines += [
            f"  is_adversarial     = {self.is_adversarial}",
            f"  adaptation_strength= {self.adaptation_strength:.4f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class SIFTAlignmentDetector:
    """
    Computes low-level feature alignment between two adjacent frames and
    produces a continuous adversarial score + a binary detection decision.

    Parameters
    ----------
    n_features : int
        Maximum number of SIFT keypoints to retain (0 = unlimited).
    contrast_threshold : float
        Lower → more keypoints (more sensitive to weak features).
    edge_threshold : float
        Higher → fewer edge-like keypoints filtered out.
    sigma : float
        Initial Gaussian blur for scale-space construction.
    lowe_ratio : float
        Threshold for Lowe's ratio test (typically 0.7–0.8).
    min_matches : int
        Minimum good matches required to attempt homography estimation.
    adversarial_threshold : float
        alignment_score below this → classified as adversarial.
    adaptation_k : float
        Steepness of the sigmoid used to derive adaptation_strength.
        Larger k → sharper transition around adversarial_threshold.
    roi_weight : float
        How much weight to give the ROI-background alignment gap when
        it is available (0 = ignore ROI, 1 = rely entirely on ROI).
    """

    def __init__(
        self,
        n_features: int = 0,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        sigma: float = 1.6,
        lowe_ratio: float = 0.75,
        min_matches: int = 4,
        adversarial_threshold: float = 0.40,
        adaptation_k: float = 10.0,
        roi_weight: float = 0.35,
    ):
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.lowe_ratio = lowe_ratio
        self.min_matches = min_matches
        self.adversarial_threshold = adversarial_threshold
        self.adaptation_k = adaptation_k
        self.roi_weight = roi_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(
        self, frame: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Extract SIFT keypoints and descriptors from a single frame.

        Args:
            frame : H×W×C uint8 BGR image, or H×W grayscale.
            mask  : Optional uint8 mask (255 = detect here, 0 = ignore).

        Returns:
            keypoints   : list of cv2.KeyPoint
            descriptors : float32 array of shape (N, 128), or None if no kps.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        kps, descs = self.sift.detectAndCompute(gray, mask)
        return kps, descs

    def compute_alignment_score(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        bbox: Optional[np.ndarray] = None,
    ) -> AlignmentResult:
        """
        Compute the inter-frame alignment score and adversarial classification.

        Args:
            frame1 : Previous frame, H×W×C uint8 BGR.
            frame2 : Current frame,  H×W×C uint8 BGR.
            bbox   : Optional [x, y, w, h] bounding box of the tracked target
                     in frame coordinates.  When provided, ROI vs. background
                     scores are computed and factored into the final score.

        Returns:
            AlignmentResult (see class docstring).
        """
        kp1, desc1 = self.extract_features(frame1)
        kp2, desc2 = self.extract_features(frame2)

        # ---- Global sub-scores ----------------------------------------
        good = self._ratio_match(desc1, desc2)

        n_kps = max(len(kp1), len(kp2), 1)
        match_ratio = min(len(good) / n_kps, 1.0)

        inlier_ratio = self._ransac_inlier_ratio(kp1, kp2, good)

        descriptor_sim = self._descriptor_similarity(good)

        # Keypoint count stability: penalise large |ΔN|
        kp_stability = 1.0 - float(
            np.clip(abs(len(kp2) - len(kp1)) / max(len(kp1), 1), 0.0, 1.0)
        )

        # ---- Global alignment score (weighted combination) -------------
        #
        # Weight rationale:
        #   match_ratio    — primary signal: few matches = disrupted texture
        #   inlier_ratio   — geometric consistency signal (motion coherence)
        #   descriptor_sim — quality of existing matches
        #   kp_stability   — secondary: sudden kp count changes indicate patch
        global_score = (
            0.35 * match_ratio
            + 0.35 * inlier_ratio
            + 0.20 * descriptor_sim
            + 0.10 * kp_stability
        )
        global_score = float(np.clip(global_score, 0.0, 1.0))

        # ---- ROI diagnostics ------------------------------------------
        roi_score = bg_score = roi_bg_gap = None
        if bbox is not None:
            roi_mask = self._bbox_mask(frame1.shape, bbox)
            bg_mask = cv2.bitwise_not(roi_mask)

            roi_kp1, roi_d1 = self._filter_by_mask(kp1, desc1, roi_mask)
            roi_kp2, roi_d2 = self._filter_by_mask(kp2, desc2, roi_mask)
            bg_kp1, bg_d1 = self._filter_by_mask(kp1, desc1, bg_mask)
            bg_kp2, bg_d2 = self._filter_by_mask(kp2, desc2, bg_mask)

            roi_score = self._region_score(roi_kp1, roi_d1, roi_kp2, roi_d2)
            bg_score = self._region_score(bg_kp1, bg_d1, bg_kp2, bg_d2)
            roi_bg_gap = float(np.clip(bg_score - roi_score, 0.0, 1.0))

            # Blend: if ROI is significantly less aligned than background,
            # penalise the global score proportionally.
            global_score = global_score * (1.0 - self.roi_weight * roi_bg_gap)
            global_score = float(np.clip(global_score, 0.0, 1.0))

        # ---- Decision outputs -----------------------------------------
        is_adversarial = global_score < self.adversarial_threshold

        # Sigmoid centred at adversarial_threshold:
        #   score → threshold  ⟹  strength ≈ 0.5
        #   score ≪ threshold  ⟹  strength → 1.0  (adapt aggressively)
        #   score ≫ threshold  ⟹  strength → 0.0  (no adaptation needed)
        raw = float(
            1.0 / (1.0 + np.exp(self.adaptation_k * (global_score - self.adversarial_threshold)))
        )
        adaptation_strength = float(np.clip(raw, 0.0, 1.0))

        return AlignmentResult(
            alignment_score=global_score,
            match_ratio=match_ratio,
            inlier_ratio=inlier_ratio,
            descriptor_sim=descriptor_sim,
            kp_stability=kp_stability,
            roi_alignment_score=roi_score,
            bg_alignment_score=bg_score,
            roi_bg_gap=roi_bg_gap,
            is_adversarial=is_adversarial,
            adaptation_strength=adaptation_strength,
        )

    def visualize(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        result: AlignmentResult,
        bbox: Optional[np.ndarray] = None,
        max_draw: int = 50,
    ) -> np.ndarray:
        """
        Draw matched keypoints side-by-side for inspection.

        Returns an annotated BGR image (frame1 | frame2 with matches).
        """
        kp1, desc1 = self.extract_features(frame1)
        kp2, desc2 = self.extract_features(frame2)
        good = self._ratio_match(desc1, desc2)

        drawn = cv2.drawMatches(
            frame1, kp1, frame2, kp2,
            good[:max_draw], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # Annotate with score
        label = (
            f"alignment={result.alignment_score:.3f} | "
            f"adv={'YES' if result.is_adversarial else 'no'} | "
            f"adapt={result.adaptation_strength:.3f}"
        )
        cv2.putText(drawn, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0) if not result.is_adversarial else (0, 0, 255), 2)

        # Draw bbox on both halves if provided
        if bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(drawn, (x, y), (x + w, y + h), (255, 128, 0), 2)
            W = frame1.shape[1]
            cv2.rectangle(drawn, (W + x, y), (W + x + w, y + h), (255, 128, 0), 2)

        return drawn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ratio_match(
        self,
        desc1: Optional[np.ndarray],
        desc2: Optional[np.ndarray],
    ) -> List[cv2.DMatch]:
        """Lowe's ratio-test matching. Returns list of good DMatch objects."""
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        raw = self.matcher.knnMatch(desc1, desc2, k=2)
        return [m for m, n in raw if m.distance < self.lowe_ratio * n.distance]

    def _ransac_inlier_ratio(
        self,
        kp1: list,
        kp2: list,
        good: list,
    ) -> float:
        """
        Estimate a homography between matched keypoints and return the
        RANSAC inlier fraction.  Returns 0 if too few matches exist.

        Note: a pure translation / affine motion is also a valid homography,
        so benign camera motion will still give a high inlier ratio.
        Adversarial patches create a geometrically inconsistent cluster of
        matches that RANSAC will classify as outliers.
        """
        if len(good) < self.min_matches:
            return 0.0
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=5.0)
        if mask is None:
            return 0.0
        return float(mask.ravel().sum() / len(good))

    def _descriptor_similarity(self, good: list) -> float:
        """
        1 - normalised mean descriptor distance of matched pairs.
        SIFT uses L2 distance; raw values are roughly in [0, 512].
        We normalise by 300 (empirically chosen as a high-distance reference).
        Returns 1.0 (most similar) down to 0.0 (most dissimilar).
        """
        if not good:
            return 0.0
        mean_dist = float(np.mean([m.distance for m in good]))
        return float(np.clip(1.0 - mean_dist / 300.0, 0.0, 1.0))

    def _bbox_mask(self, shape: tuple, bbox: np.ndarray) -> np.ndarray:
        """Return uint8 mask (255 inside bbox, 0 outside)."""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        x, y, w, h = [int(v) for v in bbox]
        x1, y1 = max(x, 0), max(y, 0)
        x2 = min(x + w, shape[1])
        y2 = min(y + h, shape[0])
        mask[y1:y2, x1:x2] = 255
        return mask

    def _filter_by_mask(
        self,
        kps: list,
        descs: Optional[np.ndarray],
        mask: np.ndarray,
    ) -> Tuple[list, Optional[np.ndarray]]:
        """Return only keypoints (and corresponding descriptors) inside mask."""
        if descs is None or len(kps) == 0:
            return [], None
        keep = []
        for i, kp in enumerate(kps):
            px, py = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]:
                if mask[py, px] > 0:
                    keep.append(i)
        if not keep:
            return [], None
        return [kps[i] for i in keep], descs[keep]

    def _region_score(
        self,
        kp1: list,
        desc1: Optional[np.ndarray],
        kp2: list,
        desc2: Optional[np.ndarray],
    ) -> float:
        """Compact alignment score for a single region (match_ratio + inlier_ratio blend)."""
        good = self._ratio_match(desc1, desc2)
        n = max(len(kp1), len(kp2), 1)
        mr = min(len(good) / n, 1.0)
        ir = self._ransac_inlier_ratio(kp1, kp2, good)
        ds = self._descriptor_similarity(good)
        return float(np.clip(0.4 * mr + 0.4 * ir + 0.2 * ds, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def compute_sift_alignment(
    frame1: np.ndarray,
    frame2: np.ndarray,
    bbox: Optional[np.ndarray] = None,
    **kwargs,
) -> AlignmentResult:
    """
    One-shot helper: create a detector with default settings and score two frames.

    Extra keyword arguments are forwarded to SIFTAlignmentDetector.__init__.

    Example
    -------
    >>> result = compute_sift_alignment(prev_frame, curr_frame, bbox=[x, y, w, h])
    >>> print(result)
    """
    detector = SIFTAlignmentDetector(**kwargs)
    return detector.compute_alignment_score(frame1, frame2, bbox)
