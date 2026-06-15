#!/usr/bin/env python3
"""
experiment_sweep_sift_attack.py

Multi-video / multi-seed sweep over experiment_sift_attack.py.

Companion to experiment_sweep.py, but for the SIFT-attack vulnerability
study (no defense pool / SIFT rerank). For each (video, seed) pair, runs
experiment_sift_attack.py as a subprocess and aggregates two quantities
across runs:

  - tracker IoU      = mean over frames of IoU(pred_bbox, gt_bbox)
                       (benign + attack, so you see the hijack delta)
  - removal_rate     = mean over frames of (1 - kp_attacked / kp_clean)
                       inside R_target  (primary SIFT-attack signal)

Per-attack-variant logs are stem-tagged so multiple sweeps over the same
videos do not clobber each other. Re-runs are skipped when the log
already exists, so re-invoking the same command resumes the sweep.

Usage
-----
  # Default: rtaa_sift_frame on all VOT2018 videos, 5 seeds each
  python experiment_sweep_sift_attack.py

  # Specific videos, eps/n_iter variations
  python experiment_sweep_sift_attack.py --videos car1 racing \
      --n_seeds 10 --attack rtaa_sift_frame --eps 16 --n_iter 20

  # Sparse-mask vs dense gradient attack head-to-head (run twice, compare)
  python experiment_sweep_sift_attack.py --attack rtaa_sift_frame --eps 16 \
      --out_dir out/sweep_sift/dense
  python experiment_sweep_sift_attack.py --attack rtaa_sift_frame --eps 16 \
      --sparse_mask --out_dir out/sweep_sift/sparse

  # Amerini ceiling
  python experiment_sweep_sift_attack.py --attack amerini_smoothing \
      --out_dir out/sweep_sift/amerini

  # Re-aggregate existing logs without re-running
  python experiment_sweep_sift_attack.py --analyse_only \
      --out_dir out/sweep_sift/sparse
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from os.path import realpath, dirname, join

import numpy as np


# ---------------------------------------------------------------------------
# Metric computation from a single per-run log
# ---------------------------------------------------------------------------

def _safe_load_scalar(d, key, default=float('nan')):
    return float(d[key]) if key in d.files else default


def compute_metrics(log_path):
    """Aggregate one per-run log (averaging over frames).

    Returns dict; all values are scalars (mean over frames within a run).
    """
    d = np.load(log_path, allow_pickle=True)

    benign_iou  = float(np.nanmean(d['benign_iou_gt']))
    attack_iou  = float(np.nanmean(d['attack_iou_gt']))
    rr          = float(np.nanmean(d['attack_removal_rate']))
    rr_gt       = float(np.nanmean(d['attack_removal_rate_gt']))
    kp_clean    = float(np.nanmean(d['attack_kp_clean']))
    kp_attacked = float(np.nanmean(d['attack_kp_attacked']))

    # Loss traces: present for gradient variants, NaN for amerini.
    def _last_finite_per_frame(arr):
        """arr is (NF, n_iter); pick the last non-NaN value per row, then mean."""
        out = []
        for row in arr:
            finite = row[np.isfinite(row)]
            if len(finite) > 0:
                out.append(float(finite[-1]))
        return float(np.mean(out)) if out else float('nan')

    out = {
        'attack_variant':    str(d['attack_variant']),
        'eps':               float(d['eps']),
        'n_iter':            int(d['n_iter']),
        'roi_source':        str(d['roi_source']),
        'benign_iou':        benign_iou,
        'attack_iou':        attack_iou,
        'iou_drop':          attack_iou - benign_iou,
        'removal_rate':      rr,
        'removal_rate_gt':   rr_gt,
        'kp_clean':          kp_clean,
        'kp_attacked':       kp_attacked,
        'L_dog_final':       _last_finite_per_frame(d['loss_dog']),
        'L_rtaa_final':      _last_finite_per_frame(d['loss_rtaa']),
    }

    # Per-frame Amerini magnitude diagnostics (NaN for non-amerini logs OR
    # for older amerini logs written before save_log was extended). Each
    # value here is meaned across frames within this single run.
    def _mean(key):
        return float(np.nanmean(d[key])) if key in d.files else float('nan')

    def _max(key):
        return float(np.nanmax(d[key])) if key in d.files else float('nan')

    # Per-frame pixel-magnitude diagnostics: generic, populated for every
    # attack variant except 'none'. Lets gradient and amerini attacks be
    # compared on the same axis as --eps.
    out['perturbation_linf_mean'] = _mean('perturbation_linf')
    out['perturbation_linf_max']  = _max ('perturbation_linf')
    out['perturbation_l1_mean']   = _mean('perturbation_l1_mean')
    out['perturbation_frac']      = _mean('perturbation_frac')

    # Amerini-only iteration count
    out['amerini_iters']          = _mean('amerini_iters')

    out['sparse_n_kps']             = _mean('sparse_n_kps')
    out['sparse_area_frac']         = _mean('sparse_area_frac')
    out['sparse_n_refreshes']       = _mean('sparse_n_refreshes')
    out['sparse_area_frac_final']   = _mean('sparse_area_frac_final')

    # Gradient-alignment diagnostics (NaN unless --diag_grad_alignment was
    # set in the per-run experiment). Per-iter arrays are (NF, n_iter);
    # collapse to start-iter / end-iter / overall scalars.
    def _start_end_all(key):
        if key not in d.files:
            return (float('nan'), float('nan'), float('nan'))
        a = d[key]
        if a.ndim != 2 or a.size == 0:
            return (float('nan'), float('nan'), float('nan'))
        return (float(np.nanmean(a[:, 0])),
                float(np.nanmean(a[:, -1])),
                float(np.nanmean(a)))
    cs0, csL, csM = _start_end_all('grad_cos_sim')
    sa0, saL, saM = _start_end_all('grad_sign_agree')
    nr0, nrL, nrM = _start_end_all('grad_norm_ratio')
    out['grad_cos_sim_start']    = cs0
    out['grad_cos_sim_end']      = csL
    out['grad_cos_sim_mean']     = csM
    out['grad_sign_agree_start'] = sa0
    out['grad_sign_agree_end']   = saL
    out['grad_sign_agree_mean']  = saM
    out['grad_norm_ratio_start'] = nr0
    out['grad_norm_ratio_end']   = nrL
    out['grad_norm_ratio_mean']  = nrM
    return out


# ---------------------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------------------

def _variant_tag(args):
    """Stem tag encoding the attack config so sweeps don't collide."""
    parts = [args.attack, f'eps{int(args.eps)}', f'it{int(args.n_iter)}']
    if args.attack in ('rtaa_sift_crop', 'rtaa_sift_frame'):
        parts.append(f'ad{int(args.alpha_dog)}')
        parts.append(f'gk{args.gamma_kornia:g}')
        parts.append(f'dc{args.dog_contrast:g}')
        parts.append(f'rw{args.rtaa_weight:g}')
    if args.attack == 'rtaa_sift_frame' and args.sparse_mask:
        # 'smk' = sparse-mask-on-kp-gradient-only (gates g_dog/g_kornia, not
        # g_rtaa). Older 'sm{N}'-tagged logs used a different semantic (mask
        # applied to final delta, constraining both objectives); keep the tag
        # distinct so prior sweeps stay loadable as a comparison baseline.
        parts.append(f'smk{args.sparse_half_side}')
        if args.sparse_refresh_every > 0:
            parts.append(f'rf{args.sparse_refresh_every}c{args.sparse_refresh_cap}')
    if args.attack == 'amerini_smoothing':
        parts.append(f'as{args.amerini_sigma:g}')
        parts.append(f'ah{args.amerini_patch_half}')
        parts.append(f'ai{args.amerini_max_iter}')
    if args.attack == 'rtaa_amerini':
        parts.append(f'rw{args.rtaa_weight:g}')
        parts.append(f'as{args.amerini_sigma:g}')
        parts.append(f'ah{args.amerini_patch_half}')
    if args.attack == 'rtaa_then_amerini':
        parts.append(f'as{args.amerini_sigma:g}')
        parts.append(f'ah{args.amerini_patch_half}')
        parts.append(f'ai{args.amerini_max_iter}')
    if args.attack == 'rtaa_sift_then_amerini':
        # Stage-1 PGD args (joint L_rtaa + L_dog) + stage-2 Amerini cap.
        parts.append(f'ad{int(args.alpha_dog)}')
        parts.append(f'dc{args.dog_contrast:g}')
        parts.append(f'rw{args.rtaa_weight:g}')
        if args.sparse_mask:
            parts.append(f'smk{args.sparse_half_side}')
            if args.sparse_refresh_every > 0:
                parts.append(f'rf{args.sparse_refresh_every}c{args.sparse_refresh_cap}')
        parts.append(f'as{args.amerini_sigma:g}')
        parts.append(f'ah{args.amerini_patch_half}')
        parts.append(f'ai{args.amerini_max_iter}')
    if args.rtaa_mask_gt and args.attack in (
            'rtaa_sift_frame', 'rtaa_amerini',
            'rtaa_then_amerini', 'rtaa_sift_then_amerini'):
        parts.append('rmgt')
    return '_'.join(parts)


def run_experiment(video, seed, args):
    """Call experiment_sift_attack.py as a subprocess. Returns log path or None."""
    tag      = _variant_tag(args)
    stem     = f"log_{video}_s{seed}_{tag}"
    log_path = join(args.out_dir, f"{stem}.npz")
    if os.path.exists(log_path):
        return log_path

    cmd = [
        sys.executable,
        join(dirname(realpath(__file__)), 'experiment_sift_attack.py'),
        '--dataset',     args.dataset,
        '--video',       video,
        '--model',       args.model,
        '--out_dir',     args.out_dir,
        '--out_stem',    stem,
        '--n_frames',    str(args.n_frames),
        '--seed',        str(seed),
        '--attack',      args.attack,
        '--eps',         str(args.eps),
        '--n_iter',      str(args.n_iter),
        '--alpha_dog',   str(args.alpha_dog),
        '--gamma_kornia', str(args.gamma_kornia),
        '--dog_contrast', str(args.dog_contrast),
        '--rtaa_weight', str(args.rtaa_weight),
        '--roi_source',  args.roi_source,
        '--no_video',                                 # skip rendering for speed
    ]
    if args.attack == 'rtaa_sift_frame' and args.sparse_mask:
        cmd += ['--sparse_mask',
                '--sparse_half_side',     str(args.sparse_half_side),
                '--sparse_refresh_every', str(args.sparse_refresh_every),
                '--sparse_refresh_cap',   str(args.sparse_refresh_cap)]
    if args.attack == 'rtaa_sift_frame' and args.diag_grad_alignment:
        cmd += ['--diag_grad_alignment']
    if args.rtaa_mask_gt and args.attack in (
            'rtaa_sift_frame', 'rtaa_amerini',
            'rtaa_then_amerini', 'rtaa_sift_then_amerini'):
        cmd += ['--rtaa_mask_gt']
    if args.save_viz:
        cmd += ['--save_viz']
    if args.attack == 'amerini_smoothing':
        cmd += [
            '--amerini_sigma',          str(args.amerini_sigma),
            '--amerini_ksize',          str(args.amerini_ksize),
            '--amerini_patch_half',     str(args.amerini_patch_half),
            '--amerini_max_iter',       str(args.amerini_max_iter),
            '--amerini_target_removal', str(args.amerini_target_removal),
        ]
    if args.attack == 'rtaa_amerini':
        cmd += [
            '--amerini_sigma',      str(args.amerini_sigma),
            '--amerini_ksize',      str(args.amerini_ksize),
            '--amerini_patch_half', str(args.amerini_patch_half),
        ]
    if args.attack == 'rtaa_then_amerini':
        cmd += [
            '--amerini_sigma',          str(args.amerini_sigma),
            '--amerini_ksize',          str(args.amerini_ksize),
            '--amerini_patch_half',     str(args.amerini_patch_half),
            '--amerini_max_iter',       str(args.amerini_max_iter),
            '--amerini_target_removal', str(args.amerini_target_removal),
        ]
    if args.attack == 'rtaa_sift_then_amerini':
        cmd += [
            '--amerini_sigma',          str(args.amerini_sigma),
            '--amerini_ksize',          str(args.amerini_ksize),
            '--amerini_patch_half',     str(args.amerini_patch_half),
            '--amerini_max_iter',       str(args.amerini_max_iter),
            '--amerini_target_removal', str(args.amerini_target_removal),
        ]
        if args.sparse_mask:
            cmd += ['--sparse_mask',
                    '--sparse_half_side',     str(args.sparse_half_side),
                    '--sparse_refresh_every', str(args.sparse_refresh_every),
                    '--sparse_refresh_cap',   str(args.sparse_refresh_cap)]
        if args.diag_grad_alignment:
            cmd += ['--diag_grad_alignment']

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED: {video} seed={seed}")
        print(r.stderr[-800:])
        return None
    return log_path


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def list_videos(dataset_name, data_dir):
    info = json.load(open(join(data_dir, dataset_name + '.json')))
    return sorted(v['name'] for v in info.values())


# ---------------------------------------------------------------------------
# Summary printers
# ---------------------------------------------------------------------------

def _print_perturbation_block(results, indent='    '):
    """Print per-frame magnitude diagnostics (any attack variant)."""
    def _mean(k): return np.nanmean([r[k] for r in results])
    def _max (k): return np.nanmax ([r[k] for r in results])
    if not np.isfinite(_mean('perturbation_linf_mean')):
        return
    print(f"{indent}perturbation magnitude (== implied L_inf eps):")
    print(f"{indent}  max |delta| per frame:   "
          f"mean={_mean('perturbation_linf_mean'):.1f}   "
          f"worst-frame max={_max('perturbation_linf_max'):.1f}")
    print(f"{indent}  mean |delta| in touched: {_mean('perturbation_l1_mean'):.2f}")
    print(f"{indent}  fraction frame touched:  "
          f"{_mean('perturbation_frac') * 100:.2f}%")
    if np.isfinite(_mean('amerini_iters')):
        print(f"{indent}  amerini outer iters/frame: {_mean('amerini_iters'):.1f}")


def _print_grad_align_block(results, indent='    '):
    """Print L_rtaa vs L_dog gradient alignment summary (when populated)."""
    def _mean(k): return np.nanmean([r[k] for r in results])
    if not np.isfinite(_mean('grad_cos_sim_mean')):
        return
    print(f"{indent}L_rtaa vs L_dog gradient alignment "
          f"(mean across runs, start->end PGD):")
    print(f"{indent}  cos sim    : "
          f"{_mean('grad_cos_sim_start'):+.3f} -> "
          f"{_mean('grad_cos_sim_end'):+.3f}   "
          f"(all-iter mean = {_mean('grad_cos_sim_mean'):+.3f})")
    print(f"{indent}  sign agree : "
          f"{_mean('grad_sign_agree_start'):.3f} -> "
          f"{_mean('grad_sign_agree_end'):.3f}   "
          f"(0.5 = random; >0.5 = agree)")
    print(f"{indent}  |g_dog|/|g_rtaa| : "
          f"{_mean('grad_norm_ratio_start'):.3g} -> "
          f"{_mean('grad_norm_ratio_end'):.3g}")


def _print_sparse_block(results, indent='    '):
    """Print sparse-mask diagnostics (when non-NaN)."""
    def _mean(k): return np.nanmean([r[k] for r in results])
    if not np.isfinite(_mean('sparse_n_kps')):
        return
    print(f"{indent}sparse mask: kps/frame={_mean('sparse_n_kps'):.1f}   "
          f"crop area perturbable={_mean('sparse_area_frac') * 100:.1f}% (init)")
    if np.isfinite(_mean('sparse_n_refreshes')):
        init_area  = _mean('sparse_area_frac')
        final_area = _mean('sparse_area_frac_final')
        print(f"{indent}             refreshes/frame={_mean('sparse_n_refreshes'):.1f}   "
              f"perturbable after refresh={final_area * 100:.1f}% "
              f"(growth: {final_area / max(init_area, 1e-9):.2f}x)")


def _print_video_summary(video, results):
    n = len(results)
    def _m(k): return np.nanmean([r[k] for r in results])
    def _s(k): return np.nanstd ([r[k] for r in results])
    print(f"\n  {video}  (n={n} seeds)")
    print(f"    benign IoU      = {_m('benign_iou'):.3f} ± {_s('benign_iou'):.3f}")
    print(f"    attack IoU      = {_m('attack_iou'):.3f} ± {_s('attack_iou'):.3f}   "
          f"(drop = {_m('iou_drop'):+.3f})")
    print(f"    removal_rate    = {_m('removal_rate'):.3f} ± {_s('removal_rate'):.3f}   "
          f"(GT-box: {_m('removal_rate_gt'):.3f})")
    print(f"    kp_clean → atk  = {_m('kp_clean'):.1f} → {_m('kp_attacked'):.1f}")
    _print_perturbation_block(results, indent='    ')
    _print_sparse_block      (results, indent='    ')
    _print_grad_align_block  (results, indent='    ')


def _print_aggregate(results, attack_label):
    n = len(results)
    def _m(k): return np.nanmean([r[k] for r in results])
    def _s(k): return np.nanstd ([r[k] for r in results])
    print(f"\n{'=' * 60}")
    print(f"AGGREGATE  [{attack_label}]   n = {n} (video, seed) pairs")
    print(f"{'=' * 60}")
    print(f"  benign  IoU   : {_m('benign_iou'):.3f} ± {_s('benign_iou'):.3f}")
    print(f"  attack  IoU   : {_m('attack_iou'):.3f} ± {_s('attack_iou'):.3f}")
    print(f"  IoU drop      : {_m('iou_drop'):+.3f} ± {_s('iou_drop'):.3f}   "
          f"(positive = tracker hijacked)")
    print(f"  removal_rate  : {_m('removal_rate'):.3f} ± {_s('removal_rate'):.3f}   "
          f"(R_target ROI)")
    print(f"  removal_rate  : {_m('removal_rate_gt'):.3f} ± {_s('removal_rate_gt'):.3f}   "
          f"(GT ROI)")
    print(f"  kp_clean      : {_m('kp_clean'):.1f}    "
          f"kp_attacked    : {_m('kp_attacked'):.1f}")
    if np.isfinite(_m('L_dog_final')):
        print(f"  L_dog (final iter, mean over frames/runs): {_m('L_dog_final'):.4g}")
    if np.isfinite(_m('L_rtaa_final')):
        print(f"  L_rtaa (final iter, mean over frames/runs): {_m('L_rtaa_final'):+.3f}")
    _print_perturbation_block(results, indent='  ')
    _print_sparse_block      (results, indent='  ')
    _print_grad_align_block  (results, indent='  ')


def _save_results(results, out_dir, tag):
    csv_path = join(out_dir, f'sweep_{tag}.csv')
    fields = ['video', 'seed', 'attack_variant', 'eps', 'n_iter', 'roi_source',
              'benign_iou', 'attack_iou', 'iou_drop',
              'removal_rate', 'removal_rate_gt',
              'kp_clean', 'kp_attacked', 'L_dog_final', 'L_rtaa_final',
              'perturbation_linf_mean', 'perturbation_linf_max',
              'perturbation_l1_mean', 'perturbation_frac',
              'amerini_iters', 'sparse_n_kps', 'sparse_area_frac',
              'sparse_n_refreshes', 'sparse_area_frac_final',
              'grad_cos_sim_start', 'grad_cos_sim_end', 'grad_cos_sim_mean',
              'grad_sign_agree_start', 'grad_sign_agree_end', 'grad_sign_agree_mean',
              'grad_norm_ratio_start', 'grad_norm_ratio_end', 'grad_norm_ratio_mean']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    print(f"\nCSV  -> {csv_path}")

    npz_path = join(out_dir, f'sweep_{tag}.npz')
    np.savez(
        npz_path,
        videos                  = np.array([r['video']           for r in results]),
        seeds                   = np.array([r['seed']            for r in results]),
        attack_variant          = np.array([r['attack_variant']  for r in results]),
        benign_iou              = np.array([r['benign_iou']      for r in results]),
        attack_iou              = np.array([r['attack_iou']      for r in results]),
        iou_drop                = np.array([r['iou_drop']        for r in results]),
        removal_rate            = np.array([r['removal_rate']    for r in results]),
        removal_rate_gt         = np.array([r['removal_rate_gt'] for r in results]),
        kp_clean                = np.array([r['kp_clean']        for r in results]),
        kp_attacked             = np.array([r['kp_attacked']     for r in results]),
        L_dog_final             = np.array([r['L_dog_final']            for r in results]),
        L_rtaa_final            = np.array([r['L_rtaa_final']           for r in results]),
        perturbation_linf_mean  = np.array([r['perturbation_linf_mean'] for r in results]),
        perturbation_linf_max   = np.array([r['perturbation_linf_max']  for r in results]),
        perturbation_l1_mean    = np.array([r['perturbation_l1_mean']   for r in results]),
        perturbation_frac       = np.array([r['perturbation_frac']      for r in results]),
        amerini_iters           = np.array([r['amerini_iters']          for r in results]),
        sparse_n_kps            = np.array([r['sparse_n_kps']             for r in results]),
        sparse_area_frac        = np.array([r['sparse_area_frac']         for r in results]),
        sparse_n_refreshes      = np.array([r['sparse_n_refreshes']       for r in results]),
        sparse_area_frac_final  = np.array([r['sparse_area_frac_final']   for r in results]),
        grad_cos_sim_start      = np.array([r['grad_cos_sim_start']       for r in results]),
        grad_cos_sim_end        = np.array([r['grad_cos_sim_end']         for r in results]),
        grad_cos_sim_mean       = np.array([r['grad_cos_sim_mean']        for r in results]),
        grad_sign_agree_start   = np.array([r['grad_sign_agree_start']    for r in results]),
        grad_sign_agree_end     = np.array([r['grad_sign_agree_end']      for r in results]),
        grad_sign_agree_mean    = np.array([r['grad_sign_agree_mean']     for r in results]),
        grad_norm_ratio_start   = np.array([r['grad_norm_ratio_start']    for r in results]),
        grad_norm_ratio_end     = np.array([r['grad_norm_ratio_end']      for r in results]),
        grad_norm_ratio_mean    = np.array([r['grad_norm_ratio_mean']     for r in results]),
    )
    print(f"NPZ  -> {npz_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Sweep experiment_sift_attack.py over videos and seeds'
    )
    parser.add_argument('--dataset',  default='VOT2018')
    parser.add_argument('--videos',   nargs='+', default=None,
                        help='Video names to sweep (default: all in dataset)')
    parser.add_argument('--n_seeds',  type=int, default=5)
    parser.add_argument('--seed_base', type=int, default=0)
    parser.add_argument('--n_frames', type=int, default=10)
    parser.add_argument('--model',    default='SiamRPNvot.model')
    parser.add_argument('--out_dir',  default='out/sweep_sift_attack')
    parser.add_argument('--analyse_only', action='store_true')

    # Mirror experiment_sift_attack.py's attack-config args
    parser.add_argument('--attack', default='rtaa_sift_frame',
                        choices=['none', 'rtaa', 'rtaa_sift_crop',
                                 'rtaa_sift_frame', 'amerini_smoothing',
                                 'rtaa_amerini', 'rtaa_then_amerini',
                                 'rtaa_sift_then_amerini'])
    parser.add_argument('--eps',          type=float, default=16.0)
    parser.add_argument('--n_iter',       type=int,   default=10)
    parser.add_argument('--alpha_dog',    type=float, default=1000.0)
    parser.add_argument('--gamma_kornia', type=float, default=0.0)
    parser.add_argument('--dog_contrast', type=float, default=0.04)
    parser.add_argument('--rtaa_weight',  type=float, default=1.0)
    parser.add_argument('--roi_source',   default='gt',
                        choices=['gt', 'prev_pred'])
    parser.add_argument('--sparse_mask',  action='store_true')
    parser.add_argument('--sparse_half_side',     type=int, default=4)
    parser.add_argument('--sparse_refresh_every', type=int, default=0,
                        help='Re-detect kps every K PGD iters and union-extend '
                             'the sparse mask. Mirrors Amerini\'s outer loop. '
                             '0 = off (single-init mask).')
    parser.add_argument('--sparse_refresh_cap',   type=int, default=5)
    parser.add_argument('--diag_grad_alignment',  action='store_true',
                        help='Log L_rtaa vs L_dog gradient agreement per PGD '
                             'iter. ~2x slower (extra backward passes). Only '
                             'meaningful for --attack rtaa_sift_frame.')
    parser.add_argument('--rtaa_mask_gt', action='store_true',
                        help='Constrain L_rtaa-driven perturbation to the '
                             'GT bbox region (physical-patch approximation). '
                             'Honoured by rtaa_sift_frame, rtaa_amerini, '
                             'rtaa_then_amerini, rtaa_sift_then_amerini.')
    parser.add_argument('--save_viz', action='store_true',
                        help='Per-frame perturbation heatmap PNGs. Generates '
                             'NF * #(video, seed) PNGs -- use on small sweeps '
                             'only.')

    parser.add_argument('--amerini_sigma',          type=float, default=0.7)
    parser.add_argument('--amerini_ksize',          type=int,   default=3)
    parser.add_argument('--amerini_patch_half',     type=int,   default=4)
    parser.add_argument('--amerini_max_iter',       type=int,   default=40)
    parser.add_argument('--amerini_target_removal', type=float, default=1.0)

    args = parser.parse_args()

    data_dir = join(dirname(realpath(__file__)), 'data')
    os.makedirs(args.out_dir, exist_ok=True)

    videos = args.videos or list_videos(args.dataset, data_dir)
    seeds  = list(range(args.seed_base, args.seed_base + args.n_seeds))

    tag = _variant_tag(args)
    print(f"Videos: {len(videos)}  Seeds per video: {args.n_seeds}  "
          f"Total runs: {len(videos) * args.n_seeds}")
    print(f"Tag: {tag}")
    print(f"Out dir: {args.out_dir}")

    all_results = []
    for video in videos:
        video_results = []
        print(f"\n{'-' * 60}\nVideo: {video}")
        for seed in seeds:
            if args.analyse_only:
                log_path = join(args.out_dir,
                                f"log_{video}_s{seed}_{tag}.npz")
            else:
                log_path = run_experiment(video, seed, args)

            if log_path is None or not os.path.exists(log_path):
                print(f"  seed={seed}  MISSING")
                continue

            try:
                m = compute_metrics(log_path)
            except Exception as e:
                print(f"  seed={seed}  ERROR: {e}")
                continue
            m['video'] = video
            m['seed']  = seed
            video_results.append(m)
            all_results.append(m)
            print(f"  seed={seed:5d}  "
                  f"iou={m['attack_iou']:.2f} (drop {m['iou_drop']:+.2f})  "
                  f"removal={m['removal_rate']:+.3f}  "
                  f"kp {m['kp_clean']:.0f}->{m['kp_attacked']:.0f}")

        if video_results:
            _print_video_summary(video, video_results)

    if not all_results:
        print("\nNo results collected.")
        return

    _print_aggregate(all_results, tag)
    _save_results(all_results, args.out_dir, tag)


if __name__ == '__main__':
    main()
