"""
Build VOT2018 JSON metadata and data/ symlinks for AttrackZone.

Usage:
    # All 60 videos:
    python build_vot_json.py --vot_root /home/share/VOT2018

    # Single video (for testing):
    python build_vot_json.py --vot_root /home/share/VOT2018 --video car1
"""

import argparse
import json
import os
import natsort
import numpy as np
from pathlib import Path


def polygon_to_rect(coords):
    """
    Convert 8-value VOT polygon [x1,y1,x2,y2,x3,y3,x4,y4] to
    axis-aligned [x, y, w, h] with top-left origin.

    Uses area-preserving scale (same logic as utils.get_axis_aligned_bbox)
    so the bbox area matches the rotated polygon area.
    """
    x_coords = coords[0::2]
    y_coords = coords[1::2]

    x1, x2 = min(x_coords), max(x_coords)
    y1, y2 = min(y_coords), max(y_coords)

    # Area of the polygon (approx as parallelogram from first two edges)
    A1 = (np.linalg.norm(coords[0:2] - coords[2:4]) *
          np.linalg.norm(coords[2:4] - coords[4:6]))
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2) if A2 > 0 else 1.0

    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1
    cx = np.mean(x_coords)
    cy = np.mean(y_coords)

    x = cx - w / 2
    y = cy - h / 2
    return [round(float(x), 2), round(float(y), 2),
            round(float(w), 2), round(float(h), 2)]


def process_video(video_name, vot_root, data_dir):
    """
    Process one VOT video:
      - Reads groundtruth.txt and color/ images
      - Creates symlink data/VOT2018/{video}/img -> {vot_root}/{video}/color
      - Returns the JSON entry dict for this video
    """
    video_dir = Path(vot_root) / video_name
    color_dir = video_dir / "color"
    gt_path   = video_dir / "groundtruth.txt"

    if not color_dir.exists():
        raise FileNotFoundError(f"No color/ dir found at {color_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(f"No groundtruth.txt found at {gt_path}")

    # --- image files ---
    image_files = natsort.natsorted(
        [f for f in os.listdir(color_dir) if f.endswith(".jpg")]
    )
    if not image_files:
        raise ValueError(f"No .jpg images found in {color_dir}")

    # --- ground truth ---
    gt = []
    with open(gt_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.replace(",", " ").split()]
            if len(vals) == 8:
                coords = np.array(vals)
                rect = polygon_to_rect(coords)
            elif len(vals) == 4:
                # Already [x, y, w, h]
                rect = [round(v, 2) for v in vals]
            else:
                raise ValueError(
                    f"{gt_path} line {line_no}: expected 4 or 8 values, got {len(vals)}"
                )
            gt.append(rect)

    if len(image_files) != len(gt):
        print(f"  [WARN] {video_name}: {len(image_files)} images vs "
              f"{len(gt)} GT lines — using min({len(image_files)}, {len(gt)})")
        n = min(len(image_files), len(gt))
        image_files = image_files[:n]
        gt = gt[:n]

    # --- symlink: data/VOT2018/{video}/img -> {vot_root}/{video}/color ---
    link_dir  = data_dir / video_name / "img"
    link_dir.parent.mkdir(parents=True, exist_ok=True)
    if link_dir.exists() or link_dir.is_symlink():
        link_dir.unlink()
    link_dir.symlink_to(color_dir.resolve())
    print(f"  symlink: {link_dir} -> {color_dir.resolve()}")

    return {
        "name": video_name,
        "image_files": image_files,
        "gt": gt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vot_root", required=True,
                        help="Path to VOT2018 root (e.g. /home/share/VOT2018)")
    parser.add_argument("--video", default=None,
                        help="Single video name to process (omit for all)")
    parser.add_argument("--dataset_name", default="VOT2018",
                        help="Name used for JSON file and data/ subdir (default: VOT2018)")
    args = parser.parse_args()

    vot_root = Path(args.vot_root)
    if not vot_root.exists():
        raise FileNotFoundError(f"VOT root not found: {vot_root}")

    # data/ lives next to this script
    script_dir = Path(__file__).resolve().parent
    data_dir   = script_dir / "data" / args.dataset_name
    json_path  = script_dir / "data" / f"{args.dataset_name}.json"

    # Determine which videos to process
    if args.video:
        video_names = [args.video]
    else:
        video_names = natsort.natsorted([
            d for d in os.listdir(vot_root)
            if (vot_root / d).is_dir()
            and (vot_root / d / "color").exists()
            and (vot_root / d / "groundtruth.txt").exists()
        ])

    print(f"Processing {len(video_names)} video(s) into {json_path}\n")

    # Load existing JSON if present (so --video can update incrementally)
    if json_path.exists():
        with open(json_path) as f:
            info = json.load(f)
    else:
        info = {}

    for video_name in video_names:
        print(f"[{video_name}]")
        try:
            entry = process_video(video_name, vot_root, data_dir)
            info[video_name] = entry
            print(f"  {len(entry['image_files'])} frames, {len(entry['gt'])} GT boxes")
        except Exception as e:
            print(f"  [ERROR] {e}")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone. JSON saved to {json_path}")
    print(f"Total videos in JSON: {len(info)}")


if __name__ == "__main__":
    main()
