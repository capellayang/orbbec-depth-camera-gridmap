"""Fuse multiple local occupancy grid maps with temporal voting.

Input grids must be 2D NumPy arrays with:
    0 = non-obstacle
    1 = occupied obstacle

This is a lightweight temporal filter for local obstacle maps. It does not use
odometry, TF, or SLAM pose alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "grid_data"


def resolve_path(path_text: str | None, default_path: Path | None = None) -> Path:
    """Resolve a path relative to the project root."""
    if path_text is None:
        if default_path is None:
            raise ValueError("path_text and default_path cannot both be None")
        return default_path

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def discover_inputs(input_dir: Path) -> list[Path]:
    """Find gridmap*.npy files in a directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    paths = sorted(
        path for path in input_dir.glob("gridmap*.npy")
        if path.name != "fused_gridmap.npy"
    )
    if not paths:
        raise FileNotFoundError(f"No gridmap*.npy files found in: {input_dir}")
    return paths


def resolve_inputs(args: argparse.Namespace) -> tuple[list[Path], Path]:
    """Resolve manual inputs or directory-discovered inputs."""
    input_dir = resolve_path(args.input_dir, DEFAULT_INPUT_DIR)
    if args.inputs:
        input_paths = [resolve_path(path_text) for path_text in args.inputs]
    else:
        input_paths = discover_inputs(input_dir)

    missing = [path for path in input_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Input gridmap file(s) not found:\n{missing_text}")

    selected = sorted(input_paths)[: args.window_size]
    if not selected:
        raise ValueError("No input gridmaps selected")
    return selected, input_dir


def load_gridmaps(paths: list[Path]) -> list[np.ndarray]:
    """Load and validate 2D binary grid maps."""
    grids: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None

    for path in paths:
        grid = np.load(path)
        if grid.ndim != 2:
            raise ValueError(f"{path} must be a 2D grid, got shape {grid.shape}")

        unique_values = set(np.unique(grid).tolist())
        if not unique_values.issubset({0, 1, False, True}):
            raise ValueError(
                f"{path} must contain only 0/1 values, got {sorted(unique_values)}"
            )

        if expected_shape is None:
            expected_shape = grid.shape
        elif grid.shape != expected_shape:
            raise ValueError(
                "All input gridmaps must have the same shape. "
                f"Expected {expected_shape}, but {path} has {grid.shape}."
            )

        grids.append(grid.astype(np.uint8))

    return grids


def fuse_gridmaps(
    grids: list[np.ndarray],
    vote_threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse binary occupancy grids with per-cell temporal voting."""
    if vote_threshold <= 0:
        raise ValueError("vote_threshold must be greater than 0")
    if vote_threshold > len(grids):
        raise ValueError(
            "vote_threshold cannot be larger than the number of selected frames "
            f"({vote_threshold} > {len(grids)})"
        )

    stack = np.stack(grids, axis=0)
    vote_counts = np.sum(stack, axis=0)
    fused = (vote_counts >= vote_threshold).astype(np.uint8)
    return fused, vote_counts


def save_fused_png(
    grid: np.ndarray,
    output_path: Path,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
    resolution: float,
    occupied_cells: int,
) -> None:
    """Save a top-down visualization consistent with pointcloud_to_gridmap.py."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is None:
        raise ImportError("OpenCV is required for PNG output: pip install opencv-python")

    height, width = grid.shape
    image = np.full((height, width, 3), 244, dtype=np.uint8)
    image[grid == 1] = (17, 17, 17)

    max_display = 1200
    scale = max(1, min(max_display // max(height, width), 12))
    display = cv2.resize(
        image,
        (width * scale, height * scale),
        interpolation=cv2.INTER_NEAREST,
    )

    display_height, display_width = display.shape[:2]
    camera_x = display_width // 2
    camera_y = display_height - 8
    red = (40, 40, 220)
    gray = (180, 180, 180)

    cv2.circle(display, (camera_x, camera_y), 6, red, -1)
    cv2.arrowedLine(
        display,
        (camera_x, camera_y - 4),
        (camera_x, max(camera_y - 70, 12)),
        red,
        2,
        tipLength=0.3,
    )
    cv2.putText(
        display,
        "Camera",
        (min(camera_x + 12, display_width - 90), max(camera_y - 10, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        red,
        1,
        cv2.LINE_AA,
    )

    labels = [
        "Fused Local Occupancy Grid (Temporal Voting)",
        f"X: {min_x:.1f} to {max_x:.1f} m",
        f"Z: {min_z:.1f} to {max_z:.1f} m",
        f"resolution: {resolution:.3f} m/cell",
        f"occupied cells: {occupied_cells}",
    ]
    for index, label in enumerate(labels):
        cv2.putText(
            display,
            label,
            (10, 22 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            gray,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        display,
        "Forward (+Z)",
        (max(camera_x - 45, 10), 22 + len(labels) * 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        red,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        "Left (-X)",
        (10, max(display_height - 36, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        gray,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        "Right (+X)",
        (max(display_width - 95, 10), max(display_height - 36, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        gray,
        1,
        cv2.LINE_AA,
    )

    success = cv2.imwrite(str(output_path), display)
    if not success:
        raise IOError(f"Failed to save PNG: {output_path}")


def write_summary_csv(
    output_path: Path,
    input_paths: list[Path],
    frame_occupied_cells: list[int],
    frame_occupied_ratios: list[float],
) -> None:
    """Save one row per input frame for quick comparison."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_index", "input_path", "occupied_cells", "occupied_ratio"],
        )
        writer.writeheader()
        for index, (path, cells, ratio) in enumerate(
            zip(input_paths, frame_occupied_cells, frame_occupied_ratios),
            start=1,
        ):
            writer.writerow(
                {
                    "frame_index": index,
                    "input_path": str(path),
                    "occupied_cells": cells,
                    "occupied_ratio": ratio,
                }
            )


def save_config(
    output_path: Path,
    args: argparse.Namespace,
    input_paths: list[Path],
    grid_shape: tuple[int, int],
    frame_occupied_cells: list[int],
    frame_occupied_ratios: list[float],
    fused_occupied_cells: int,
    fused_occupied_ratio: float,
    output_npy: Path,
    output_png: Path,
    summary_csv: Path,
) -> None:
    """Save fusion parameters and result statistics."""
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "temporal_voting",
        "input_dir": str(resolve_path(args.input_dir, DEFAULT_INPUT_DIR)),
        "input_paths": [str(path) for path in input_paths],
        "input_frame_count": len(input_paths),
        "used_frame_count": len(input_paths),
        "grid_height": grid_shape[0],
        "grid_width": grid_shape[1],
        "window_size": args.window_size,
        "vote_threshold": args.vote_threshold,
        "frame_occupied_cells": frame_occupied_cells,
        "frame_occupied_ratios": frame_occupied_ratios,
        "mean_input_occupied_ratio": float(np.mean(frame_occupied_ratios)),
        "fused_occupied_cells": fused_occupied_cells,
        "fused_occupied_ratio": fused_occupied_ratio,
        "output_npy": str(output_npy),
        "output_png": str(output_png),
        "summary_csv": str(summary_csv),
        "note": "Lightweight local temporal voting only. No odometry, TF, or SLAM pose alignment is used.",
    }
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse multiple 0/1 local occupancy grid maps with temporal voting."
    )
    parser.add_argument(
        "--input_dir",
        default="grid_data",
        help="Directory containing gridmap*.npy files. Default: grid_data",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="Manually specify multiple .npy gridmap files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to <input_dir>/fused/.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Number of sorted input frames to use. Default: 5",
    )
    parser.add_argument(
        "--vote_threshold",
        type=int,
        default=3,
        help="Cells with at least this many occupied votes become occupied. Default: 3",
    )
    parser.add_argument("--min-x", type=float, default=-2.5)
    parser.add_argument("--max-x", type=float, default=2.5)
    parser.add_argument("--min-z", type=float, default=0.3)
    parser.add_argument("--max-z", type=float, default=5.0)
    parser.add_argument("--resolution", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.window_size <= 0:
        raise ValueError("window_size must be greater than 0")

    input_paths, input_dir = resolve_inputs(args)
    output_dir = resolve_path(args.output_dir) if args.output_dir else input_dir / "fused"
    output_dir.mkdir(parents=True, exist_ok=True)

    grids = load_gridmaps(input_paths)
    fused_grid, _ = fuse_gridmaps(grids, vote_threshold=args.vote_threshold)

    frame_occupied_cells = [int(np.count_nonzero(grid)) for grid in grids]
    total_cells = int(grids[0].size)
    frame_occupied_ratios = [cells / total_cells for cells in frame_occupied_cells]
    fused_occupied_cells = int(np.count_nonzero(fused_grid))
    fused_occupied_ratio = fused_occupied_cells / total_cells

    output_npy = output_dir / "fused_gridmap.npy"
    output_png = output_dir / "fused_gridmap.png"
    config_json = output_dir / "fusion_config.json"
    summary_csv = output_dir / "fusion_summary.csv"

    np.save(output_npy, fused_grid.astype(np.uint8))
    save_fused_png(
        fused_grid,
        output_png,
        min_x=args.min_x,
        max_x=args.max_x,
        min_z=args.min_z,
        max_z=args.max_z,
        resolution=args.resolution,
        occupied_cells=fused_occupied_cells,
    )
    write_summary_csv(summary_csv, input_paths, frame_occupied_cells, frame_occupied_ratios)
    save_config(
        config_json,
        args,
        input_paths=input_paths,
        grid_shape=grids[0].shape,
        frame_occupied_cells=frame_occupied_cells,
        frame_occupied_ratios=frame_occupied_ratios,
        fused_occupied_cells=fused_occupied_cells,
        fused_occupied_ratio=fused_occupied_ratio,
        output_npy=output_npy,
        output_png=output_png,
        summary_csv=summary_csv,
    )

    print("")
    print("Grid map fusion summary")
    print(f"  Input frame count: {len(discover_inputs(input_dir)) if not args.inputs else len(args.inputs)}")
    print(f"  Used frame count: {len(input_paths)}")
    print(f"  Grid size: {grids[0].shape[1]} x {grids[0].shape[0]} cells")
    print(f"  Window size: {args.window_size}")
    print(f"  Vote threshold: {args.vote_threshold}")
    print("  Per-frame occupied cells:")
    for path, cells, ratio in zip(input_paths, frame_occupied_cells, frame_occupied_ratios):
        print(f"    {path.name}: {cells} cells ({ratio:.2%})")
    print(f"  Mean input occupied ratio: {float(np.mean(frame_occupied_ratios)):.2%}")
    print(f"  Fused occupied cells: {fused_occupied_cells}")
    print(f"  Fused occupied ratio: {fused_occupied_ratio:.2%}")
    print(f"  Output NPY: {output_npy}")
    print(f"  Output PNG: {output_png}")
    print(f"  Config JSON: {config_json}")
    print(f"  Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
