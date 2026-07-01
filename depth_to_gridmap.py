"""Convert a saved depth image directly into a 2D local occupancy grid map.

This script removes the intermediate .ply dependency:
    depth image -> 3D camera points -> X-Z local occupancy grid

Coordinate convention:
    X: left-right, positive to the right
    Y: vertical, positive upward
    Z: forward depth, positive forward
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from pointcloud_to_gridmap import (
    build_gridmap,
    filter_points,
    filter_small_components,
    remove_ground_points,
    save_grid_png,
    verify_gridmap,
)

try:
    import cv2
except ImportError:
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DEPTH = PROJECT_ROOT / "depth_data" / "depth_0001.npy"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "gridmap_from_depth"

DEFAULT_FX = 574.9614356132867
DEFAULT_FY = 574.9614061048316


@dataclass
class DepthGridmapConfig:
    """Parameters needed to turn one depth image into a local occupancy grid."""

    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    depth_unit: str = "mm"
    min_x: float = -2.5
    max_x: float = 2.5
    min_y: float = -0.5
    max_y: float = 1.5
    min_z: float = 0.3
    max_z: float = 5.0
    resolution: float = 0.05
    obstacle_threshold: int = 3
    remove_ground: bool = False
    ground_y_threshold: float = -0.45
    min_component_size: int = 1
    y_axis_up: bool = True


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


def load_depth_image(path: Path) -> np.ndarray:
    """Load .npy or 16-bit .png depth data."""
    if not path.exists():
        raise FileNotFoundError(f"Input depth image not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".png":
        if cv2 is None:
            raise ImportError("OpenCV is required to read 16-bit PNG depth images.")
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to read depth PNG: {path}")
    else:
        raise ValueError(f"Unsupported depth format: {path.suffix}. Use .npy or 16-bit .png.")

    if depth.ndim != 2:
        raise ValueError(f"Depth image must be 2D, got shape {depth.shape}")
    return depth


def depth_to_meters(depth: np.ndarray, depth_unit: str) -> np.ndarray:
    """Convert input depth units to meters."""
    if depth_unit == "mm":
        return depth.astype(np.float32) / 1000.0
    if depth_unit == "m":
        return depth.astype(np.float32)
    raise ValueError("depth_unit must be 'mm' or 'm'")


def resolve_intrinsics(
    args: argparse.Namespace,
    width: int,
    height: int,
) -> tuple[float, float, float, float, bool]:
    """Use provided intrinsics or a learning-stage default estimate."""
    fx = args.fx
    fy = args.fy
    cx = args.cx
    cy = args.cy
    used_estimate = False

    if fx is None or fy is None:
        fx = DEFAULT_FX if fx is None else fx
        fy = DEFAULT_FY if fy is None else fy
        used_estimate = True
    if cx is None or cy is None:
        cx = width / 2.0 if cx is None else cx
        cy = height / 2.0 if cy is None else cy
        used_estimate = True

    return float(fx), float(fy), float(cx), float(cy), used_estimate


def _config_value(config: DepthGridmapConfig | argparse.Namespace | dict, name: str):
    if isinstance(config, dict):
        return config.get(name, getattr(DepthGridmapConfig(), name))
    return getattr(config, name, getattr(DepthGridmapConfig(), name))


def normalize_gridmap_config(
    config: DepthGridmapConfig | argparse.Namespace | dict,
) -> DepthGridmapConfig:
    """Accept a dataclass, argparse namespace, or dict as reusable function config."""
    return DepthGridmapConfig(
        fx=_config_value(config, "fx"),
        fy=_config_value(config, "fy"),
        cx=_config_value(config, "cx"),
        cy=_config_value(config, "cy"),
        depth_unit=_config_value(config, "depth_unit"),
        min_x=float(_config_value(config, "min_x")),
        max_x=float(_config_value(config, "max_x")),
        min_y=float(_config_value(config, "min_y")),
        max_y=float(_config_value(config, "max_y")),
        min_z=float(_config_value(config, "min_z")),
        max_z=float(_config_value(config, "max_z")),
        resolution=float(_config_value(config, "resolution")),
        obstacle_threshold=int(_config_value(config, "obstacle_threshold")),
        remove_ground=bool(_config_value(config, "remove_ground")),
        ground_y_threshold=float(_config_value(config, "ground_y_threshold")),
        min_component_size=int(_config_value(config, "min_component_size")),
        y_axis_up=bool(_config_value(config, "y_axis_up")),
    )


def resolve_intrinsics_for_config(
    config: DepthGridmapConfig,
    width: int,
    height: int,
) -> tuple[float, float, float, float, bool]:
    """Use explicit intrinsics when available, otherwise keep the old CLI defaults."""
    fx = config.fx
    fy = config.fy
    cx = config.cx
    cy = config.cy
    used_estimate = False

    if fx is None or fy is None:
        fx = DEFAULT_FX if fx is None else fx
        fy = DEFAULT_FY if fy is None else fy
        used_estimate = True
    if cx is None or cy is None:
        cx = width / 2.0 if cx is None else cx
        cy = height / 2.0 if cy is None else cy
        used_estimate = True

    return float(fx), float(fy), float(cx), float(cy), used_estimate


def depth_to_points(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    y_axis_up: bool,
) -> tuple[np.ndarray, int]:
    """Back-project valid depth pixels into camera-frame 3D points."""
    height, width = depth_m.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    valid = (depth_m > 0) & np.isfinite(depth_m)
    z = depth_m
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    if y_axis_up:
        # Image v grows downward, while the project convention uses Y upward.
        y = -y

    points = np.stack((x, y, z), axis=-1)
    return points[valid], int(np.count_nonzero(valid))


def generate_gridmap_from_depth(
    depth: np.ndarray,
    config: DepthGridmapConfig | argparse.Namespace | dict,
    include_debug: bool = False,
) -> tuple[np.ndarray, dict, dict | None]:
    """Convert a depth frame into a binary local occupancy grid map.

    Args:
        depth: 2D depth image as a numpy array.
        config: DepthGridmapConfig, argparse.Namespace, or dict with camera,
            filtering, grid, and denoising parameters.
        include_debug: Return intermediate arrays for local inspection.

    Returns:
        gridmap, metrics, debug. Debug is None unless include_debug is enabled.
    """
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be 2D, got shape {depth.shape}")

    cfg = normalize_gridmap_config(config)
    depth_m = depth_to_meters(depth, cfg.depth_unit)
    height, width = depth_m.shape
    fx, fy, cx, cy, used_estimated_intrinsics = resolve_intrinsics_for_config(
        cfg,
        width=width,
        height=height,
    )

    points, valid_depth_pixels = depth_to_points(
        depth_m,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        y_axis_up=cfg.y_axis_up,
    )
    raw_point_count = len(points)

    filtered_points, filter_stats = filter_points(
        points,
        min_x=cfg.min_x,
        max_x=cfg.max_x,
        min_y=cfg.min_y,
        max_y=cfg.max_y,
        min_z=cfg.min_z,
        max_z=cfg.max_z,
    )
    filtered_points, ground_stats = remove_ground_points(
        filtered_points,
        remove_ground=cfg.remove_ground,
        ground_y_threshold=cfg.ground_y_threshold,
    )

    gridmap, cell_counts, _ = build_gridmap(
        filtered_points,
        min_x=cfg.min_x,
        max_x=cfg.max_x,
        min_z=cfg.min_z,
        max_z=cfg.max_z,
        resolution=cfg.resolution,
        obstacle_threshold=cfg.obstacle_threshold,
    )
    gridmap, removed_components, removed_component_cells = filter_small_components(
        gridmap,
        min_component_size=cfg.min_component_size,
    )

    occupied_cells = int(np.count_nonzero(gridmap))
    occupied_ratio = occupied_cells / int(gridmap.size) if gridmap.size else 0.0
    verification_errors = verify_gridmap(gridmap)

    metrics = {
        "depth_shape": tuple(depth_m.shape),
        "depth_unit": cfg.depth_unit,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "used_estimated_intrinsics": used_estimated_intrinsics,
        "y_axis_up": cfg.y_axis_up,
        "min_x": cfg.min_x,
        "max_x": cfg.max_x,
        "min_y": cfg.min_y,
        "max_y": cfg.max_y,
        "min_z": cfg.min_z,
        "max_z": cfg.max_z,
        "resolution": cfg.resolution,
        "obstacle_threshold": cfg.obstacle_threshold,
        "remove_ground": cfg.remove_ground,
        "ground_y_threshold": cfg.ground_y_threshold,
        "min_component_size": cfg.min_component_size,
        "valid_depth_pixels": valid_depth_pixels,
        "raw_point_count": raw_point_count,
        "valid_point_count": len(filtered_points),
        **filter_stats,
        **ground_stats,
        "grid_height": gridmap.shape[0],
        "grid_width": gridmap.shape[1],
        "occupied_cells": occupied_cells,
        "occupied_ratio": occupied_ratio,
        "removed_small_components": removed_components,
        "removed_small_component_cells": removed_component_cells,
        "verification_passed": not verification_errors,
        "verification_errors": verification_errors,
    }

    debug = None
    if include_debug:
        debug = {
            "depth_m": depth_m,
            "points": points,
            "filtered_points": filtered_points,
            "cell_counts": cell_counts,
        }

    return gridmap.astype(np.uint8), metrics, debug


def save_config(
    path: Path,
    args: argparse.Namespace,
    input_depth: Path,
    output_npy: Path,
    output_png: Path,
    depth_shape: tuple[int, int],
    valid_depth_pixels: int,
    raw_point_count: int,
    valid_point_count: int,
    filter_stats: dict[str, int],
    ground_stats: dict[str, int],
    grid_shape: tuple[int, int],
    occupied_cells: int,
    occupied_ratio: float,
    removed_components: int,
    removed_component_cells: int,
    intrinsics: tuple[float, float, float, float],
    used_estimated_intrinsics: bool,
    verification_errors: list[str],
) -> None:
    """Save depth-to-grid parameters and result statistics."""
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_depth": str(input_depth),
        "output_npy": str(output_npy),
        "output_png": str(output_png),
        "depth_shape": list(depth_shape),
        "depth_unit": args.depth_unit,
        "fx": intrinsics[0],
        "fy": intrinsics[1],
        "cx": intrinsics[2],
        "cy": intrinsics[3],
        "used_estimated_intrinsics": used_estimated_intrinsics,
        "y_axis_up": args.y_axis_up,
        "min_x": args.min_x,
        "max_x": args.max_x,
        "min_y": args.min_y,
        "max_y": args.max_y,
        "min_z": args.min_z,
        "max_z": args.max_z,
        "resolution": args.resolution,
        "obstacle_threshold": args.obstacle_threshold,
        "remove_ground": args.remove_ground,
        "ground_y_threshold": args.ground_y_threshold,
        "min_component_size": args.min_component_size,
        "valid_depth_pixels": valid_depth_pixels,
        "raw_point_count": raw_point_count,
        "valid_point_count": valid_point_count,
        **filter_stats,
        **ground_stats,
        "grid_height": grid_shape[0],
        "grid_width": grid_shape[1],
        "occupied_cells": occupied_cells,
        "occupied_ratio": occupied_ratio,
        "removed_small_components": removed_components,
        "removed_small_component_cells": removed_component_cells,
        "verification_passed": not verification_errors,
        "verification_errors": verification_errors,
        "future_note": "Reserved for free/unknown/ray-casting occupancy modeling.",
    }
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(
    input_depth: Path,
    output_npy: Path,
    output_png: Path,
    config_path: Path,
    depth_shape: tuple[int, int],
    depth_unit: str,
    valid_depth_pixels: int,
    raw_point_count: int,
    valid_point_count: int,
    filter_stats: dict[str, int],
    ground_stats: dict[str, int],
    grid_shape: tuple[int, int],
    resolution: float,
    occupied_cells: int,
    occupied_ratio: float,
    intrinsics: tuple[float, float, float, float],
    used_estimated_intrinsics: bool,
    verification_errors: list[str],
) -> None:
    """Print a compact run summary."""
    print("")
    print("Depth-to-gridmap summary")
    print(f"  Input depth: {input_depth}")
    print(f"  Depth image size: {depth_shape[1]} x {depth_shape[0]}")
    print(f"  Depth unit: {depth_unit}")
    print(
        "  Camera intrinsics: "
        f"fx={intrinsics[0]:.3f}, fy={intrinsics[1]:.3f}, "
        f"cx={intrinsics[2]:.3f}, cy={intrinsics[3]:.3f}"
    )
    if used_estimated_intrinsics:
        print("  Intrinsics note: using estimated values. ROS2 should read /camera/depth/camera_info.")
    print(f"  Valid depth pixels: {valid_depth_pixels}")
    print(f"  Converted 3D point count: {raw_point_count}")
    print(f"  Valid point count after filtering: {valid_point_count}")
    print(f"  Invalid NaN/Inf point count: {filter_stats['invalid_point_count']}")
    print(f"  X/Z range filtered point count: {filter_stats['xz_filtered_point_count']}")
    print(f"  Height filtered point count: {filter_stats['height_filtered_point_count']}")
    if ground_stats["ground_removal_enabled"]:
        print(
            "  Ground removal: "
            f"{ground_stats['ground_before_point_count']} -> "
            f"{ground_stats['ground_after_point_count']} points "
            f"({ground_stats['ground_removed_point_count']} removed)"
        )
    else:
        print("  Ground removal: disabled")
    print(f"  Map size: {grid_shape[1]} x {grid_shape[0]} cells")
    print(f"  Resolution: {resolution:.3f} m/cell")
    print(f"  Occupied cells: {occupied_cells}")
    print(f"  Occupied ratio: {occupied_ratio:.2%}")
    print(f"  Verification: {'PASS' if not verification_errors else 'FAIL'}")
    for error in verification_errors:
        print(f"    - {error}")
    print(f"  Output NPY: {output_npy}")
    print(f"  Output PNG: {output_png}")
    print(f"  Config JSON: {config_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a saved depth image directly into a 2D local occupancy grid map."
    )
    parser.add_argument(
        "--input_depth",
        default=str(DEFAULT_INPUT_DEPTH),
        help="Input .npy or 16-bit .png depth image.",
    )
    parser.add_argument(
        "--depth_unit",
        choices=["mm", "m"],
        default="mm",
        help="Input depth unit. Default: mm",
    )
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument(
        "--y_axis_up",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert image v-down coordinates into project Y-up coordinates. Default: enabled.",
    )
    parser.add_argument("--min-z", "--min_z", dest="min_z", type=float, default=0.3)
    parser.add_argument("--max-z", "--max_z", dest="max_z", type=float, default=5.0)
    parser.add_argument("--min-x", "--min_x", dest="min_x", type=float, default=-2.5)
    parser.add_argument("--max-x", "--max_x", dest="max_x", type=float, default=2.5)
    parser.add_argument("--min-y", "--min_y", dest="min_y", type=float, default=-0.5)
    parser.add_argument("--max-y", "--max_y", dest="max_y", type=float, default=1.5)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument(
        "--obstacle-threshold",
        "--obstacle_threshold",
        dest="obstacle_threshold",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--remove_ground",
        "--remove-ground",
        action="store_true",
        help="Remove low points before grid projection with a simple Y threshold.",
    )
    parser.add_argument(
        "--ground-y-threshold",
        "--ground_y_threshold",
        type=float,
        default=-0.45,
    )
    parser.add_argument(
        "--min-component-size",
        "--min_component_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory. Default: data/gridmap_from_depth/",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    input_depth = resolve_path(args.input_depth)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_raw = load_depth_image(input_depth)
    grid, metrics, _ = generate_gridmap_from_depth(depth_raw, args)

    filter_stats = {
        "invalid_point_count": metrics["invalid_point_count"],
        "xz_filtered_point_count": metrics["xz_filtered_point_count"],
        "height_filtered_point_count": metrics["height_filtered_point_count"],
        "filtered_out_point_count": metrics["filtered_out_point_count"],
    }
    ground_stats = {
        "ground_removal_enabled": metrics["ground_removal_enabled"],
        "ground_removed_point_count": metrics["ground_removed_point_count"],
        "ground_before_point_count": metrics["ground_before_point_count"],
        "ground_after_point_count": metrics["ground_after_point_count"],
    }
    intrinsics = (metrics["fx"], metrics["fy"], metrics["cx"], metrics["cy"])

    output_npy = output_dir / "gridmap.npy"
    output_png = output_dir / "gridmap.png"
    config_path = output_dir / "config.json"

    np.save(output_npy, grid.astype(np.uint8))
    save_grid_png(
        grid,
        output_png,
        min_x=args.min_x,
        max_x=args.max_x,
        min_z=args.min_z,
        max_z=args.max_z,
        resolution=args.resolution,
        obstacle_cells=metrics["occupied_cells"],
    )
    save_config(
        config_path,
        args,
        input_depth=input_depth,
        output_npy=output_npy,
        output_png=output_png,
        depth_shape=metrics["depth_shape"],
        valid_depth_pixels=metrics["valid_depth_pixels"],
        raw_point_count=metrics["raw_point_count"],
        valid_point_count=metrics["valid_point_count"],
        filter_stats=filter_stats,
        ground_stats=ground_stats,
        grid_shape=grid.shape,
        occupied_cells=metrics["occupied_cells"],
        occupied_ratio=metrics["occupied_ratio"],
        removed_components=metrics["removed_small_components"],
        removed_component_cells=metrics["removed_small_component_cells"],
        intrinsics=intrinsics,
        used_estimated_intrinsics=metrics["used_estimated_intrinsics"],
        verification_errors=metrics["verification_errors"],
    )
    print_summary(
        input_depth=input_depth,
        output_npy=output_npy,
        output_png=output_png,
        config_path=config_path,
        depth_shape=metrics["depth_shape"],
        depth_unit=args.depth_unit,
        valid_depth_pixels=metrics["valid_depth_pixels"],
        raw_point_count=metrics["raw_point_count"],
        valid_point_count=metrics["valid_point_count"],
        filter_stats=filter_stats,
        ground_stats=ground_stats,
        grid_shape=grid.shape,
        resolution=args.resolution,
        occupied_cells=metrics["occupied_cells"],
        occupied_ratio=metrics["occupied_ratio"],
        intrinsics=intrinsics,
        used_estimated_intrinsics=metrics["used_estimated_intrinsics"],
        verification_errors=metrics["verification_errors"],
    )


if __name__ == "__main__":
    main()
