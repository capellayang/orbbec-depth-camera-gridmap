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
    enable_raycast: bool = False
    ray_step: float | None = None
    raycast_stride: int = 1


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
        enable_raycast=bool(_config_value(config, "enable_raycast")),
        ray_step=_config_value(config, "ray_step"),
        raycast_stride=int(_config_value(config, "raycast_stride")),
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


def _grid_shape(min_x: float, max_x: float, min_z: float, max_z: float, resolution: float) -> tuple[int, int]:
    """Return grid height and width for the configured X-Z map range."""
    if resolution <= 0:
        raise ValueError("resolution must be greater than 0")
    if max_x <= min_x or max_z <= min_z:
        raise ValueError("max_x/max_z must be greater than min_x/min_z")
    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_z - min_z) / resolution))
    return height, width


def _world_to_grid_cell(
    x: float,
    z: float,
    min_x: float,
    min_z: float,
    resolution: float,
    height: int,
    width: int,
) -> tuple[int, int]:
    """Map X-Z coordinates to row/column indices used by the saved top-down grid."""
    col = int(np.floor((x - min_x) / resolution))
    z_bin = int(np.floor((z - min_z) / resolution))
    col = int(np.clip(col, 0, width - 1))
    z_bin = int(np.clip(z_bin, 0, height - 1))
    row = height - 1 - z_bin
    return row, col


def _clip_segment_to_grid(
    start_x: float,
    start_z: float,
    end_x: float,
    end_z: float,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
) -> tuple[float, float, float, float] | None:
    """Clip a camera-to-point segment to the map rectangle with Liang-Barsky clipping."""
    dx = end_x - start_x
    dz = end_z - start_z
    t_min = 0.0
    t_max = 1.0

    for p, q in (
        (-dx, start_x - min_x),
        (dx, max_x - start_x),
        (-dz, start_z - min_z),
        (dz, max_z - start_z),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t_min = max(t_min, t)
        else:
            t_max = min(t_max, t)
        if t_min > t_max:
            return None

    clipped_start_x = start_x + t_min * dx
    clipped_start_z = start_z + t_min * dz
    clipped_end_x = start_x + t_max * dx
    clipped_end_z = start_z + t_max * dz
    return clipped_start_x, clipped_start_z, clipped_end_x, clipped_end_z


def _bresenham_cells(
    start_row: int,
    start_col: int,
    end_row: int,
    end_col: int,
) -> list[tuple[int, int]]:
    """Return grid cells along a line, including both endpoints."""
    cells: list[tuple[int, int]] = []
    row = start_row
    col = start_col
    d_col = abs(end_col - start_col)
    d_row = -abs(end_row - start_row)
    step_col = 1 if start_col < end_col else -1
    step_row = 1 if start_row < end_row else -1
    error = d_col + d_row

    while True:
        cells.append((row, col))
        if row == end_row and col == end_col:
            break
        doubled_error = 2 * error
        if doubled_error >= d_row:
            error += d_row
            col += step_col
        if doubled_error <= d_col:
            error += d_col
            row += step_row

    return cells


def build_occupancy_grid_with_ray_casting(
    points: np.ndarray,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
    resolution: float,
    obstacle_threshold: int,
    ray_step: float | None = None,
    mark_free: bool = True,
    raycast_stride: int = 1,
) -> tuple[np.ndarray, dict]:
    """Build a ROS-style -1/0/100 OccupancyGrid with free-space ray casting."""
    if obstacle_threshold <= 0:
        raise ValueError("obstacle_threshold must be greater than 0")
    if ray_step is not None and ray_step <= 0:
        raise ValueError("ray_step must be greater than 0 when provided")
    if raycast_stride <= 0:
        raise ValueError("raycast_stride must be >= 1")

    height, width = _grid_shape(min_x, max_x, min_z, max_z, resolution)
    occupancy_grid = np.full((height, width), -1, dtype=np.int16)
    cell_counts = np.zeros((height, width), dtype=np.int32)

    finite_points = points[np.isfinite(points).all(axis=1)] if len(points) else points
    roi_mask = (
        (finite_points[:, 0] >= min_x)
        & (finite_points[:, 0] <= max_x)
        & (finite_points[:, 2] >= min_z)
        & (finite_points[:, 2] <= max_z)
    ) if len(finite_points) else np.zeros((0,), dtype=bool)
    roi_points = finite_points[roi_mask]

    if len(roi_points) > 0:
        cols = np.floor((roi_points[:, 0] - min_x) / resolution).astype(np.int32)
        cols = np.clip(cols, 0, width - 1)
        z_bins = np.floor((roi_points[:, 2] - min_z) / resolution).astype(np.int32)
        z_bins = np.clip(z_bins, 0, height - 1)
        rows = height - 1 - z_bins
        np.add.at(cell_counts, (rows, cols), 1)

        ray_points = roi_points[::raycast_stride]
        processed_points = int(len(ray_points))
        skipped_points = int(len(roi_points) - processed_points)
        processed_cells = 0

        if mark_free and processed_points > 0:
            target_cells = np.column_stack((rows[::raycast_stride], cols[::raycast_stride]))
            unique_target_cells = np.unique(target_cells, axis=0)
            processed_cells = int(len(unique_target_cells))
            for target_row, target_col in unique_target_cells:
                target_z_bin = height - 1 - int(target_row)
                target_x = min_x + (int(target_col) + 0.5) * resolution
                target_z = min_z + (target_z_bin + 0.5) * resolution
                clipped = _clip_segment_to_grid(
                    0.0,
                    0.0,
                    target_x,
                    target_z,
                    min_x,
                    max_x,
                    min_z,
                    max_z,
                )
                if clipped is None:
                    continue
                start_x, start_z, _, _ = clipped
                start_row, start_col = _world_to_grid_cell(
                    start_x,
                    start_z,
                    min_x,
                    min_z,
                    resolution,
                    height,
                    width,
                )
                ray_cells = _bresenham_cells(
                    start_row,
                    start_col,
                    int(target_row),
                    int(target_col),
                )
                for row, col in ray_cells[:-1]:
                    if occupancy_grid[row, col] != 100:
                        occupancy_grid[row, col] = 0
        else:
            processed_points = int(len(roi_points[::raycast_stride]))
            skipped_points = int(len(roi_points) - processed_points)
            processed_cells = 0
    else:
        processed_points = 0
        skipped_points = 0
        processed_cells = 0

    occupied_mask = cell_counts >= obstacle_threshold
    occupancy_grid[occupied_mask] = 100

    unknown_cells = int(np.count_nonzero(occupancy_grid == -1))
    free_cells = int(np.count_nonzero(occupancy_grid == 0))
    occupied_cells = int(np.count_nonzero(occupancy_grid == 100))
    total_cells = int(occupancy_grid.size)
    metrics = {
        "raycast_enabled": True,
        "ray_step": resolution if ray_step is None else ray_step,
        "raycast_stride": raycast_stride,
        "raycast_processed_points": processed_points,
        "raycast_skipped_points": skipped_points,
        "raycast_candidate_cells": int(np.count_nonzero(cell_counts)),
        "raycast_processed_cells": processed_cells,
        "occupancy_unknown_cells": unknown_cells,
        "occupancy_free_cells": free_cells,
        "occupancy_occupied_cells": occupied_cells,
        "occupancy_unknown_ratio": unknown_cells / total_cells if total_cells else 0.0,
        "occupancy_free_ratio": free_cells / total_cells if total_cells else 0.0,
        "occupancy_occupied_ratio": occupied_cells / total_cells if total_cells else 0.0,
    }
    return occupancy_grid, metrics


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

    occupancy_grid = None
    occupancy_metrics = {
        "raycast_enabled": False,
        "ray_step": cfg.ray_step,
        "raycast_stride": cfg.raycast_stride,
    }
    if cfg.enable_raycast:
        occupancy_grid, occupancy_metrics = build_occupancy_grid_with_ray_casting(
            filtered_points,
            min_x=cfg.min_x,
            max_x=cfg.max_x,
            min_z=cfg.min_z,
            max_z=cfg.max_z,
            resolution=cfg.resolution,
            obstacle_threshold=cfg.obstacle_threshold,
            ray_step=cfg.ray_step,
            mark_free=True,
            raycast_stride=cfg.raycast_stride,
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
        **occupancy_metrics,
    }

    debug = None
    if include_debug:
        debug = {
            "depth_m": depth_m,
            "points": points,
            "filtered_points": filtered_points,
            "cell_counts": cell_counts,
        }
        if occupancy_grid is not None:
            debug["occupancy_grid"] = occupancy_grid

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
    raycast_metrics: dict | None = None,
    occupancy_npy: Path | None = None,
    occupancy_png: Path | None = None,
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
        "occupancy_grid_npy": str(occupancy_npy) if occupancy_npy else None,
        "occupancy_grid_png": str(occupancy_png) if occupancy_png else None,
    }
    if raycast_metrics:
        config.update(raycast_metrics)
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


def save_occupancy_grid_png(
    occupancy_grid: np.ndarray,
    output_path: Path,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
    resolution: float,
    metrics: dict,
) -> None:
    """Save a ROS-style -1/0/100 OccupancyGrid visualization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except ImportError:
        plt = None
        ListedColormap = None
        BoundaryNorm = None

    if plt is None:
        save_occupancy_grid_png_with_cv2(
            occupancy_grid,
            output_path,
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z,
            resolution=resolution,
            metrics=metrics,
        )
        return

    cmap = ListedColormap(["#9b9b9b", "#f8f8f8", "#111111"])
    norm = BoundaryNorm([-1.5, -0.5, 50.0, 100.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.imshow(
        occupancy_grid,
        cmap=cmap,
        norm=norm,
        origin="upper",
        extent=[min_x, max_x, min_z, max_z],
        interpolation="nearest",
        aspect="equal",
    )

    ax.plot(0.0, min_z, marker="^", markersize=10, color="#d62728", clip_on=False)
    ax.annotate(
        "Camera",
        xy=(0.0, min_z),
        xytext=(0.08, min_z + 0.12),
        color="#d62728",
        fontsize=9,
    )
    ax.arrow(
        0.0,
        min_z + 0.05,
        0.0,
        min(0.6, max_z - min_z) * 0.25,
        color="#d62728",
        width=max((max_x - min_x) * 0.004, 0.01),
        head_width=max((max_x - min_x) * 0.04, 0.08),
        length_includes_head=True,
    )

    ax.set_title("Ray-casting OccupancyGrid (-1 unknown, 0 free, 100 occupied)")
    ax.set_xlabel("X right positive (m)")
    ax.set_ylabel("Z forward positive (m)")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_z, max_z)
    ax.grid(color="#cfcfcf", linewidth=0.4, alpha=0.55)
    ax.text(
        0.01,
        0.99,
        (
            f"resolution: {resolution:.3f} m/cell\n"
            f"unknown: {metrics['occupancy_unknown_cells']}\n"
            f"free: {metrics['occupancy_free_cells']}\n"
            f"occupied: {metrics['occupancy_occupied_cells']}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82},
    )
    ax.text(min_x, min_z, "Left (-X)", va="bottom", ha="left", fontsize=8, color="#555555")
    ax.text(max_x, min_z, "Right (+X)", va="bottom", ha="right", fontsize=8, color="#555555")
    ax.text(0.0, max_z, "Forward (+Z)", va="top", ha="center", fontsize=8, color="#d62728")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_occupancy_grid_png_with_cv2(
    occupancy_grid: np.ndarray,
    output_path: Path,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
    resolution: float,
    metrics: dict,
) -> None:
    """OpenCV fallback for saving a three-value OccupancyGrid PNG."""
    if cv2 is None:
        raise ImportError(
            "PNG output needs either matplotlib or opencv-python. "
            "Install one of them with pip."
        )

    height, width = occupancy_grid.shape
    image = np.full((height, width, 3), 155, dtype=np.uint8)
    image[occupancy_grid == 0] = (248, 248, 248)
    image[occupancy_grid == 100] = (17, 17, 17)

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
    gray = (90, 90, 90)

    cv2.circle(display, (camera_x, camera_y), 6, red, -1)
    cv2.arrowedLine(
        display,
        (camera_x, camera_y - 4),
        (camera_x, max(camera_y - 70, 12)),
        red,
        2,
        tipLength=0.3,
    )

    labels = [
        "Ray-casting OccupancyGrid",
        "-1 unknown | 0 free | 100 occupied",
        f"X: {min_x:.1f} to {max_x:.1f} m, Z: {min_z:.1f} to {max_z:.1f} m",
        f"resolution: {resolution:.3f} m/cell",
        (
            f"unknown/free/occupied: "
            f"{metrics['occupancy_unknown_cells']}/"
            f"{metrics['occupancy_free_cells']}/"
            f"{metrics['occupancy_occupied_cells']}"
        ),
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
        "--enable-raycast",
        "--enable_raycast",
        dest="enable_raycast",
        action="store_true",
        help="Build a ROS-style -1/0/100 OccupancyGrid with free-space ray casting.",
    )
    parser.add_argument(
        "--ray-step",
        "--ray_step",
        dest="ray_step",
        type=float,
        default=None,
        help="Ray sampling step in meters. Grid-level ray casting currently defaults to resolution.",
    )
    parser.add_argument(
        "--raycast-stride",
        "--raycast_stride",
        dest="raycast_stride",
        type=int,
        default=1,
        help="Use every Nth filtered point for free-space ray casting. Default: 1",
    )
    parser.add_argument(
        "--save-occupancy-grid",
        "--save_occupancy_grid",
        dest="save_occupancy_grid",
        action="store_true",
        help="Save occupancy_grid.npy and occupancy_grid.png. Implies --enable-raycast.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory. Default: data/gridmap_from_depth/",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.raycast_stride < 1:
        raise ValueError("--raycast-stride must be >= 1")
    if args.save_occupancy_grid:
        args.enable_raycast = True

    input_depth = resolve_path(args.input_depth)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_raw = load_depth_image(input_depth)
    grid, metrics, debug = generate_gridmap_from_depth(
        depth_raw,
        args,
        include_debug=args.enable_raycast,
    )

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
    occupancy_npy = output_dir / "occupancy_grid.npy"
    occupancy_png = output_dir / "occupancy_grid.png"
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
    occupancy_grid = debug.get("occupancy_grid") if debug else None
    if args.save_occupancy_grid and occupancy_grid is not None:
        np.save(occupancy_npy, occupancy_grid.astype(np.int16))
        save_occupancy_grid_png(
            occupancy_grid,
            occupancy_png,
            min_x=args.min_x,
            max_x=args.max_x,
            min_z=args.min_z,
            max_z=args.max_z,
            resolution=args.resolution,
            metrics=metrics,
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
        raycast_metrics={
            key: value
            for key, value in metrics.items()
            if key.startswith("raycast_")
            or key.startswith("occupancy_")
            or key == "ray_step"
        },
        occupancy_npy=occupancy_npy if args.save_occupancy_grid else None,
        occupancy_png=occupancy_png if args.save_occupancy_grid else None,
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
    if args.enable_raycast:
        print("  Ray casting: enabled")
        print(
            "  OccupancyGrid cells: "
            f"unknown={metrics['occupancy_unknown_cells']}, "
            f"free={metrics['occupancy_free_cells']}, "
            f"occupied={metrics['occupancy_occupied_cells']}"
        )
        print(
            "  Ray casting workload: "
            f"processed_points={metrics['raycast_processed_points']}, "
            f"skipped_points={metrics['raycast_skipped_points']}, "
            f"candidate_cells={metrics['raycast_candidate_cells']}, "
            f"processed_cells={metrics['raycast_processed_cells']}"
        )
        if args.save_occupancy_grid:
            print(f"  OccupancyGrid NPY: {occupancy_npy}")
            print(f"  OccupancyGrid PNG: {occupancy_png}")


if __name__ == "__main__":
    main()
