"""Convert a PLY point cloud into a 2D local occupancy grid map.

Coordinate convention:
    X: left-right, positive to the right
    Y: vertical, positive upward
    Z: forward depth, positive forward

The grid is generated from the X-Z top-down plane. In the saved PNG,
positive Z points upward and the camera is drawn near the bottom center.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError(
        "Open3D is required. Install it with: pip install open3d"
    ) from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    plt = None
    ListedColormap = None

try:
    import cv2
except ImportError:
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT_CANDIDATES = (
    PROJECT_ROOT / "data" / "pointcloud" / "output.ply",
    PROJECT_ROOT / "astra_pointcloud_raw.ply",
    PROJECT_ROOT / "astra_pointcloud_processed.ply",
    PROJECT_ROOT / "astra_pointcloud.ply",
    PROJECT_ROOT / "filtered_for_grid.ply",
)

DEFAULT_GRIDMAP_DIR = PROJECT_ROOT / "data" / "gridmap"
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"


def sanitize_experiment_name(name: str) -> str:
    """Keep experiment folder names portable and predictable."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    safe_name = safe_name.strip("._-")
    if not safe_name:
        raise ValueError("experiment_name must contain at least one letter or number")
    return safe_name


def resolve_input_path(input_path: str | None) -> Path:
    """Use the requested PLY path, or fall back to the first sample that exists."""
    if input_path:
        path = Path(input_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Input point cloud not found: {path}")
        return path

    for path in DEFAULT_INPUT_CANDIDATES:
        if path.exists():
            return path

    candidates = "\n".join(f"  - {path}" for path in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(
        "No default point cloud was found. Pass a .ply file path explicitly.\n"
        f"Checked:\n{candidates}"
    )


def load_pointcloud(path: Path) -> np.ndarray:
    """Read a PLY point cloud with Open3D and return an Nx3 NumPy array."""
    point_cloud = o3d.io.read_point_cloud(str(path))
    if point_cloud.is_empty():
        raise ValueError(f"Point cloud is empty or could not be read: {path}")

    points = np.asarray(point_cloud.points, dtype=np.float64)
    return points


def apply_axis_flips(points: np.ndarray, flip_y: bool, flip_z: bool) -> np.ndarray:
    """Apply optional axis flips for datasets with different camera conventions."""
    points = points.copy()
    if flip_y:
        points[:, 1] *= -1.0
    if flip_z:
        points[:, 2] *= -1.0
    return points


def filter_points(
    points: np.ndarray,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: float,
    max_z: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Remove invalid points and keep only points inside the configured 3D ROI."""
    finite_mask = np.isfinite(points).all(axis=1)
    finite_points = points[finite_mask]

    xz_mask = (
        (finite_points[:, 0] >= min_x)
        & (finite_points[:, 0] <= max_x)
        & (finite_points[:, 2] >= min_z)
        & (finite_points[:, 2] <= max_z)
    )
    xz_points = finite_points[xz_mask]

    height_mask = (xz_points[:, 1] >= min_y) & (xz_points[:, 1] <= max_y)
    filtered_points = xz_points[height_mask]
    stats = {
        "invalid_point_count": len(points) - len(finite_points),
        "xz_filtered_point_count": len(finite_points) - len(xz_points),
        "height_filtered_point_count": len(xz_points) - len(filtered_points),
        "filtered_out_point_count": len(points) - len(filtered_points),
    }

    return filtered_points, stats


def remove_ground_points(
    points: np.ndarray,
    remove_ground: bool,
    ground_y_threshold: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Optionally remove low points with a simple Y-height threshold."""
    before_count = len(points)
    if not remove_ground:
        return points, {
            "ground_removal_enabled": False,
            "ground_removed_point_count": 0,
            "ground_before_point_count": before_count,
            "ground_after_point_count": before_count,
        }

    keep_mask = points[:, 1] >= ground_y_threshold
    kept_points = points[keep_mask]
    return kept_points, {
        "ground_removal_enabled": True,
        "ground_removed_point_count": before_count - len(kept_points),
        "ground_before_point_count": before_count,
        "ground_after_point_count": len(kept_points),
    }


def build_gridmap(
    points: np.ndarray,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
    resolution: float,
    obstacle_threshold: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Project points onto X-Z and build a binary occupancy grid."""
    if resolution <= 0:
        raise ValueError("resolution must be greater than 0")
    if obstacle_threshold <= 0:
        raise ValueError("obstacle_threshold must be greater than 0")
    if max_x <= min_x or max_z <= min_z:
        raise ValueError("max_x/max_z must be greater than min_x/min_z")

    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_z - min_z) / resolution))
    cell_counts = np.zeros((height, width), dtype=np.int32)

    if len(points) > 0:
        cols = np.floor((points[:, 0] - min_x) / resolution).astype(np.int32)
        cols = np.clip(cols, 0, width - 1)

        z_bins = np.floor((points[:, 2] - min_z) / resolution).astype(np.int32)
        z_bins = np.clip(z_bins, 0, height - 1)

        # Row 0 is the top of the image, so larger Z maps upward.
        rows = height - 1 - z_bins

        valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        np.add.at(cell_counts, (rows[valid], cols[valid]), 1)

    occupancy_grid = (cell_counts >= obstacle_threshold).astype(np.uint8)
    obstacle_cells = int(np.count_nonzero(occupancy_grid))
    return occupancy_grid, cell_counts, obstacle_cells


def filter_small_components(
    occupancy_grid: np.ndarray,
    min_component_size: int,
) -> tuple[np.ndarray, int, int]:
    """Remove small 8-connected occupied components from a binary grid."""
    if min_component_size <= 1:
        return occupancy_grid, 0, 0

    height, width = occupancy_grid.shape
    visited = np.zeros_like(occupancy_grid, dtype=bool)
    filtered_grid = occupancy_grid.copy()
    removed_components = 0
    removed_cells = 0

    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for start_row in range(height):
        for start_col in range(width):
            if visited[start_row, start_col] or occupancy_grid[start_row, start_col] == 0:
                continue

            stack = [(start_row, start_col)]
            component: list[tuple[int, int]] = []
            visited[start_row, start_col] = True

            while stack:
                row, col = stack.pop()
                component.append((row, col))
                for d_row, d_col in neighbors:
                    next_row = row + d_row
                    next_col = col + d_col
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and not visited[next_row, next_col]
                        and occupancy_grid[next_row, next_col] == 1
                    ):
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))

            if len(component) < min_component_size:
                removed_components += 1
                removed_cells += len(component)
                for row, col in component:
                    filtered_grid[row, col] = 0

    return filtered_grid, removed_components, removed_cells


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve output files while preserving the old default output behavior."""
    if args.experiment_name:
        experiment_name = sanitize_experiment_name(args.experiment_name)
        output_dir = DEFAULT_EXPERIMENTS_DIR / experiment_name
    else:
        experiment_name = None
        output_dir = DEFAULT_GRIDMAP_DIR

    output_npy = Path(args.output_npy).expanduser() if args.output_npy else output_dir / "gridmap.npy"
    output_png = Path(args.output_png).expanduser() if args.output_png else output_dir / "gridmap.png"
    if not output_npy.is_absolute():
        output_npy = PROJECT_ROOT / output_npy
    if not output_png.is_absolute():
        output_png = PROJECT_ROOT / output_png

    config_path = output_dir / "config.json"
    return output_npy, output_png, config_path, output_dir


def resolve_experiment_base_dir(args: argparse.Namespace) -> Path:
    """Resolve the base directory used by batch parameter experiments."""
    experiment_name = args.experiment_name
    if not experiment_name:
        experiment_name = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    return DEFAULT_EXPERIMENTS_DIR / sanitize_experiment_name(experiment_name)


def format_param_dir(resolution: float, obstacle_threshold: int) -> str:
    """Build stable folder names such as res_0.05_thr_3."""
    resolution_text = f"{resolution:.3f}".rstrip("0").rstrip(".")
    if "." not in resolution_text:
        resolution_text = f"{resolution_text}.0"
    return f"res_{resolution_text}_thr_{obstacle_threshold}"


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def verify_gridmap(occupancy_grid: np.ndarray) -> list[str]:
    """Run simple offline checks that catch broken grid outputs early."""
    errors: list[str] = []
    if occupancy_grid.ndim != 2:
        errors.append(f"grid must be 2D, got {occupancy_grid.ndim}D")
    if occupancy_grid.size == 0:
        errors.append("grid is empty")
    unique_values = set(np.unique(occupancy_grid).tolist())
    if not unique_values.issubset({0, 1}):
        errors.append(f"grid must contain only 0/1 values, got {sorted(unique_values)}")
    return errors


def build_tuning_advice(obstacle_ratio: float, valid_points: int) -> list[str]:
    """Create concise tuning hints for offline grid-map quality checks."""
    advice: list[str] = []
    if valid_points == 0:
        advice.append(
            "No valid points remain. Check min/max ranges and try --flip-z if depth appears reversed."
        )
    elif obstacle_ratio < 0.005:
        advice.append(
            "Obstacle cells are very sparse. Try lowering --obstacle-threshold or increasing --resolution."
        )
    elif obstacle_ratio > 0.20:
        advice.append(
            "Obstacle cells are dense. Try increasing --obstacle-threshold or narrowing --min-y/--max-y."
        )
    else:
        advice.append("Obstacle density looks reasonable for a first offline grid-map check.")

    advice.append("If the map direction looks reversed, try --flip-y or --flip-z.")
    return advice


def save_config(
    path: Path,
    args: argparse.Namespace,
    input_path: Path,
    output_npy: Path,
    output_png: Path,
    raw_count: int,
    valid_count: int,
    filter_stats: dict[str, int],
    ground_stats: dict[str, int],
    grid_width: int,
    grid_height: int,
    obstacle_cells: int,
    obstacle_ratio: float,
    min_component_size: int,
    removed_components: int,
    removed_component_cells: int,
    verification_errors: list[str],
    advice: list[str],
) -> None:
    """Save parameters and result metadata for repeatable offline experiments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_name": args.experiment_name,
        "input_ply": str(input_path),
        "output_npy": str(output_npy),
        "output_png": str(output_png),
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
        "min_component_size": min_component_size,
        "flip_y": args.flip_y,
        "flip_z": args.flip_z,
        "raw_point_count": raw_count,
        "valid_point_count": valid_count,
        **filter_stats,
        **ground_stats,
        "grid_width": grid_width,
        "grid_height": grid_height,
        "obstacle_cells": obstacle_cells,
        "obstacle_ratio": obstacle_ratio,
        "removed_small_components": removed_components,
        "removed_small_component_cells": removed_component_cells,
        "verification_passed": not verification_errors,
        "verification_errors": verification_errors,
        "tuning_advice": advice,
    }
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(
    input_path: Path,
    output_npy: Path,
    output_png: Path,
    config_path: Path,
    raw_count: int,
    valid_count: int,
    filter_stats: dict[str, int],
    ground_stats: dict[str, int],
    grid_width: int,
    grid_height: int,
    resolution: float,
    obstacle_cells: int,
    obstacle_ratio: float,
    removed_components: int,
    removed_component_cells: int,
    verification_errors: list[str],
    advice: list[str],
) -> None:
    """Print the run summary in one stable block for easier comparison."""
    print("")
    print("Grid map summary")
    print(f"  Input point cloud: {input_path}")
    print(f"  Raw point count: {raw_count}")
    print(f"  Valid point count after filtering: {valid_count}")
    print(f"  Invalid NaN/Inf point count: {filter_stats['invalid_point_count']}")
    print(f"  X/Z range filtered point count: {filter_stats['xz_filtered_point_count']}")
    print(f"  Height filtered point count: {filter_stats['height_filtered_point_count']}")
    print(f"  Filtered-out point count: {filter_stats['filtered_out_point_count']}")
    if ground_stats["ground_removal_enabled"]:
        print(
            "  Ground removal: "
            f"{ground_stats['ground_before_point_count']} -> "
            f"{ground_stats['ground_after_point_count']} points "
            f"({ground_stats['ground_removed_point_count']} removed)"
        )
    else:
        print("  Ground removal: disabled")
    print(f"  Map size: {grid_width} x {grid_height} cells")
    print(f"  Resolution: {resolution:.3f} m/cell")
    print(f"  Obstacle cells: {obstacle_cells}")
    print(f"  Obstacle ratio: {obstacle_ratio:.2%}")
    print(
        f"  Small component filter: {removed_components} components, "
        f"{removed_component_cells} cells removed"
    )
    print(f"  Verification: {'PASS' if not verification_errors else 'FAIL'}")
    for error in verification_errors:
        print(f"    - {error}")
    print(f"  Output NPY: {output_npy}")
    print(f"  Output PNG: {output_png}")
    print(f"  Config JSON: {config_path}")
    print("")
    print("Tuning advice")
    for item in advice:
        print(f"  - {item}")


def save_grid_png(
    occupancy_grid: np.ndarray,
    output_path: Path,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
    resolution: float,
    obstacle_cells: int,
) -> None:
    """Save a top-down visualization where obstacles are dark cells."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if plt is None:
        save_grid_png_with_cv2(
            occupancy_grid,
            output_path,
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z,
            resolution=resolution,
            obstacle_cells=obstacle_cells,
        )
        return

    cmap = ListedColormap(["#f4f4f4", "#111111"])
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.imshow(
        occupancy_grid,
        cmap=cmap,
        vmin=0,
        vmax=1,
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

    ax.set_title("Local Occupancy Grid (X-Z Top View)")
    ax.set_xlabel("X right positive (m)")
    ax.set_ylabel("Z forward positive (m)")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_z, max_z)
    ax.grid(color="#cfcfcf", linewidth=0.4, alpha=0.7)
    ax.text(
        0.01,
        0.99,
        f"resolution: {resolution:.3f} m/cell\nobstacle cells: {obstacle_cells}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
    )
    ax.text(min_x, min_z, "Left", va="bottom", ha="left", fontsize=8, color="#555555")
    ax.text(max_x, min_z, "Right", va="bottom", ha="right", fontsize=8, color="#555555")
    ax.text(0.0, max_z, "Forward", va="top", ha="center", fontsize=8, color="#d62728")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_grid_png_with_cv2(
    occupancy_grid: np.ndarray,
    output_path: Path,
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
    resolution: float,
    obstacle_cells: int,
) -> None:
    """OpenCV fallback for environments without Matplotlib."""
    if cv2 is None:
        raise ImportError(
            "PNG output needs either matplotlib or opencv-python. "
            "Install one of them with pip."
        )

    height, width = occupancy_grid.shape
    image = np.full((height, width, 3), 244, dtype=np.uint8)
    image[occupancy_grid == 1] = (17, 17, 17)

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
        "Local Occupancy Grid (X-Z top view)",
        f"X: {min_x:.1f} to {max_x:.1f} m",
        f"Z: {min_z:.1f} to {max_z:.1f} m",
        f"resolution: {resolution:.3f} m/cell",
        f"obstacle cells: {obstacle_cells}",
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
        description="Convert a .ply point cloud into a 2D local occupancy grid map."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input .ply point cloud. Defaults to data/pointcloud/output.ply or an existing project sample.",
    )
    parser.add_argument(
        "--output-npy",
        default=None,
        help="Output .npy path for the binary occupancy grid.",
    )
    parser.add_argument(
        "--output-png",
        default=None,
        help="Output .png path for the grid visualization.",
    )
    parser.add_argument(
        "--experiment_name",
        "--experiment-name",
        dest="experiment_name",
        default=None,
        help="Experiment name. Outputs go to data/experiments/<experiment_name>/.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run a batch experiment over resolution and obstacle-threshold values.",
    )
    parser.add_argument(
        "--resolutions",
        default="0.03,0.05,0.10",
        help="Comma-separated resolutions for --batch, in meters per cell.",
    )
    parser.add_argument(
        "--obstacle-thresholds",
        default="1,3,5",
        help="Comma-separated obstacle thresholds for --batch.",
    )
    parser.add_argument("--min-z", type=float, default=0.3, help="Minimum depth in meters.")
    parser.add_argument("--max-z", type=float, default=5.0, help="Maximum depth in meters.")
    parser.add_argument("--min-x", type=float, default=-2.5, help="Minimum X in meters.")
    parser.add_argument("--max-x", type=float, default=2.5, help="Maximum X in meters.")
    parser.add_argument("--min-y", type=float, default=-0.5, help="Minimum Y in meters.")
    parser.add_argument("--max-y", type=float, default=1.5, help="Maximum Y in meters.")
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="Grid resolution in meters per cell.",
    )
    parser.add_argument(
        "--obstacle-threshold",
        type=int,
        default=3,
        help="Minimum point count per cell to mark it as occupied.",
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
        help="When --remove_ground is enabled, remove points with Y below this value.",
    )
    parser.add_argument(
        "--min-component-size",
        "--min_component_size",
        type=int,
        default=1,
        help="Remove occupied connected components smaller than this many cells.",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Flip Y after loading if the source cloud uses downward-positive Y.",
    )
    parser.add_argument(
        "--flip-z",
        action="store_true",
        help="Flip Z after loading if the source cloud uses backward-positive Z.",
    )
    return parser.parse_args(argv)


def prepare_points(
    args: argparse.Namespace,
    input_path: Path,
) -> tuple[np.ndarray, int, dict[str, int], dict[str, int]]:
    """Load, flip, ROI-filter, and optionally remove ground points once."""
    points = load_pointcloud(input_path)
    raw_count = len(points)
    points = apply_axis_flips(points, flip_y=args.flip_y, flip_z=args.flip_z)
    filtered_points, filter_stats = filter_points(
        points,
        min_x=args.min_x,
        max_x=args.max_x,
        min_y=args.min_y,
        max_y=args.max_y,
        min_z=args.min_z,
        max_z=args.max_z,
    )
    filtered_points, ground_stats = remove_ground_points(
        filtered_points,
        remove_ground=args.remove_ground,
        ground_y_threshold=args.ground_y_threshold,
    )
    return filtered_points, raw_count, filter_stats, ground_stats


def run_gridmap_case(
    args: argparse.Namespace,
    input_path: Path,
    filtered_points: np.ndarray,
    raw_count: int,
    filter_stats: dict[str, int],
    ground_stats: dict[str, int],
    output_dir: Path,
    resolution: float,
    obstacle_threshold: int,
    print_result: bool,
    output_npy: Path | None = None,
    output_png: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, object]:
    """Generate one grid map case and save its outputs."""
    case_args = argparse.Namespace(**vars(args))
    case_args.resolution = resolution
    case_args.obstacle_threshold = obstacle_threshold

    output_npy = output_npy or output_dir / "gridmap.npy"
    output_png = output_png or output_dir / "gridmap.png"
    config_path = config_path or output_dir / "config.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    occupancy_grid, _, obstacle_cells = build_gridmap(
        filtered_points,
        min_x=args.min_x,
        max_x=args.max_x,
        min_z=args.min_z,
        max_z=args.max_z,
        resolution=resolution,
        obstacle_threshold=obstacle_threshold,
    )
    occupancy_grid, removed_components, removed_component_cells = filter_small_components(
        occupancy_grid,
        min_component_size=args.min_component_size,
    )
    obstacle_cells = int(np.count_nonzero(occupancy_grid))
    height, width = occupancy_grid.shape
    total_cells = int(occupancy_grid.size)
    obstacle_ratio = obstacle_cells / total_cells if total_cells > 0 else 0.0
    verification_errors = verify_gridmap(occupancy_grid)
    advice = build_tuning_advice(obstacle_ratio, len(filtered_points))

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, occupancy_grid)
    save_grid_png(
        occupancy_grid,
        output_png,
        min_x=args.min_x,
        max_x=args.max_x,
        min_z=args.min_z,
        max_z=args.max_z,
        resolution=resolution,
        obstacle_cells=obstacle_cells,
    )
    save_config(
        config_path,
        case_args,
        input_path=input_path,
        output_npy=output_npy,
        output_png=output_png,
        raw_count=raw_count,
        valid_count=len(filtered_points),
        filter_stats=filter_stats,
        ground_stats=ground_stats,
        grid_width=width,
        grid_height=height,
        obstacle_cells=obstacle_cells,
        obstacle_ratio=obstacle_ratio,
        min_component_size=args.min_component_size,
        removed_components=removed_components,
        removed_component_cells=removed_component_cells,
        verification_errors=verification_errors,
        advice=advice,
    )
    if print_result:
        print_summary(
            input_path=input_path,
            output_npy=output_npy,
            output_png=output_png,
            config_path=config_path,
            raw_count=raw_count,
            valid_count=len(filtered_points),
            filter_stats=filter_stats,
            ground_stats=ground_stats,
            grid_width=width,
            grid_height=height,
            resolution=resolution,
            obstacle_cells=obstacle_cells,
            obstacle_ratio=obstacle_ratio,
            removed_components=removed_components,
            removed_component_cells=removed_component_cells,
            verification_errors=verification_errors,
            advice=advice,
        )

    return {
        "resolution": resolution,
        "obstacle_threshold": obstacle_threshold,
        "min_y": args.min_y,
        "max_y": args.max_y,
        "valid_points": len(filtered_points),
        "occupied_cells": obstacle_cells,
        "occupied_ratio": obstacle_ratio,
        "removed_small_components": removed_components,
        "removed_small_component_cells": removed_component_cells,
        "output_dir": str(output_dir),
    }


def run_single_experiment(args: argparse.Namespace, input_path: Path) -> None:
    output_npy, output_png, config_path, output_dir = resolve_output_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_points, raw_count, filter_stats, ground_stats = prepare_points(args, input_path)
    run_gridmap_case(
        args,
        input_path=input_path,
        filtered_points=filtered_points,
        raw_count=raw_count,
        filter_stats=filter_stats,
        ground_stats=ground_stats,
        output_dir=output_dir,
        resolution=args.resolution,
        obstacle_threshold=args.obstacle_threshold,
        print_result=True,
        output_npy=output_npy,
        output_png=output_png,
        config_path=config_path,
    )


def write_summary_csv(summary_path: Path, rows: list[dict[str, object]]) -> None:
    """Write the batch comparison summary."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "resolution",
        "obstacle_threshold",
        "min_y",
        "max_y",
        "valid_points",
        "occupied_cells",
        "occupied_ratio",
        "output_dir",
        "removed_small_components",
        "removed_small_component_cells",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def run_batch_experiments(args: argparse.Namespace, input_path: Path) -> None:
    base_dir = resolve_experiment_base_dir(args)
    base_dir.mkdir(parents=True, exist_ok=True)
    resolutions = parse_float_list(args.resolutions)
    thresholds = parse_int_list(args.obstacle_thresholds)
    if not resolutions:
        raise ValueError("--resolutions must contain at least one value")
    if not thresholds:
        raise ValueError("--obstacle-thresholds must contain at least one value")

    filtered_points, raw_count, filter_stats, ground_stats = prepare_points(args, input_path)
    rows: list[dict[str, object]] = []

    print("")
    print("Batch grid-map experiment")
    print(f"  Base output directory: {base_dir}")
    print(f"  Resolution values: {resolutions}")
    print(f"  Obstacle thresholds: {thresholds}")

    for resolution in resolutions:
        for threshold in thresholds:
            output_dir = base_dir / format_param_dir(resolution, threshold)
            print(f"  Running {output_dir.name} ...")
            row = run_gridmap_case(
                args,
                input_path=input_path,
                filtered_points=filtered_points,
                raw_count=raw_count,
                filter_stats=filter_stats,
                ground_stats=ground_stats,
                output_dir=output_dir,
                resolution=resolution,
                obstacle_threshold=threshold,
                print_result=False,
            )
            rows.append(row)

    summary_path = base_dir / "summary.csv"
    write_summary_csv(summary_path, rows)
    print("")
    print("Batch summary")
    print(f"  Cases generated: {len(rows)}")
    print(f"  Summary CSV: {summary_path}")
    print("  Compare gridmap.png files and occupied_ratio values to choose parameters.")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = resolve_input_path(args.input)
    if args.batch:
        run_batch_experiments(args, input_path)
    else:
        run_single_experiment(args, input_path)


if __name__ == "__main__":
    main()
