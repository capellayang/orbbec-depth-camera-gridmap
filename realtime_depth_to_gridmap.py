"""Real-time Orbbec depth stream to local occupancy grid map.

This is a Python validation step before ROS2 integration:
    real-time depth frame -> depth_to_gridmap.generate_gridmap_from_depth -> gridmap

The camera backend currently reuses the project's OpenNI utilities. A later ROS2
node should replace this reader with a depth topic subscription and camera_info
intrinsics.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from depth_to_gridmap import (
    DEFAULT_FX,
    DEFAULT_FY,
    DepthGridmapConfig, 
    generate_gridmap_from_depth,
    resolve_path,
)
from pointcloud_to_gridmap import save_grid_png


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "realtime_gridmap"
DEPTH_WINDOW_NAME = "Orbbec Depth"
GRIDMAP_WINDOW_NAME = "Realtime Gridmap"


def get_cv2():
    """Import OpenCV only when the realtime script actually runs."""
    import cv2

    return cv2


def json_ready(value):
    """Convert numpy scalars/arrays and tuples into JSON-safe values."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def build_gridmap_display(cv2, gridmap: np.ndarray, metrics: dict | None) -> np.ndarray:
    """Create a fast OpenCV preview image for the binary occupancy grid."""
    if gridmap is None:
        image = np.full((320, 320, 3), 245, dtype=np.uint8)
        cv2.putText(
            image,
            "Waiting for gridmap",
            (32, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 80, 80),
            2,
            cv2.LINE_AA,
        )
        return image

    height, width = gridmap.shape
    image = np.full((height, width, 3), 244, dtype=np.uint8)
    image[gridmap == 1] = (20, 20, 20)

    max_display = 900
    scale = max(1, min(max_display // max(height, width), 10))
    display = cv2.resize(
        image,
        (width * scale, height * scale),
        interpolation=cv2.INTER_NEAREST,
    )

    camera_x = display.shape[1] // 2
    camera_y = display.shape[0] - 8
    cv2.circle(display, (camera_x, camera_y), 5, (40, 40, 220), -1)
    cv2.arrowedLine(
        display,
        (camera_x, camera_y - 4),
        (camera_x, max(camera_y - 55, 12)),
        (40, 40, 220),
        2,
        tipLength=0.3,
    )

    if metrics is not None:
        cv2.putText(
            display,
            f"occupied: {metrics['occupied_cells']} ({metrics['occupied_ratio']:.2%})",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    return display


def save_realtime_outputs(
    output_dir: Path,
    save_index: int,
    depth: np.ndarray,
    gridmap: np.ndarray,
    metrics: dict,
) -> None:
    """Save one realtime sample as depth, grid, preview PNG, and metrics JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_path = output_dir / f"depth_{save_index:04d}.npy"
    grid_npy_path = output_dir / f"gridmap_{save_index:04d}.npy"
    grid_png_path = output_dir / f"gridmap_{save_index:04d}.png"
    metrics_path = output_dir / f"metrics_{save_index:04d}.json"

    np.save(depth_path, depth)
    np.save(grid_npy_path, gridmap.astype(np.uint8))
    save_grid_png(
        gridmap,
        grid_png_path,
        min_x=metrics["min_x"],
        max_x=metrics["max_x"],
        min_z=metrics["min_z"],
        max_z=metrics["max_z"],
        resolution=metrics["resolution"],
        obstacle_cells=metrics["occupied_cells"],
    )

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "depth_npy": str(depth_path),
        "gridmap_npy": str(grid_npy_path),
        "gridmap_png": str(grid_png_path),
        **metrics,
    }
    metrics_path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_metrics(metrics: dict) -> None:
    """Print a compact live processing line."""
    print(
        "frame={frame_index} "
        "valid_depth={valid_depth_pixels} "
        "valid_points={valid_point_count} "
        "occupied={occupied_cells} "
        "ratio={occupied_ratio:.2%} "
        "time={processing_time_ms:.1f}ms "
        "fps={estimated_fps:.2f} "
        "avg_fps={average_fps:.2f}".format(**metrics)
    )


def print_low_fps_hint(average_fps: float) -> None:
    print("")
    print(f"Average processing FPS is {average_fps:.2f}, below 10 FPS.")
    print("Consider lowering camera resolution, increasing --process_every_n_frames,")
    print("shrinking the map range, increasing --resolution, or further vectorizing numpy code.")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read Orbbec depth frames in real time and generate local grid maps."
    )
    parser.add_argument("--fx", type=float, default=DEFAULT_FX)
    parser.add_argument("--fy", type=float, default=DEFAULT_FY)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--depth_unit", choices=["mm", "m"], default="mm")
    parser.add_argument("--min-x", "--min_x", dest="min_x", type=float, default=-2.5)
    parser.add_argument("--max-x", "--max_x", dest="max_x", type=float, default=2.5)
    parser.add_argument("--min-y", "--min_y", dest="min_y", type=float, default=-0.5)
    parser.add_argument("--max-y", "--max_y", dest="max_y", type=float, default=1.5)
    parser.add_argument("--min-z", "--min_z", dest="min_z", type=float, default=0.3)
    parser.add_argument("--max-z", "--max_z", dest="max_z", type=float, default=5.0)
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
    parser.add_argument("--ground-y-threshold", "--ground_y_threshold", type=float, default=-0.45)
    parser.add_argument("--min-component-size", "--min_component_size", type=int, default=1)
    parser.add_argument(
        "--y_axis_up",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert image v-down coordinates into project Y-up coordinates.",
    )
    parser.add_argument(
        "--process_every_n_frames",
        type=int,
        default=1,
        help="Only generate a gridmap every N camera frames. Default: 1",
    )
    parser.add_argument(
        "--save_every_n_frames",
        type=int,
        default=0,
        help="Save every N processed frames. 0 disables periodic saving. Default: 0",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for optional realtime samples. Default: data/realtime_gridmap/",
    )
    parser.add_argument(
        "--print_every_n_frames",
        type=int,
        default=30,
        help="Print metrics every N processed frames. Default: 30",
    )
    parser.add_argument("--warmup_seconds", type=float, default=1.0)
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show OpenCV depth and gridmap windows. Default: enabled.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Optional test limit. 0 means run until q/ESC.",
    )
    return parser.parse_args(argv)


def build_processing_config(args: argparse.Namespace) -> DepthGridmapConfig:
    return DepthGridmapConfig(
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        depth_unit=args.depth_unit,
        min_x=args.min_x,
        max_x=args.max_x,
        min_y=args.min_y,
        max_y=args.max_y,
        min_z=args.min_z,
        max_z=args.max_z,
        resolution=args.resolution,
        obstacle_threshold=args.obstacle_threshold,
        remove_ground=args.remove_ground,
        ground_y_threshold=args.ground_y_threshold,
        min_component_size=args.min_component_size,
        y_axis_up=args.y_axis_up,
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.process_every_n_frames < 1:
        raise ValueError("--process_every_n_frames must be >= 1")
    if args.save_every_n_frames < 0:
        raise ValueError("--save_every_n_frames must be >= 0")

    output_dir = resolve_path(args.output_dir)
    config = build_processing_config(args)
    context = None
    cv2 = None
    latest_gridmap = None
    latest_metrics = None
    frame_index = 0
    processed_count = 0
    save_count = 0
    total_processing_time = 0.0
    warned_low_fps = False

    try:
        # TODO(ROS2): replace this OpenNI reader with a depth Image subscription
        # and camera_info intrinsics when the ROS2 node is introduced.
        from orbbec_openni_utils import (
            is_window_open,
            make_depth_display,
            open_depth_stream,
            print_camera_startup_notice,
            read_depth_frame,
            safe_release,
        )

        print_camera_startup_notice()
        context = open_depth_stream(warmup_seconds=args.warmup_seconds)

        if args.display:
            cv2 = get_cv2()
            cv2.namedWindow(DEPTH_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.namedWindow(GRIDMAP_WINDOW_NAME, cv2.WINDOW_NORMAL)

        print("Realtime depth-to-gridmap started. Press q or ESC to quit.")
        if args.save_every_n_frames <= 0:
            print("Periodic saving is disabled. Use --save_every_n_frames N to enable it.")
        else:
            print(f"Saving every {args.save_every_n_frames} processed frames to {output_dir}")

        while True:
            frame_index += 1
            depth = read_depth_frame(context.depth_stream)
            should_process = frame_index % args.process_every_n_frames == 0

            if should_process:
                start_time = time.perf_counter()
                latest_gridmap, latest_metrics, _ = generate_gridmap_from_depth(depth, config)
                processing_time = time.perf_counter() - start_time
                processed_count += 1
                total_processing_time += processing_time

                estimated_fps = 1.0 / processing_time if processing_time > 0 else 0.0
                average_fps = (
                    processed_count / total_processing_time
                    if total_processing_time > 0
                    else 0.0
                )
                latest_metrics = {
                    **latest_metrics,
                    "frame_index": frame_index,
                    "processed_frame_index": processed_count,
                    "processing_time_ms": processing_time * 1000.0,
                    "estimated_fps": estimated_fps,
                    "average_fps": average_fps,
                }

                should_print = (
                    processed_count == 1
                    or args.print_every_n_frames <= 1
                    or processed_count % args.print_every_n_frames == 0
                )
                if should_print:
                    print_metrics(latest_metrics)
                    if average_fps < 10.0 and processed_count >= 10 and not warned_low_fps:
                        print_low_fps_hint(average_fps)
                        warned_low_fps = True

                should_save = (
                    args.save_every_n_frames > 0
                    and processed_count % args.save_every_n_frames == 0
                )
                if should_save:
                    save_count += 1
                    save_realtime_outputs(
                        output_dir=output_dir,
                        save_index=save_count,
                        depth=depth,
                        gridmap=latest_gridmap,
                        metrics=latest_metrics,
                    )
                    print(f"Saved realtime sample {save_count:04d} to {output_dir}")

            key = 255
            if cv2 is not None:
                depth_display = make_depth_display(cv2, depth, color=True)
                if latest_metrics is not None:
                    cv2.putText(
                        depth_display,
                        f"frame {frame_index} | fps {latest_metrics['average_fps']:.2f}",
                        (20, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(DEPTH_WINDOW_NAME, depth_display)
                cv2.imshow(
                    GRIDMAP_WINDOW_NAME,
                    build_gridmap_display(cv2, latest_gridmap, latest_metrics),
                )

                if (
                    not is_window_open(cv2, DEPTH_WINDOW_NAME)
                    or not is_window_open(cv2, GRIDMAP_WINDOW_NAME)
                ):
                    print("Display window closed, exiting safely.")
                    break
                key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                print("Exit key received.")
                break
            if args.max_frames > 0 and frame_index >= args.max_frames:
                print(f"Reached --max_frames {args.max_frames}.")
                break

    except Exception:
        print("Realtime depth-to-gridmap failed:")
        traceback.print_exc()

    finally:
        average_fps = (
            processed_count / total_processing_time if total_processing_time > 0 else 0.0
        )
        if processed_count > 0:
            print("")
            print(f"Processed frames: {processed_count}")
            print(f"Average processing FPS: {average_fps:.2f}")
            if average_fps < 10.0:
                print_low_fps_hint(average_fps)

        if "safe_release" in locals():
            safe_release(context, cv2)


if __name__ == "__main__":
    main()
