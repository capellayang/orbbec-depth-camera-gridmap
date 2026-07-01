import json
import time
import traceback
from datetime import datetime

import numpy as np
from orbbec_openni_utils import (
    is_window_open,
    open_depth_stream,
    print_camera_startup_notice,
    read_depth_frame,
    safe_release,
)

RESULT_JSON = "obstacle_result.json"

MIN_VALID_DEPTH_MM = 300
MAX_VALID_DEPTH_MM = 4000
DANGER_DISTANCE_MM = 800
WARNING_DISTANCE_MM = 1200

ROI_Y_START_RATIO = 0.35
ROI_Y_END_RATIO = 0.90
ROI_X_START_RATIO = 0.15
ROI_X_END_RATIO = 0.85

MIN_REGION_VALID_POINTS = 30
cv2 = None


def get_cv2():
    """延迟导入 OpenCV，先完成 OpenNI 深度流初始化。"""
    global cv2
    if cv2 is None:
        import cv2 as cv2_module

        cv2 = cv2_module
    return cv2

def get_roi(depth):
    """截取用于避障检测的中下部 ROI。"""
    height, width = depth.shape

    x_start = int(width * ROI_X_START_RATIO)
    x_end = int(width * ROI_X_END_RATIO)
    y_start = int(height * ROI_Y_START_RATIO)
    y_end = int(height * ROI_Y_END_RATIO)

    roi = depth[y_start:y_end, x_start:x_end]
    bounds = {
        "x_start": x_start,
        "x_end": x_end,
        "y_start": y_start,
        "y_end": y_end,
    }
    return roi, bounds


def split_roi_regions(roi):
    """将 ROI 横向划分为 left/front/right 三个区域。"""
    height, width = roi.shape
    split_1 = width // 3
    split_2 = (width * 2) // 3

    return {
        "left": {
            "depth": roi[:, :split_1],
            "x_start": 0,
            "x_end": split_1,
            "y_start": 0,
            "y_end": height,
        },
        "front": {
            "depth": roi[:, split_1:split_2],
            "x_start": split_1,
            "x_end": split_2,
            "y_start": 0,
            "y_end": height,
        },
        "right": {
            "depth": roi[:, split_2:],
            "x_start": split_2,
            "x_end": width,
            "y_start": 0,
            "y_end": height,
        },
    }


def compute_region_distance(region):
    """计算单个区域的有效点数量和代表距离。"""
    valid_mask = (
        (region > MIN_VALID_DEPTH_MM)
        & (region < MAX_VALID_DEPTH_MM)
    )
    valid_depths = region[valid_mask]
    valid_count = int(valid_depths.size)

    if valid_count < MIN_REGION_VALID_POINTS:
        return {
            "valid_count": valid_count,
            "min_distance_mm": None,
            "median_distance_mm": None,
            "percentile10_distance_mm": None,
        }

    return {
        "valid_count": valid_count,
        "min_distance_mm": int(np.min(valid_depths)),
        "median_distance_mm": int(np.median(valid_depths)),
        "percentile10_distance_mm": int(np.percentile(valid_depths, 10)),
    }


def decide_action(left_distance, front_distance, right_distance):
    """根据左右前方距离判断避障状态和建议动作。"""
    if front_distance is None:
        return {
            "obstacle_detected": True,
            "danger_level": "unknown",
            "suggested_action": "stop",
        }

    if front_distance < DANGER_DISTANCE_MM:
        left_clearance = left_distance if left_distance is not None else -1
        right_clearance = right_distance if right_distance is not None else -1

        if left_clearance < 0 and right_clearance < 0:
            suggested_action = "stop"
        elif left_clearance > right_clearance:
            suggested_action = "turn_left"
        else:
            suggested_action = "turn_right"

        return {
            "obstacle_detected": True,
            "danger_level": "danger",
            "suggested_action": suggested_action,
        }

    if front_distance < WARNING_DISTANCE_MM:
        return {
            "obstacle_detected": True,
            "danger_level": "warning",
            "suggested_action": "slow_down",
        }

    return {
        "obstacle_detected": False,
        "danger_level": "safe",
        "suggested_action": "go_forward",
    }


def analyze_obstacle(depth):
    """分析一帧深度图，输出三区域距离和避障建议。"""
    roi, roi_bounds = get_roi(depth)
    regions = split_roi_regions(roi)

    region_stats = {}
    region_bounds = {}
    for name, region_info in regions.items():
        stats = compute_region_distance(region_info["depth"])
        region_stats[name] = stats

        if stats["percentile10_distance_mm"] is None:
            print(f"提示: {name} 区域有效深度点不足: {stats['valid_count']}")

        region_bounds[name] = {
            "x_start": roi_bounds["x_start"] + region_info["x_start"],
            "x_end": roi_bounds["x_start"] + region_info["x_end"],
            "y_start": roi_bounds["y_start"] + region_info["y_start"],
            "y_end": roi_bounds["y_start"] + region_info["y_end"],
        }

    left_distance = region_stats["left"]["percentile10_distance_mm"]
    front_distance = region_stats["front"]["percentile10_distance_mm"]
    right_distance = region_stats["right"]["percentile10_distance_mm"]

    action = decide_action(left_distance, front_distance, right_distance)

    return {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "left_distance_mm": left_distance,
        "front_distance_mm": front_distance,
        "right_distance_mm": right_distance,
        "obstacle_detected": action["obstacle_detected"],
        "danger_level": action["danger_level"],
        "suggested_action": action["suggested_action"],
        "valid_counts": {
            "left": region_stats["left"]["valid_count"],
            "front": region_stats["front"]["valid_count"],
            "right": region_stats["right"]["valid_count"],
        },
        "region_stats": region_stats,
        "roi_bounds": roi_bounds,
        "region_bounds": region_bounds,
    }


def _format_distance(distance):
    """格式化距离显示文本。"""
    if distance is None:
        return "None"
    return f"{distance}mm"


def draw_visualization(depth, result):
    """绘制深度图、ROI、三区域距离和避障状态。"""
    valid_mask = (
        (depth > MIN_VALID_DEPTH_MM)
        & (depth < MAX_VALID_DEPTH_MM)
    )
    clipped_depth = np.clip(depth, MIN_VALID_DEPTH_MM, MAX_VALID_DEPTH_MM)
    depth_show = cv2.convertScaleAbs(clipped_depth, alpha=255.0 / MAX_VALID_DEPTH_MM)
    depth_show[~valid_mask] = 0
    depth_color = cv2.applyColorMap(depth_show, cv2.COLORMAP_JET)

    roi_bounds = result["roi_bounds"]
    cv2.rectangle(
        depth_color,
        (roi_bounds["x_start"], roi_bounds["y_start"]),
        (roi_bounds["x_end"], roi_bounds["y_end"]),
        (255, 255, 255),
        2,
    )

    colors = {
        "left": (255, 180, 0),
        "front": (0, 255, 255),
        "right": (255, 180, 0),
    }
    labels = {
        "left": f"L: {_format_distance(result['left_distance_mm'])}",
        "front": f"F: {_format_distance(result['front_distance_mm'])}",
        "right": f"R: {_format_distance(result['right_distance_mm'])}",
    }

    for name, bounds in result["region_bounds"].items():
        cv2.rectangle(
            depth_color,
            (bounds["x_start"], bounds["y_start"]),
            (bounds["x_end"], bounds["y_end"]),
            colors[name],
            2,
        )
        cv2.putText(
            depth_color,
            labels[name],
            (bounds["x_start"] + 10, bounds["y_start"] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colors[name],
            2,
        )

    level = result["danger_level"]
    action = result["suggested_action"]
    status_color = (0, 255, 0)
    if level == "warning":
        status_color = (0, 220, 255)
    elif level in ("danger", "unknown"):
        status_color = (0, 0, 255)

    cv2.putText(
        depth_color,
        f"level: {level}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2,
    )
    cv2.putText(
        depth_color,
        f"action: {action}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2,
    )
    cv2.putText(
        depth_color,
        "Press q/ESC to quit",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return depth_color


def save_result_json(result, path):
    """保存最新避障检测结果为 JSON。"""
    json_result = {
        "timestamp": result["timestamp"],
        "left_distance_mm": result["left_distance_mm"],
        "front_distance_mm": result["front_distance_mm"],
        "right_distance_mm": result["right_distance_mm"],
        "obstacle_detected": result["obstacle_detected"],
        "danger_level": result["danger_level"],
        "suggested_action": result["suggested_action"],
        "valid_counts": result["valid_counts"],
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(json_result, file, ensure_ascii=False, indent=2)


def main():
    """实时读取深度流并执行通用避障检测。"""
    global cv2
    context = None
    window_name = "Depth Obstacle Detector"
    print_camera_startup_notice()

    try:
        context = open_depth_stream()
        cv2 = get_cv2()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("开始避障检测，按 q 或 ESC 安全退出")
        while True:
            depth = read_depth_frame(context.depth_stream)
            result = analyze_obstacle(depth)
            save_result_json(result, RESULT_JSON)

            print(
                f"left={_format_distance(result['left_distance_mm'])}, "
                f"front={_format_distance(result['front_distance_mm'])}, "
                f"right={_format_distance(result['right_distance_mm'])}, "
                f"level={result['danger_level']}, "
                f"action={result['suggested_action']}"
            )

            depth_color = draw_visualization(depth, result)
            cv2.imshow(window_name, depth_color)

            if not is_window_open(cv2, window_name):
                print("检测到窗口已关闭，准备安全退出")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("收到安全退出按键")
                break

            time.sleep(0.02)

    except Exception:
        print("程序异常:")
        traceback.print_exc()

    finally:
        safe_release(context, cv2)


if __name__ == "__main__":
    main()
