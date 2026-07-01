"""实时读取 Orbbec 深度流，并安全保存一帧深度图。

输出文件：
1. depth_0001.npy：uint16 原始深度数组，供后续点云生成脚本读取。
2. depth_raw.png：uint16 原始深度图，不 normalize，不转 uint8。
3. depth_display.png：uint8 伪彩色显示图，仅用于观察。
"""

import os
import traceback

import numpy as np

from orbbec_openni_utils import (
    is_window_open,
    make_depth_display,
    open_depth_stream,
    print_camera_startup_notice,
    read_depth_frame,
    safe_release,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "depth_data")
WINDOW_NAME = "Astra Depth"

# 自动保存模式用于命令行批处理，普通手动运行时不受影响。
AUTO_SAVE = os.environ.get("AUTO_SAVE_DEPTH_FRAME") == "1"
AUTO_SAVE_WARMUP_FRAMES = int(os.environ.get("AUTO_SAVE_WARMUP_FRAMES", "60"))
AUTO_SAVE_QUIT_AFTER_SAVE = os.environ.get("AUTO_SAVE_QUIT_AFTER_SAVE", "1") != "0"
ENABLE_PREVIEW_WINDOW = os.environ.get("DEPTH_PREVIEW_WINDOW", "1") != "0"


def get_cv2():
    """延迟导入 OpenCV，减少 OpenCV 窗口和 OpenNI 初始化互相影响的概率。"""
    import cv2

    return cv2


def print_depth_stats(depth):
    """保存前打印关键深度图信息。"""
    center_depth = int(depth[depth.shape[0] // 2, depth.shape[1] // 2])
    valid_depth = depth[depth > 0]

    print("depth shape:", depth.shape)
    print("depth dtype:", depth.dtype)
    if depth.dtype != np.uint16:
        print("警告: depth dtype 不是 uint16，请检查 OpenNI 帧格式")

    if valid_depth.size > 0:
        print("min depth:", int(valid_depth.min()), "mm")
        print("max depth:", int(valid_depth.max()), "mm")
    else:
        print("min depth: None")
        print("max depth: None")
    print("center depth:", center_depth, "mm")


def save_depth_files(depth, display_image, save_index):
    """分别保存原始深度数据、uint16 原始 PNG 和 uint8 显示图。"""
    os.makedirs(SAVE_DIR, exist_ok=True)

    npy_path = os.path.join(SAVE_DIR, f"depth_{save_index:04d}.npy")
    raw_png_path = os.path.join(SAVE_DIR, "depth_raw.png")
    display_png_path = os.path.join(SAVE_DIR, "depth_display.png")
    numbered_display_path = os.path.join(SAVE_DIR, f"depth_show_{save_index:04d}.png")

    np.save(npy_path, depth)
    print(f"已保存原始深度数组: {npy_path}")
    print_depth_stats(depth)

    cv2 = get_cv2()
    if not cv2.imwrite(raw_png_path, depth):
        raise IOError(f"保存 uint16 原始深度图失败: {raw_png_path}")
    if not cv2.imwrite(display_png_path, display_image):
        raise IOError(f"保存显示图失败: {display_png_path}")
    if not cv2.imwrite(numbered_display_path, display_image):
        raise IOError(f"保存编号显示图失败: {numbered_display_path}")

    print(f"已保存 uint16 原始深度图: {raw_png_path}")
    print(f"已保存 uint8 显示图: {display_png_path}")


def main():
    """打开深度流，显示预览并按键保存深度帧。"""
    context = None
    cv2 = None
    save_count = 0
    frame_count = 0

    print_camera_startup_notice()

    try:
        context = open_depth_stream()

        if ENABLE_PREVIEW_WINDOW:
            cv2 = get_cv2()
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        print("操作说明：")
        print("按 s 保存当前深度图")
        print("按 q 或 ESC 安全退出")
        if AUTO_SAVE:
            print("自动保存模式已开启")

        while True:
            frame_count += 1
            depth = read_depth_frame(context.depth_stream)
            center_depth = int(depth[depth.shape[0] // 2, depth.shape[1] // 2])

            if cv2 is not None:
                display_image = make_depth_display(cv2, depth, color=True)
                cv2.putText(
                    display_image,
                    f"Center: {center_depth} mm",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display_image,
                    "Press s to save, q/ESC to quit",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, display_image)

                if not is_window_open(cv2, WINDOW_NAME):
                    print("检测到窗口已关闭，准备安全退出")
                    break

                key = cv2.waitKey(1) & 0xFF
            else:
                display_image = np.clip(depth.astype(np.float32) * 0.03, 0, 255).astype(np.uint8)
                key = 255

            should_auto_save = AUTO_SAVE and frame_count >= AUTO_SAVE_WARMUP_FRAMES
            if key == ord("s") or should_auto_save:
                save_count += 1
                save_depth_files(depth, display_image, save_count)

                if should_auto_save and AUTO_SAVE_QUIT_AFTER_SAVE:
                    print("自动保存完成，准备安全退出")
                    break

            if key == ord("q") or key == 27:
                print("收到安全退出按键")
                break

    except Exception:
        print("程序异常：")
        traceback.print_exc()

    finally:
        safe_release(context, cv2)


if __name__ == "__main__":
    main()
