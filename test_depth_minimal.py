"""Orbbec 深度流最小稳定性测试。

用于确认 OpenNI2 可以打开设备、读取 uint16 深度帧，并安全退出。
"""

import traceback

from orbbec_openni_utils import (
    is_window_open,
    make_depth_display,
    open_depth_stream,
    print_camera_startup_notice,
    read_depth_frame,
    safe_release,
)


WINDOW_NAME = "Astra Depth Minimal"
cv2 = None


def get_cv2():
    """延迟导入 OpenCV，先完成 OpenNI 深度流初始化。"""
    global cv2
    if cv2 is None:
        import cv2 as cv2_module

        cv2 = cv2_module
    return cv2


def main():
    """显示实时深度图和中心点深度，按 q 或 ESC 安全退出。"""
    global cv2
    context = None
    print_camera_startup_notice()

    try:
        context = open_depth_stream()
        cv2 = get_cv2()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        print("深度流启动成功")
        print("按 q 或 ESC 安全退出")

        while True:
            depth = read_depth_frame(context.depth_stream)
            height, width = depth.shape
            center_depth = int(depth[height // 2, width // 2])

            display = make_depth_display(cv2, depth, color=True)
            cv2.putText(
                display,
                f"Center: {center_depth} mm",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                "Press q/ESC to quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, display)

            if not is_window_open(cv2, WINDOW_NAME):
                print("检测到窗口已关闭，准备安全退出")
                break

            key = cv2.waitKey(1) & 0xFF
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
