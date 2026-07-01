"""实时点击 Orbbec 深度图，输出像素点的 u/v/depth/X/Y/Z。

鼠标回调只读取最新的 uint16 原始深度帧，不读取 uint8 显示图。
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


# 相机内参，单位：像素
FX = 574.9614356132867
FY = 574.9614061048316
CX = 320.0
CY = 240.0
DEPTH_SCALE = 1000.0  # 原始深度单位 mm -> m

WINDOW_NAME = "Click Depth Point"

cv2 = None
latest_depth = None
latest_display = None


def get_cv2():
    """延迟导入 OpenCV，先让 OpenNI 深度流完成初始化。"""
    global cv2
    if cv2 is None:
        import cv2 as cv2_module

        cv2 = cv2_module
    return cv2


def mouse_callback(event, x, y, flags, param):
    """鼠标点击后，从 latest_depth 中读取原始深度并反投影到 3D。"""
    global latest_depth, latest_display

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if latest_depth is None:
        print("当前还没有可用深度帧")
        return

    height, width = latest_depth.shape
    u = int(x)
    v = int(y)

    if u < 0 or u >= width or v < 0 or v >= height:
        print("点击位置超出深度图范围")
        return

    depth_raw = int(latest_depth[v, u])
    print("\n==============================")
    print(f"u, v: {u}, {v}")
    print(f"depth raw: {depth_raw} mm")

    if depth_raw == 0:
        print("该点深度为 0，通常表示无效深度")
        print("==============================")
        return

    z = depth_raw / DEPTH_SCALE
    x_m = (u - CX) * z / FX
    y_m = (v - CY) * z / FY

    print(f"X: {x_m:.6f} m")
    print(f"Y: {y_m:.6f} m")
    print(f"Z: {z:.6f} m")
    print("==============================")

    if latest_display is not None:
        display = latest_display.copy()
        cv2.circle(display, (u, v), 5, (255, 255, 255), -1)
        cv2.putText(
            display,
            f"u={u}, v={v}, Z={z:.3f}m",
            (min(u + 10, width - 220), max(v - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, display)


def main():
    """启动实时深度流，点击窗口中的点并输出 3D 坐标。"""
    global cv2, latest_depth, latest_display

    context = None
    print_camera_startup_notice()

    try:
        context = open_depth_stream()
        cv2 = get_cv2()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

        print("点击窗口中的任意点，会输出 u, v, depth raw, X, Y, Z")
        print("按 q 或 ESC 安全退出")

        while True:
            depth = read_depth_frame(context.depth_stream)
            latest_depth = depth.copy()

            center_u = depth.shape[1] // 2
            center_v = depth.shape[0] // 2
            center_depth = int(latest_depth[center_v, center_u])

            display = make_depth_display(cv2, latest_depth, color=True)
            cv2.putText(
                display,
                f"Center: {center_depth} mm",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                "Click point to print depth, q/ESC to quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            latest_display = display.copy()
            cv2.imshow(WINDOW_NAME, latest_display)

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
