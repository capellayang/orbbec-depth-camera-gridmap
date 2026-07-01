"""Orbbec Astra Pro Plus 的 OpenNI2 公共工具。

本文件只负责实时深度流的稳定初始化、读取和释放，避免多个脚本各自
维护不同的 OpenNI 路径和退出逻辑。
"""

import os
import time
from dataclasses import dataclass

import numpy as np
from openni import openni2


OPENNI_DIR = (
    r"F:\Orbbec\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows"
    r"\Win64-Release\tools\NiViewer"
)


_openni_initialized = False


@dataclass
class DepthCameraContext:
    """记录 OpenNI 和深度流状态，便于 finally 中按状态安全释放。"""

    device: object = None
    depth_stream: object = None
    openni_initialized: bool = False
    depth_stream_started: bool = False


def print_camera_startup_notice():
    """打印启动前提示，减少设备被占用导致的崩溃和误判。"""
    print("请确保 NiViewer 或其他 Python 深度相机程序已经关闭。")
    print("按 q 或 ESC 安全退出，不建议直接点击窗口右上角关闭。")
    print(f"OpenNI 路径: {OPENNI_DIR}")


def init_openni():
    """使用统一路径初始化 OpenNI2，同一进程内避免重复 initialize。"""
    global _openni_initialized

    if _openni_initialized:
        print("OpenNI 已经初始化，跳过重复初始化")
        return True

    if not os.path.isdir(OPENNI_DIR):
        raise FileNotFoundError(f"OpenNI 路径不存在: {OPENNI_DIR}")

    os.chdir(OPENNI_DIR)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(OPENNI_DIR)

    print("正在初始化 OpenNI...")
    try:
        openni2.initialize(OPENNI_DIR)
    except Exception as exc:
        raise RuntimeError(
            "OpenNI 初始化失败。可能原因：OpenNI 路径不一致、驱动未正确安装，"
            "或上一个程序没有正常释放相机。"
        ) from exc

    _openni_initialized = True
    print("OpenNI initialized")
    return True


def open_depth_stream(warmup_seconds=1.0):
    """打开设备并启动深度流，返回带状态的上下文对象。"""
    context = DepthCameraContext()
    context.openni_initialized = init_openni()

    print("正在打开设备...")
    try:
        context.device = openni2.Device.open_any()
    except Exception as exc:
        raise RuntimeError(
            "打开 Orbbec 设备失败。可能原因：相机未连接、NiViewer 占用了相机、"
            "上一个 Python 程序没有正常释放相机，或 OpenNI 路径不一致。"
        ) from exc

    print("设备已打开")
    print(context.device.get_device_info())

    if warmup_seconds > 0:
        time.sleep(warmup_seconds)

    print("正在创建深度流...")
    try:
        context.depth_stream = context.device.create_depth_stream()
    except Exception as exc:
        raise RuntimeError(
            "创建深度流失败。可能原因：相机被占用、驱动状态异常、USB 连接不稳定，"
            "或上一个程序没有正常释放深度流。"
        ) from exc

    print("正在启动深度流...")
    try:
        context.depth_stream.start()
    except Exception as exc:
        raise RuntimeError("启动深度流失败，请尝试拔插相机后重试。") from exc

    context.depth_stream_started = True
    print("depth_stream started")
    return context


def read_depth_frame(depth_stream):
    """读取一帧 uint16 深度图，单位 mm；读取后立刻 copy。"""
    frame = depth_stream.read_frame()
    width = frame.width
    height = frame.height
    frame_data = frame.get_buffer_as_uint16()
    depth = np.frombuffer(frame_data, dtype=np.uint16).reshape((height, width)).copy()
    return depth


def safe_release(context=None, cv2_module=None):
    """按固定顺序安全释放资源：停止深度流、卸载 OpenNI、销毁 OpenCV 窗口。"""
    global _openni_initialized

    depth_stream = getattr(context, "depth_stream", None)
    depth_stream_started = bool(getattr(context, "depth_stream_started", False))
    openni_initialized = bool(getattr(context, "openni_initialized", _openni_initialized))

    print("正在释放资源...")
    print("准备停止深度流")
    if depth_stream is not None and depth_stream_started:
        try:
            depth_stream.stop()
            print("depth_stream stopped")
        except Exception as exc:
            print("停止深度流失败:", exc)
    else:
        print("depth_stream 未启动或不存在，跳过 stop")

    if openni_initialized or _openni_initialized:
        try:
            openni2.unload()
            _openni_initialized = False
            print("OpenNI unloaded")
        except Exception as exc:
            print("OpenNI unload 失败:", exc)
    else:
        print("OpenNI 未初始化，跳过 unload")

    if cv2_module is not None:
        try:
            cv2_module.destroyAllWindows()
            print("OpenCV windows destroyed")
        except Exception as exc:
            print("销毁 OpenCV 窗口失败:", exc)

    print("程序安全退出")


def is_window_open(cv2_module, window_name):
    """检测 OpenCV 窗口是否仍然存在，避免点 X 后主循环继续 read_frame。"""
    try:
        return cv2_module.getWindowProperty(window_name, cv2_module.WND_PROP_VISIBLE) >= 1
    except Exception:
        return False


def make_depth_display(cv2_module, depth, color=True):
    """把 uint16 深度图转换成仅用于显示的 uint8 图。"""
    depth_display = cv2_module.normalize(
        depth,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2_module.NORM_MINMAX,
    )
    depth_display = depth_display.astype(np.uint8)
    if color:
        return cv2_module.applyColorMap(depth_display, cv2_module.COLORMAP_JET)
    return depth_display
