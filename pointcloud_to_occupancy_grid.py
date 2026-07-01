import os

try:
    import numpy as np
except ImportError as exc:
    raise ImportError("缺少 NumPy，请先安装 numpy，例如: pip install numpy") from exc

try:
    import cv2
except ImportError as exc:
    raise ImportError("缺少 OpenCV，请先安装 opencv-python，例如: pip install opencv-python") from exc

try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError("缺少 Open3D，请先安装 open3d，例如: pip install open3d") from exc


# =========================
# 基础配置区
# =========================
# 可以在这里切换输入点云文件，例如：
# INPUT_PLY = "astra_pointcloud.ply"
# INPUT_PLY = "astra_pointcloud_processed.ply"
INPUT_PLY = "person.ply"

FILTERED_PLY = "filtered_for_grid.ply"
OUTPUT_IMAGE = "occupancy_grid.png"

# 点云过滤范围，单位：米
MIN_X = -2.0
MAX_X = 2.0
MIN_Z = 0.3
MAX_Z = 4.0

# 占据栅格地图范围，单位：米
GRID_MIN_X = -2.0
GRID_MAX_X = 2.0
GRID_MIN_Z = 0.0
GRID_MAX_Z = 4.0

# 栅格分辨率：每个格子代表多少米
RESOLUTION = 0.05

# 一个栅格内至少有多少个点，才认为这个栅格被障碍物占据
MIN_POINTS_PER_CELL = 3

# 可选体素下采样。设置为 None 可关闭下采样。
VOXEL_SIZE = 0.02

# OpenCV 显示窗口倍率。栅格本身是 80x80，放大后更方便观察。
DISPLAY_SCALE = 8


def load_pointcloud(path):
    """读取 .ply 点云文件，并打印点数和坐标范围。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到点云文件: {path}")

    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"点云为空或读取失败: {path}")

    print("点云读取成功")
    print(f"输入文件: {path}")
    print_pointcloud_stats(pcd, "原始点云")
    return pcd


def print_pointcloud_stats(pcd, name):
    """打印点云点数，以及 X/Y/Z 三个方向的坐标范围。"""
    points = np.asarray(pcd.points)
    print(f"{name}点数: {len(points)}")

    if len(points) == 0:
        print(f"{name}为空，无法统计坐标范围")
        return

    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    print(f"{name}坐标范围:")
    print(f"  min X = {min_xyz[0]:.3f} m, max X = {max_xyz[0]:.3f} m")
    print(f"  min Y = {min_xyz[1]:.3f} m, max Y = {max_xyz[1]:.3f} m")
    print(f"  min Z = {min_xyz[2]:.3f} m, max Z = {max_xyz[2]:.3f} m")


def filter_pointcloud(pcd):
    """去除无效点，并按 X/Z 范围过滤点云。"""
    points = np.asarray(pcd.points)
    print("\n开始点云过滤...")

    # np.isfinite 同时去除 NaN、正无穷和负无穷，避免后续索引计算出错。
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    print(f"去除 NaN/无穷点后点数: {len(points)}")

    range_mask = (
        (points[:, 2] > MIN_Z)
        & (points[:, 2] < MAX_Z)
        & (points[:, 0] > MIN_X)
        & (points[:, 0] < MAX_X)
    )
    points = points[range_mask]
    print(f"按 X/Z 范围过滤后点数: {len(points)}")

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points)

    if VOXEL_SIZE is not None and VOXEL_SIZE > 0:
        print(f"执行体素下采样，voxel_size = {VOXEL_SIZE:.3f} m")
        filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        print(f"体素下采样后点数: {len(filtered_pcd.points)}")

    print_pointcloud_stats(filtered_pcd, "过滤后点云")
    return filtered_pcd


def pointcloud_to_occupancy_grid(pcd):
    """将 X-Z 平面上的点投影到 2D 占据栅格地图。"""
    points = np.asarray(pcd.points)

    grid_width = int((GRID_MAX_X - GRID_MIN_X) / RESOLUTION)
    grid_height = int((GRID_MAX_Z - GRID_MIN_Z) / RESOLUTION)
    print("\n开始生成 X-Z 平面占据栅格地图...")
    print(f"地图宽度: {grid_width} 格")
    print(f"地图高度: {grid_height} 格")
    print(f"地图分辨率: {RESOLUTION:.3f} m/格")

    cell_counts = np.zeros((grid_height, grid_width), dtype=np.int32)

    # X 轴映射到图像列：左侧为负 X，右侧为正 X。
    col_indices = ((points[:, 0] - GRID_MIN_X) / RESOLUTION).astype(np.int32)

    # Z 轴映射到图像行：图像第 0 行在最上方，所以越远的 Z 越靠上。
    row_indices = ((GRID_MAX_Z - points[:, 2]) / RESOLUTION).astype(np.int32)

    valid_mask = (
        (col_indices >= 0)
        & (col_indices < grid_width)
        & (row_indices >= 0)
        & (row_indices < grid_height)
    )
    col_indices = col_indices[valid_mask]
    row_indices = row_indices[valid_mask]

    # np.add.at 可以正确处理多个点落在同一个栅格里的累加。
    np.add.at(cell_counts, (row_indices, col_indices), 1)

    occupied_grid = cell_counts >= MIN_POINTS_PER_CELL
    occupied_count = int(np.count_nonzero(occupied_grid))
    print(f"参与栅格投影的点数: {len(col_indices)}")
    print(f"占据栅格阈值: 每格至少 {MIN_POINTS_PER_CELL} 个点")
    print(f"占据栅格数量: {occupied_count}")

    return occupied_grid, cell_counts, occupied_count


def visualize_grid(occupied_grid, occupied_count):
    """生成、显示并保存 2D 占据栅格地图图片。"""
    grid_height, grid_width = occupied_grid.shape

    # 灰色表示空闲/未知，黑色表示障碍物占据。
    image = np.full((grid_height, grid_width, 3), 180, dtype=np.uint8)
    image[occupied_grid] = (0, 0, 0)

    # 将小栅格图放大，方便观察和标注。
    display = cv2.resize(
        image,
        (grid_width * DISPLAY_SCALE, grid_height * DISPLAY_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )

    height, width = display.shape[:2]
    camera_x = width // 2
    camera_y = height - 1

    # 在底部中间标记相机位置。相机朝向为向上，也就是 Z 变大的方向。
    cv2.circle(display, (camera_x, camera_y - 10), 8, (0, 0, 255), -1)
    cv2.arrowedLine(
        display,
        (camera_x, camera_y - 10),
        (camera_x, camera_y - 50),
        (0, 0, 255),
        2,
        tipLength=0.35,
    )
    cv2.putText(
        display,
        "Camera",
        (camera_x + 12, camera_y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    # 标注地图参数，帮助确认坐标和分辨率是否符合预期。
    labels = [
        f"X: {GRID_MIN_X:.1f}m to {GRID_MAX_X:.1f}m",
        f"Z: {GRID_MIN_Z:.1f}m to {GRID_MAX_Z:.1f}m",
        f"resolution: {RESOLUTION:.2f} m/cell",
        f"occupied cells: {occupied_count}",
    ]
    for index, text in enumerate(labels):
        y = 22 + index * 22
        cv2.putText(
            display,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 220),
            2,
            cv2.LINE_AA,
        )

    # 在底部标注 X 方向，在左侧标注 Z 方向。
    cv2.putText(
        display,
        "-X",
        (10, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        "+X",
        (width - 35, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        "+Z far",
        (width - 75, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    success = cv2.imwrite(OUTPUT_IMAGE, display)
    if not success:
        raise IOError(f"占据栅格地图图片保存失败: {OUTPUT_IMAGE}")

    print(f"\n占据栅格地图图片已保存: {OUTPUT_IMAGE}")
    cv2.imshow("2D Local Occupancy Grid (X-Z)", display)
    print("按任意键关闭地图窗口")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_pointcloud(pcd, path):
    """保存过滤后的点云，便于用 Open3D 或 CloudCompare 单独检查。"""
    success = o3d.io.write_point_cloud(path, pcd)
    if not success:
        raise IOError(f"过滤后点云保存失败: {path}")
    print(f"过滤后的点云已保存: {path}")


def main():
    """离线执行：读取单帧点云，过滤，然后投影为 2D 局部占据栅格地图。"""
    pcd = load_pointcloud(INPUT_PLY)
    filtered_pcd = filter_pointcloud(pcd)
    save_pointcloud(filtered_pcd, FILTERED_PLY)

    occupied_grid, _, occupied_count = pointcloud_to_occupancy_grid(filtered_pcd)
    visualize_grid(occupied_grid, occupied_count)


if __name__ == "__main__":
    main()
