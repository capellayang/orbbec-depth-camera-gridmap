import os

import numpy as np
import open3d as o3d


DEPTH_PATH = os.path.join("depth_data", "depth_0007.npy")
OUTPUT_PLY = "astra_pointcloud_raw.ply"

FX = 580.0
FY = 580.0
MIN_Z = 0.5
MAX_Z = 4.0


def load_depth(path):
    """读取保存好的深度图 .npy 文件。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到深度图文件: {path}")

    return np.load(path)


def depth_to_point_cloud(depth, fx, fy, cx, cy, min_z, max_z):
    """使用相机内参将深度图转换为点云，只保留指定距离范围内的点。"""
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    z = depth.astype(np.float32) / 1000.0
    valid = (z >= min_z) & (z <= max_z)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    y = -y

    points = np.stack((x, y, z), axis=-1)
    return points[valid]


def create_point_cloud(points):
    """根据三维点数组创建 Open3D 点云对象。"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def save_point_cloud(pcd, path):
    """保存点云为 .ply 文件，保存失败时抛出异常。"""
    success = o3d.io.write_point_cloud(path, pcd)
    if not success:
        raise IOError(f"点云保存失败: {path}")


def visualize_point_cloud(pcd):
    """显示生成的原始点云和坐标轴。"""
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    o3d.visualization.draw_geometries(
        [pcd, axis],
        window_name="原始点云",
    )


def main():
    """读取深度图，转换并保存未经后处理的原始点云。"""
    print("正在读取深度图...")
    depth = load_depth(DEPTH_PATH)
    print("深度图读取成功")
    print("depth shape:", depth.shape)
    print("depth dtype:", depth.dtype)

    valid_depth = depth[depth > 0]
    if valid_depth.size > 0:
        print("最小有效深度:", valid_depth.min(), "mm")
        print("最大有效深度:", valid_depth.max(), "mm")
    else:
        print("警告: 深度图中没有有效深度值")

    height, width = depth.shape
    cx = width / 2.0
    cy = height / 2.0

    print("正在生成原始点云...")
    points = depth_to_point_cloud(depth, FX, FY, cx, cy, MIN_Z, MAX_Z)
    print("原始点云生成成功")
    print("原始点云点数:", len(points))

    pcd = create_point_cloud(points)

    print("正在保存原始点云...")
    save_point_cloud(pcd, OUTPUT_PLY)
    print(f"原始点云已保存: {OUTPUT_PLY}")

    visualize_point_cloud(pcd)


if __name__ == "__main__":
    main()
