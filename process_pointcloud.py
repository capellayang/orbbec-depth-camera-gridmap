import copy
import os

import numpy as np
import open3d as o3d


INPUT_PLY = "astra_pointcloud_raw.ply"
OUTPUT_PLY = "astra_pointcloud_processed.ply"

VOXEL_SIZE = 0.02

STATISTICAL_NB_NEIGHBORS = 30
STATISTICAL_STD_RATIO = 1.2

RADIUS_NB_POINTS = 20
RADIUS = 0.04

MIN_Z = 0.6
MAX_Z = 3.0


def load_point_cloud(path):
    """读取 .ply 点云文件。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到输入点云文件: {path}")

    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"点云文件为空或读取失败: {path}")

    return pcd


def print_point_cloud_info(pcd, name):
    """打印点云点数量。"""
    print(f"{name}点数: {len(pcd.points)}")


def voxel_downsample(pcd, voxel_size):
    """进行体素下采样。"""
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def remove_statistical_outliers(pcd, nb_neighbors, std_ratio):
    """进行统计离群点滤波。"""
    filtered_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return filtered_pcd


def remove_radius_outliers(pcd, nb_points, radius):
    """进行半径离群点滤波。"""
    filtered_pcd, _ = pcd.remove_radius_outlier(
        nb_points=nb_points,
        radius=radius,
    )
    return filtered_pcd


def crop_by_distance(pcd, min_z, max_z):
    """根据 Z 方向距离范围裁剪点云。"""
    points = np.asarray(pcd.points)
    if points.size == 0:
        return pcd

    z_values = points[:, 2]
    valid_indices = np.where((z_values >= min_z) & (z_values <= max_z))[0]
    return pcd.select_by_index(valid_indices)


def visualize_point_clouds(original_pcd, processed_pcd):
    """将原始点云和处理后点云左右分开展示。"""
    original_vis = copy.deepcopy(original_pcd)
    processed_vis = copy.deepcopy(processed_pcd)

    original_vis.paint_uniform_color([0.6, 0.6, 0.6])
    processed_vis.paint_uniform_color([0.1, 0.45, 1.0])

    original_vis.translate([-3.0, 0.0, 0.0])
    processed_vis.translate([3.0, 0.0, 0.0])

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    o3d.visualization.draw_geometries(
        [original_vis, processed_vis, axis],
        window_name="原始点云 / 处理后点云",
    )


def save_point_cloud(pcd, path):
    """保存点云为 .ply 文件，保存失败时抛出异常。"""
    success = o3d.io.write_point_cloud(path, pcd)
    if not success:
        raise IOError(f"点云保存失败: {path}")


def main():
    """执行离线点云处理流程。"""
    print("正在读取点云...")
    original_pcd = load_point_cloud(INPUT_PLY)
    print_point_cloud_info(original_pcd, "原始点云")

    print("正在进行体素下采样...")
    processed_pcd = voxel_downsample(original_pcd, VOXEL_SIZE)
    print_point_cloud_info(processed_pcd, "下采样后")

    print("正在进行统计滤波...")
    processed_pcd = remove_statistical_outliers(
        processed_pcd,
        STATISTICAL_NB_NEIGHBORS,
        STATISTICAL_STD_RATIO,
    )
    print_point_cloud_info(processed_pcd, "统计滤波后")

    print("正在进行半径滤波...")
    processed_pcd = remove_radius_outliers(
        processed_pcd,
        RADIUS_NB_POINTS,
        RADIUS,
    )
    print_point_cloud_info(processed_pcd, "半径滤波后")

    print("正在根据 Z 距离裁剪...")
    processed_pcd = crop_by_distance(processed_pcd, MIN_Z, MAX_Z)
    print_point_cloud_info(processed_pcd, "裁剪后")

    print("正在保存处理后的点云...")
    save_point_cloud(processed_pcd, OUTPUT_PLY)
    print(f"处理后点云已保存: {OUTPUT_PLY}")

    visualize_point_clouds(original_pcd, processed_pcd)


if __name__ == "__main__":
    main()
