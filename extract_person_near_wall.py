import copy
import math
import os

import numpy as np
import open3d as o3d


INPUT_PLY = "astra_pointcloud_raw.ply"
WALL_PLY = "wall.ply"
NON_WALL_PLY = "non_wall.ply"
FOREGROUND_PLY = "foreground.ply"
PERSON_MAIN_CLUSTER_PLY = "person_main_cluster.ply"
PERSON_PLY = "person.ply"
OTHER_FOREGROUND_PLY = "other_foreground.ply"
CLUSTER_INFO_TXT = "candidate_clusters_info.txt"

VOXEL_SIZE = 0.005

PLANE_DISTANCE_THRESHOLD = 0.02
PLANE_RANSAC_N = 3
PLANE_NUM_ITERATIONS = 1000

FOREGROUND_MIN_WALL_DISTANCE = 0.06
FOREGROUND_MAX_WALL_DISTANCE = 0.80

DBSCAN_EPS = 0.05
DBSCAN_MIN_POINTS = 20

HUMAN_MIN_POINTS = 50
HUMAN_MIN_HEIGHT = 0.6
HUMAN_MAX_HEIGHT = 2.2
HUMAN_MIN_WIDTH = 0.1
HUMAN_MAX_WIDTH = 1.2
HUMAN_MIN_THICKNESS = 0.03
HUMAN_MAX_THICKNESS = 1.0
HUMAN_MIN_MEAN_WALL_DISTANCE = 0.08
HUMAN_MIN_MAX_WALL_DISTANCE = 0.12
TARGET_HUMAN_HEIGHT = 1.5
GROUND_MARGIN = 0.35

MERGE_BBOX_DISTANCE = 0.25
MERGE_MIN_Y_OVERLAP_RATIO = 0.60
MERGE_Y_OVERLAP_MAX_XZ_DISTANCE = 0.30

ALLOW_FALLBACK_TO_LARGEST = False


def load_point_cloud(path):
    """读取原始点云文件。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到输入点云文件: {path}")

    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"点云文件为空或读取失败: {path}")

    return pcd


def downsample_point_cloud(pcd, voxel_size):
    """对点云进行轻度体素下采样。"""
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def compute_point_to_plane_distance(points, plane_model):
    """计算点到 RANSAC 墙面平面的距离。"""
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise ValueError("plane_model 法向量长度为 0，无法计算点到平面距离")

    return np.abs(points @ normal + d) / normal_norm


def segment_wall_plane(pcd, distance_threshold, ransac_n, num_iterations):
    """使用 RANSAC 平面拟合分割墙面点和非墙面点。"""
    if len(pcd.points) < ransac_n:
        raise ValueError("点云点数太少，无法进行平面分割")

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    wall_pcd = pcd.select_by_index(inliers)
    non_wall_pcd = pcd.select_by_index(inliers, invert=True)
    return plane_model, wall_pcd, non_wall_pcd


def extract_foreground_points(
    non_wall_pcd,
    plane_model,
    min_wall_distance,
    max_wall_distance,
):
    """从非墙面点云中剔除贴墙残余点，保留离墙合理的前景点。"""
    points = np.asarray(non_wall_pcd.points)
    if points.size == 0:
        empty = o3d.geometry.PointCloud()
        return empty, empty

    distances = compute_point_to_plane_distance(points, plane_model)
    foreground_mask = (
        (distances > min_wall_distance)
        & (distances < max_wall_distance)
    )
    foreground_indices = np.where(foreground_mask)[0]
    residual_indices = np.where(~foreground_mask)[0]

    foreground_pcd = non_wall_pcd.select_by_index(foreground_indices)
    residual_pcd = non_wall_pcd.select_by_index(residual_indices)
    return foreground_pcd, residual_pcd


def cluster_non_wall_points(pcd, eps, min_points):
    """对前景点云进行 DBSCAN 聚类。"""
    if pcd.is_empty():
        return np.array([], dtype=int), {}

    labels = np.array(
        pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=True,
        )
    )

    cluster_counts = {}
    for label in sorted(set(labels)):
        if label == -1:
            continue
        cluster_counts[int(label)] = int(np.sum(labels == label))

    return labels, cluster_counts


def compute_scene_min_y(foreground_pcd, labels):
    """统计所有前景有效聚类中的最低 y 值。"""
    points = np.asarray(foreground_pcd.points)
    if points.size == 0:
        return None

    clustered_indices = np.where(labels != -1)[0]
    if clustered_indices.size > 0:
        return float(np.min(points[clustered_indices, 1]))

    return float(np.min(points[:, 1]))


def is_human_candidate(cluster_info, scene_min_y):
    """根据人体尺寸、离墙距离和低位约束判断候选。"""
    count = cluster_info["point_count"]
    width, height, thickness = cluster_info["extent"]

    if scene_min_y is None:
        return False

    return (
        count >= HUMAN_MIN_POINTS
        and HUMAN_MIN_HEIGHT <= height <= HUMAN_MAX_HEIGHT
        and HUMAN_MIN_WIDTH <= width <= HUMAN_MAX_WIDTH
        and HUMAN_MIN_THICKNESS <= thickness <= HUMAN_MAX_THICKNESS
        and cluster_info["mean_wall_distance"] >= HUMAN_MIN_MEAN_WALL_DISTANCE
        and cluster_info["max_wall_distance"] >= HUMAN_MIN_MAX_WALL_DISTANCE
        and cluster_info["min_y"] <= scene_min_y + GROUND_MARGIN
    )


def score_human_candidate(cluster_info, scene_min_y):
    """对人体候选聚类打分，综合点数、高度、底部位置、离墙距离和尺寸。"""
    count = cluster_info["point_count"]
    width, height, thickness = cluster_info["extent"]
    mean_wall_distance = cluster_info["mean_wall_distance"]
    min_y = cluster_info["min_y"]

    point_score = math.log1p(count)
    height_score = max(
        0.0,
        1.0 - abs(height - TARGET_HUMAN_HEIGHT) / TARGET_HUMAN_HEIGHT,
    )
    ground_score = max(
        0.0,
        1.0 - abs(min_y - scene_min_y) / GROUND_MARGIN,
    )
    wall_distance_score = min(mean_wall_distance / 0.35, 1.0)
    too_close_penalty = max(0.0, 0.15 - mean_wall_distance) * 8.0
    size_penalty = 0.8 * width + 1.2 * thickness

    return (
        point_score
        + 5.0 * height_score
        + 3.0 * ground_score
        + 2.0 * wall_distance_score
        - too_close_penalty
        - size_penalty
    )


def analyze_clusters(foreground_pcd, labels, cluster_counts, plane_model):
    """分析每个前景聚类的尺寸、低位、离墙距离、候选状态和评分。"""
    cluster_infos = []
    scene_min_y = compute_scene_min_y(foreground_pcd, labels)
    all_points = np.asarray(foreground_pcd.points)

    for label, count in sorted(cluster_counts.items()):
        indices = np.where(labels == label)[0]
        cluster_points = all_points[indices]
        cluster_pcd = foreground_pcd.select_by_index(indices)
        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        extent = bbox.get_extent()
        center = bbox.get_center()
        distances = compute_point_to_plane_distance(cluster_points, plane_model)

        cluster_info = {
            "label": int(label),
            "point_count": int(count),
            "indices": indices,
            "bbox": bbox,
            "min_bound": min_bound,
            "max_bound": max_bound,
            "extent": extent,
            "center": center,
            "min_y": float(min_bound[1]),
            "max_y": float(max_bound[1]),
            "mean_wall_distance": float(np.mean(distances)),
            "max_wall_distance": float(np.max(distances)),
            "is_candidate": False,
            "score": None,
        }
        cluster_info["is_candidate"] = is_human_candidate(
            cluster_info,
            scene_min_y,
        )
        if cluster_info["is_candidate"]:
            cluster_info["score"] = score_human_candidate(
                cluster_info,
                scene_min_y,
            )

        cluster_infos.append(cluster_info)

    return cluster_infos, scene_min_y


def get_bbox_distance(bbox_a, bbox_b):
    """计算两个轴对齐包围盒之间的最短距离，重叠时为 0。"""
    min_a = bbox_a.get_min_bound()
    max_a = bbox_a.get_max_bound()
    min_b = bbox_b.get_min_bound()
    max_b = bbox_b.get_max_bound()

    gaps = []
    for axis in range(3):
        if max_a[axis] < min_b[axis]:
            gaps.append(min_b[axis] - max_a[axis])
        elif max_b[axis] < min_a[axis]:
            gaps.append(min_a[axis] - max_b[axis])
        else:
            gaps.append(0.0)

    return float(np.linalg.norm(gaps))


def get_y_overlap_ratio(bbox_a, bbox_b):
    """计算两个包围盒在 y 方向上的重叠比例。"""
    min_a = bbox_a.get_min_bound()
    max_a = bbox_a.get_max_bound()
    min_b = bbox_b.get_min_bound()
    max_b = bbox_b.get_max_bound()

    overlap = max(0.0, min(max_a[1], max_b[1]) - max(min_a[1], min_b[1]))
    height_a = max_a[1] - min_a[1]
    height_b = max_b[1] - min_b[1]
    min_height = max(min(height_a, height_b), 1e-6)
    return float(overlap / min_height)


def get_bbox_xz_distance(bbox_a, bbox_b):
    """计算两个包围盒在 x-z 平面上的间距，重叠时为 0。"""
    min_a = bbox_a.get_min_bound()
    max_a = bbox_a.get_max_bound()
    min_b = bbox_b.get_min_bound()
    max_b = bbox_b.get_max_bound()

    gaps = []
    for axis in (0, 2):
        if max_a[axis] < min_b[axis]:
            gaps.append(min_b[axis] - max_a[axis])
        elif max_b[axis] < min_a[axis]:
            gaps.append(min_a[axis] - max_b[axis])
        else:
            gaps.append(0.0)

    return float(np.linalg.norm(gaps))


def select_person_cluster(foreground_pcd, labels, cluster_infos):
    """选择评分最高的人体候选；默认不回退到最大聚类。"""
    if not cluster_infos:
        return None, None, None, False

    candidates = [info for info in cluster_infos if info["is_candidate"]]
    if candidates:
        selected_info = max(candidates, key=lambda item: item["score"])
        person_indices = selected_info["indices"]
        person_pcd = foreground_pcd.select_by_index(person_indices)
        return person_pcd, selected_info, candidates, False

    print("没有找到符合人体尺寸和离墙条件的聚类")
    print(
        "请尝试调整参数: FOREGROUND_MIN_WALL_DISTANCE, "
        "DBSCAN_EPS, HUMAN_MIN_HEIGHT, GROUND_MARGIN"
    )

    if not ALLOW_FALLBACK_TO_LARGEST:
        print("ALLOW_FALLBACK_TO_LARGEST = False，已禁止回退到最大聚类")
        return None, None, [], False

    selected_info = max(cluster_infos, key=lambda item: item["point_count"])
    person_indices = selected_info["indices"]
    person_pcd = foreground_pcd.select_by_index(person_indices)
    print(f"fallback 选择聚类 {selected_info['label']} 作为人体点云")
    return person_pcd, selected_info, [], True


def merge_nearby_person_clusters(foreground_pcd, labels, cluster_infos, selected_info):
    """将与主人体聚类相邻或 y 范围明显重叠的聚类合并为人体。"""
    if selected_info is None:
        return None, [], np.array([], dtype=int)

    selected_label = selected_info["label"]
    selected_bbox = selected_info["bbox"]
    merged_labels = [selected_label]
    person_indices = selected_info["indices"]

    for info in cluster_infos:
        label = info["label"]
        if label == selected_label:
            continue

        bbox_distance = get_bbox_distance(selected_bbox, info["bbox"])
        y_overlap_ratio = get_y_overlap_ratio(selected_bbox, info["bbox"])
        xz_distance = get_bbox_xz_distance(selected_bbox, info["bbox"])

        if (
            bbox_distance < MERGE_BBOX_DISTANCE
            or (
                y_overlap_ratio >= MERGE_MIN_Y_OVERLAP_RATIO
                and xz_distance < MERGE_Y_OVERLAP_MAX_XZ_DISTANCE
            )
        ):
            test_indices = np.unique(np.concatenate([person_indices, info["indices"]]))
            test_pcd = foreground_pcd.select_by_index(test_indices)
            test_extent = test_pcd.get_axis_aligned_bounding_box().get_extent()
            if (
                test_extent[0] <= HUMAN_MAX_WIDTH
                and test_extent[1] <= HUMAN_MAX_HEIGHT
                and test_extent[2] <= HUMAN_MAX_THICKNESS
            ):
                merged_labels.append(label)
                person_indices = test_indices
                print(
                    f"合并邻近聚类 {label}: "
                    f"bbox_distance={bbox_distance:.4f}, "
                    f"y_overlap_ratio={y_overlap_ratio:.4f}, "
                    f"xz_distance={xz_distance:.4f}"
                )
            else:
                print(
                    f"跳过邻近聚类 {label}: 合并后尺寸过大 "
                    f"width={test_extent[0]:.4f}, "
                    f"height={test_extent[1]:.4f}, "
                    f"thickness={test_extent[2]:.4f}"
                )

    person_pcd = foreground_pcd.select_by_index(person_indices)
    return person_pcd, merged_labels, person_indices


def create_other_foreground_pcd(foreground_pcd, person_indices):
    """从前景点云中去掉最终人体点，得到其他前景点。"""
    if person_indices.size == 0:
        return copy.deepcopy(foreground_pcd)

    all_indices = np.arange(len(foreground_pcd.points))
    person_mask = np.zeros(len(foreground_pcd.points), dtype=bool)
    person_mask[person_indices] = True
    other_indices = all_indices[~person_mask]
    return foreground_pcd.select_by_index(other_indices)


def compute_bounding_box(person_pcd):
    """计算人体点云的轴对齐包围盒。"""
    if person_pcd is None or person_pcd.is_empty():
        return None

    bbox = person_pcd.get_axis_aligned_bounding_box()
    bbox.color = [0.0, 1.0, 0.0]
    return bbox


def visualize_result(
    wall_pcd,
    residual_pcd,
    person_pcd,
    other_foreground_pcd,
    bbox,
):
    """可视化墙面、贴墙残余、最终人体、其他前景点和包围盒。"""
    wall_vis = copy.deepcopy(wall_pcd)
    residual_vis = copy.deepcopy(residual_pcd)
    person_vis = copy.deepcopy(person_pcd) if person_pcd is not None else None
    other_vis = copy.deepcopy(other_foreground_pcd)

    wall_vis.paint_uniform_color([0.6, 0.6, 0.6])
    residual_vis.paint_uniform_color([0.82, 0.82, 0.82])
    other_vis.paint_uniform_color([1.0, 0.1, 0.1])

    geometries = [wall_vis]
    if not residual_vis.is_empty():
        geometries.append(residual_vis)
    if not other_vis.is_empty():
        geometries.append(other_vis)
    if person_vis is not None and not person_vis.is_empty():
        person_vis.paint_uniform_color([0.1, 0.45, 1.0])
        geometries.append(person_vis)
    if bbox is not None:
        geometries.append(bbox)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    geometries.append(axis)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Wall / Person Segmentation",
    )


def save_point_cloud(pcd, path):
    """保存点云为 .ply 文件，保存失败时抛出异常。"""
    success = o3d.io.write_point_cloud(path, pcd)
    if not success:
        raise IOError(f"点云保存失败: {path}")


def print_point_count(name, pcd):
    """打印点云点数量。"""
    print(f"{name}点数: {len(pcd.points)}")


def format_vector(vector):
    """格式化三维向量，便于打印和写入文本。"""
    return "[" + ", ".join(f"{value:.6f}" for value in vector) + "]"


def print_cluster_analysis(cluster_infos, scene_min_y):
    """打印每个聚类的尺寸、低位和离墙距离分析。"""
    if scene_min_y is None:
        print("scene_min_y: 无有效前景聚类")
    else:
        print(f"scene_min_y: {scene_min_y:.6f}")

    if not cluster_infos:
        print("没有找到有效聚类")
        return

    print("聚类分析结果:")
    for info in cluster_infos:
        extent = info["extent"]
        score_text = (
            f"{info['score']:.6f}" if info["score"] is not None else "不符合候选条件"
        )
        print(f"聚类 label: {info['label']}")
        print(f"  点数量: {info['point_count']}")
        print(f"  bbox 最小坐标: {format_vector(info['min_bound'])}")
        print(f"  bbox 最大坐标: {format_vector(info['max_bound'])}")
        print(f"  宽度 extent[0]: {extent[0]:.6f}")
        print(f"  高度 extent[1]: {extent[1]:.6f}")
        print(f"  厚度 extent[2]: {extent[2]:.6f}")
        print(f"  中心坐标: {format_vector(info['center'])}")
        print(f"  min_y: {info['min_y']:.6f}")
        print(f"  max_y: {info['max_y']:.6f}")
        print(f"  mean_wall_distance: {info['mean_wall_distance']:.6f}")
        print(f"  max_wall_distance: {info['max_wall_distance']:.6f}")
        print(f"  是否人体候选: {info['is_candidate']}")
        print(f"  评分: {score_text}")


def save_cluster_analysis(cluster_infos, scene_min_y, path):
    """将聚类分析结果保存到文本文件。"""
    lines = []
    if scene_min_y is None:
        lines.append("scene_min_y: 无有效前景聚类")
    else:
        lines.append(f"scene_min_y: {scene_min_y:.6f}")
    lines.append("")

    if not cluster_infos:
        lines.append("没有找到有效聚类")
    else:
        for info in cluster_infos:
            extent = info["extent"]
            score_text = (
                f"{info['score']:.6f}"
                if info["score"] is not None
                else "不符合候选条件"
            )
            lines.extend(
                [
                    f"聚类 label: {info['label']}",
                    f"点数量: {info['point_count']}",
                    f"bbox 最小坐标: {format_vector(info['min_bound'])}",
                    f"bbox 最大坐标: {format_vector(info['max_bound'])}",
                    f"宽度 extent[0]: {extent[0]:.6f}",
                    f"高度 extent[1]: {extent[1]:.6f}",
                    f"厚度 extent[2]: {extent[2]:.6f}",
                    f"中心坐标: {format_vector(info['center'])}",
                    f"min_y: {info['min_y']:.6f}",
                    f"max_y: {info['max_y']:.6f}",
                    f"mean_wall_distance: {info['mean_wall_distance']:.6f}",
                    f"max_wall_distance: {info['max_wall_distance']:.6f}",
                    f"是否人体候选: {info['is_candidate']}",
                    f"评分: {score_text}",
                    "",
                ]
            )

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def print_bounding_box_info(person_pcd, bbox):
    """打印人体包围盒信息。"""
    if bbox is None:
        print("未计算人体包围盒，因为没有有效人体点云")
        return

    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    extent = bbox.get_extent()
    center = bbox.get_center()

    print("人体点数量:", len(person_pcd.points))
    print("包围盒最小坐标:", min_bound)
    print("包围盒最大坐标:", max_bound)
    print("人体宽度:", extent[0])
    print("人体高度:", extent[1])
    print("人体厚度:", extent[2])
    print("人体中心坐标:", center)


def main():
    """执行墙面分割、前景提取、人体候选评分、聚类合并和可视化流程。"""
    print("正在读取原始点云...")
    raw_pcd = load_point_cloud(INPUT_PLY)
    print_point_count("原始点云", raw_pcd)

    print("正在进行轻度体素下采样...")
    downsampled_pcd = downsample_point_cloud(raw_pcd, VOXEL_SIZE)
    print_point_count("下采样后点云", downsampled_pcd)

    print("正在使用 RANSAC 分割墙面...")
    plane_model, wall_pcd, non_wall_pcd = segment_wall_plane(
        downsampled_pcd,
        PLANE_DISTANCE_THRESHOLD,
        PLANE_RANSAC_N,
        PLANE_NUM_ITERATIONS,
    )
    print("墙面平面模型: ax + by + cz + d = 0")
    print("平面参数:", plane_model)
    print_point_count("墙面点云", wall_pcd)
    print_point_count("非墙面点云", non_wall_pcd)

    print("正在提取离墙前景点云...")
    foreground_pcd, residual_pcd = extract_foreground_points(
        non_wall_pcd,
        plane_model,
        FOREGROUND_MIN_WALL_DISTANCE,
        FOREGROUND_MAX_WALL_DISTANCE,
    )
    print_point_count("前景点云", foreground_pcd)
    print_point_count("贴墙残余点云", residual_pcd)

    print("正在保存墙面、非墙面和前景点云...")
    save_point_cloud(wall_pcd, WALL_PLY)
    save_point_cloud(non_wall_pcd, NON_WALL_PLY)
    save_point_cloud(foreground_pcd, FOREGROUND_PLY)
    print(f"墙面点云已保存: {WALL_PLY}")
    print(f"非墙面点云已保存: {NON_WALL_PLY}")
    print(f"前景点云已保存: {FOREGROUND_PLY}")

    print("正在对前景点云进行 DBSCAN 聚类...")
    labels, cluster_counts = cluster_non_wall_points(
        foreground_pcd,
        DBSCAN_EPS,
        DBSCAN_MIN_POINTS,
    )

    print("正在分析人体候选聚类...")
    cluster_infos, scene_min_y = analyze_clusters(
        foreground_pcd,
        labels,
        cluster_counts,
        plane_model,
    )
    print_cluster_analysis(cluster_infos, scene_min_y)
    save_cluster_analysis(cluster_infos, scene_min_y, CLUSTER_INFO_TXT)
    print(f"聚类分析结果已保存: {CLUSTER_INFO_TXT}")

    main_cluster_pcd, selected_info, candidates, fallback_used = select_person_cluster(
        foreground_pcd,
        labels,
        cluster_infos,
    )

    if main_cluster_pcd is None:
        print("没有有效人体候选，已跳过 person_main_cluster.ply 和 person.ply 保存")
        person_pcd = None
        other_foreground_pcd = copy.deepcopy(foreground_pcd)
        save_point_cloud(other_foreground_pcd, OTHER_FOREGROUND_PLY)
        print(f"其他前景点云已保存: {OTHER_FOREGROUND_PLY}")
        final_bbox = None
    else:
        if fallback_used:
            print(f"fallback 主人体聚类: {selected_info['label']}")
        else:
            print(
                f"评分最高的主人体聚类: {selected_info['label']}, "
                f"评分: {selected_info['score']:.6f}"
            )

        print_point_count("主人体聚类点云", main_cluster_pcd)
        save_point_cloud(main_cluster_pcd, PERSON_MAIN_CLUSTER_PLY)
        print(f"主人体聚类已保存: {PERSON_MAIN_CLUSTER_PLY}")

        print("正在合并邻近人体聚类...")
        person_pcd, merged_labels, person_indices = merge_nearby_person_clusters(
            foreground_pcd,
            labels,
            cluster_infos,
            selected_info,
        )
        print("最终人体聚类 labels:", merged_labels)
        print_point_count("合并后人体点云", person_pcd)

        other_foreground_pcd = create_other_foreground_pcd(
            foreground_pcd,
            person_indices,
        )
        print_point_count("其他前景点云", other_foreground_pcd)

        save_point_cloud(person_pcd, PERSON_PLY)
        save_point_cloud(other_foreground_pcd, OTHER_FOREGROUND_PLY)
        print(f"人体点云已保存: {PERSON_PLY}")
        print(f"其他前景点云已保存: {OTHER_FOREGROUND_PLY}")

        final_bbox = compute_bounding_box(person_pcd)
        print_bounding_box_info(person_pcd, final_bbox)

    visualize_result(
        wall_pcd,
        residual_pcd,
        person_pcd,
        other_foreground_pcd,
        final_bbox,
    )


if __name__ == "__main__":
    main()
