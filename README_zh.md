# Orbbec Astra Pro Plus 深度相机建图与 SLAM 学习项目

## 项目简介

本项目基于 Orbbec Astra Pro Plus 深度相机，使用 Python、OpenNI2、OpenCV、NumPy 和 Open3D，从深度图采集开始，逐步学习深度图显示、像素到三维坐标转换、点云生成、点云预处理、局部建图和占据栅格地图。

项目早期曾做过墙面、人形、非墙面、聚类等点云提取实验。当前主线已经从“人体点云提取实验”转向“移动机器人避障与建图”，目标是为后续学习 ROS2、Nav2、RTAB-Map 和 SLAM 算法打基础。

当前已经完成到“单帧点云生成 2D 局部占据栅格地图”的初级阶段。下一步重点是高度过滤、地面去除和更标准的 occupancy grid。

## 当前项目状态

| 阶段 | 状态 | 对应文件 | 说明 |
|---|---|---|---|
| Orbbec 深度相机接入 | 已完成 | `orbbec_openni_utils.py`, `test_depth_minimal.py` | 已能通过 OpenNI2 打开 Astra Pro Plus 深度流 |
| OpenNI 深度流读取 | 已完成但需继续稳定 | `orbbec_openni_utils.py`, `save_depth_frame.py` | 已统一 OpenNI 路径和资源释放逻辑，仍可能遇到底层 DLL 崩溃 |
| 深度图显示与保存 | 已完成 | `save_depth_frame.py`, `depth_data/` | 可保存 `.npy` 原始深度数组和显示图 |
| 相机内参估算 | 已完成初版 | `click_depth_point.py` | 当前使用基于 OpenNI FOV 估算的内参 |
| 深度像素转 X/Y/Z | 已完成 | `click_depth_point.py`, `depth_to_pointcloud.py` | 已实现点击像素输出 3D 坐标 |
| 深度图生成点云 | 已完成 | `depth_to_pointcloud.py` | 可从 `.npy` 深度图生成 `.ply` 点云 |
| 点云保存为 `.ply` | 已完成 | `depth_to_pointcloud.py`, `process_pointcloud.py` | 已生成原始和处理后的点云 |
| 点云预处理 | 已完成初版 | `process_pointcloud.py` | 包含体素下采样、统计滤波、半径滤波和 Z 裁剪 |
| 点云投影到 X-Z 平面 | 已完成初版 | `pointcloud_to_occupancy_grid.py` | 将单帧点云投影到局部 2D 地图 |
| 单帧 2D 局部占据栅格地图 | 已完成初版 | `pointcloud_to_occupancy_grid.py`, `occupancy_grid.png` | 已按点数阈值生成 occupied/unknown 简化栅格 |
| 高度过滤、地面去除、自由空间判断 | 尚未开始 | 计划中 | 下一阶段重点 |
| 多帧地图融合 | 尚未开始 | 计划中 | 需要位姿或里程计 |
| 位姿估计 | 尚未开始 | 计划中 | 后续 SLAM 前置内容 |
| SLAM | 尚未开始 | 计划中 | 当前还不是完整 SLAM |
| ROS2 / Nav2 接入 | 尚未开始 | 计划中 | 后续迁移方向 |

## 环境依赖

- Windows
- Conda 环境：`orbbec`
- Python
- OpenNI2 / `openni`
- OpenCV / `opencv-python`
- NumPy
- Open3D
- 代码中还用到 Python 标准库：`os`, `time`, `traceback`, `json`, `datetime`, `copy`, `math`, `dataclasses`

统一 OpenNI 路径：

```text
F:\Orbbec\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\tools\NiViewer
```

注意：不要混用 `samples/bin`、`sdk/libs`、`Win32-Release` 等其他 OpenNI2.dll 路径。

## 相机参数

当前深度流分辨率：

```text
width = 640
height = 480
```

当前估算内参：

```text
fx = 574.9614356132867
fy = 574.9614061048316
cx = 320.0
cy = 240.0
```

这些参数来自 OpenNI FOV 估算，适合学习和初步实验。后续迁移到 ROS2 时，应优先使用 `/camera/depth/camera_info` 中发布的相机内参。

## 核心公式

深度图像素 `(u, v)` 到相机坐标系三维点：

```text
Z = depth[v, u] / 1000.0
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

其中：

- `depth raw` 单位为 `mm`
- `X/Y/Z` 单位为 `m`
- 当前约定中 `X` 为左右方向，`Y` 为上下方向，`Z` 为前方深度方向

## 文件说明

### 当前主线脚本

| 文件 | 作用 | 阶段 | 建议 |
|---|---|---|---|
| `orbbec_openni_utils.py` | 统一 OpenNI 路径、初始化、读帧 `.copy()` 和安全释放 | 深度图采集 / 稳定性 | 保留 |
| `test_depth_minimal.py` | 最小深度流测试，显示中心点深度 | 深度图采集 / 显示 | 保留 |
| `save_depth_frame.py` | 保存原始深度数组和显示图 | 深度图采集 / 保存 | 保留 |
| `click_depth_point.py` | 点击深度图输出 `u/v/depth/X/Y/Z` | 相机内参 / 三维坐标转换 | 保留 |
| `depth_click.py` | 兼容旧文件名，入口转到 `click_depth_point.py` | 兼容入口 | 可保留或后续归档 |
| `depth_to_pointcloud.py` | 从 `.npy` 深度图生成 Open3D 点云并保存 `.ply` | 深度图转点云 | 保留 |
| `view_ply.py` | 用 Open3D 查看 `.ply` 点云 | 点云查看 | 保留 |
| `process_pointcloud.py` | 点云下采样、离群点滤波、裁剪并保存处理后点云 | 点云预处理 | 保留 |
| `pointcloud_to_occupancy_grid.py` | 将单帧 `.ply` 点云投影为 X-Z 平面 2D 局部占据栅格地图 | 当前建图主线 | 保留 |
| `depth_obstacle_detector.py` | 基于深度图 ROI 的简单三分区避障检测 | 障碍物检测 | 保留，后续可和建图主线整合 |

### 旧方向实验脚本

| 文件 | 作用 | 阶段 | 建议 |
|---|---|---|---|
| `extract_person_near_wall.py` | RANSAC 分割墙面，提取非墙面/前景/疑似人体点云，输出聚类信息 | 旧方向：人体/墙面提取实验 | 暂时归档，不继续扩展 |
| `print.py` | 兼容旧测试入口，当前转到 `click_depth_point.py` | 临时/兼容 | 建议后续重命名或删除，避免和 Python 内置 `print` 混淆 |

### 点云与地图数据

| 文件 | 来源和用途 | 阶段 | 建议 |
|---|---|---|---|
| `astra_pointcloud_raw.ply` | 由 `depth_to_pointcloud.py` 从深度图生成的原始点云 | 深度图转点云 | 保留 |
| `astra_pointcloud.ply` | 早期保存的点云文件 | 点云数据 | 保留作为样例 |
| `astra_pointcloud_processed.ply` | 由 `process_pointcloud.py` 输出的处理后点云 | 点云预处理 | 保留 |
| `filtered_for_grid.ply` | 生成占据栅格前的过滤点云 | 当前建图主线 | 保留 |
| `occupancy_grid.png` | `pointcloud_to_occupancy_grid.py` 输出的 2D 局部占据栅格图 | 当前建图主线 | 保留 |
| `wall.ply` | 墙面分割结果 | 旧方向实验 | 归档 |
| `non_wall.ply` | 非墙面点云结果 | 旧方向实验 | 归档 |
| `foreground.ply` | 墙前景点云结果 | 旧方向实验 | 归档 |
| `person.ply` | 疑似人体点云结果 | 旧方向实验 | 归档，不再作为主线输入 |
| `person_main_cluster.ply` | 人体主聚类结果 | 旧方向实验 | 归档 |
| `other_foreground.ply` | 其他前景点云结果 | 旧方向实验 | 归档 |

注意：`pointcloud_to_occupancy_grid.py` 当前配置可能仍指向旧实验文件，例如 `person.ply`。运行当前建图主线时，建议将 `INPUT_PLY` 设置为 `astra_pointcloud_raw.ply` 或 `astra_pointcloud_processed.ply`。

### 调试和输出文件

| 文件/目录 | 作用 | 建议 |
|---|---|---|
| `depth_data/depth_0001.npy` 到 `depth_0007.npy` | 原始 uint16 深度帧，单位 mm | 保留作为样例数据 |
| `depth_data/depth_show_0001.png` 到 `depth_show_0007.png` | 深度图显示图，仅用于观察 | 保留或后续归档 |
| `candidate_clusters_info.txt` | 人体/墙面提取实验的聚类分析输出 | 归档 |
| `obstacle_result.json` | 避障检测结果输出，目前为空文件 | 保留，后续运行避障脚本会更新 |
| `__pycache__/` | Python 字节码缓存 | 可删除，不影响项目 |
| `.git/` | Git 仓库数据 | 保留 |
| `.agents/` | Codex/代理工作目录 | 保留或按工具需要处理 |

## 推荐运行顺序

建议使用 `orbbec` Conda 环境运行。

1. 测试深度相机：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" test_depth_minimal.py
```

2. 保存一帧深度图：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" save_depth_frame.py
```

3. 点击深度图查看 `u/v/depth/X/Y/Z`：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" click_depth_point.py
```

4. 从深度图生成点云：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" depth_to_pointcloud.py
```

5. 查看点云：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" view_ply.py
```

6. 点云预处理：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" process_pointcloud.py
```

7. 生成局部占据栅格地图：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" pointcloud_to_occupancy_grid.py
```

## 当前建图流程图

```text
Orbbec Astra Pro Plus
        |
        v
OpenNI2 深度流
        |
        v
uint16 深度图 depth[v, u]，单位 mm
        |
        v
相机内参 fx, fy, cx, cy
        |
        v
3D 点云 X/Y/Z，单位 m
        |
        v
点云过滤：NaN、距离范围、体素下采样、离群点滤波
        |
        v
X-Z 平面投影
        |
        v
2D 局部占据栅格地图
        |
        v
后续：高度过滤 / 地面去除 / free-unknown-occupied / 多帧融合 / SLAM
```

## 已知问题

- OpenNI 退出或创建深度流时，偶尔可能出现 `python.exe - 应用程序错误`，这是底层 OpenNI2.dll / Orbbec 驱动 / OpenCV 窗口交互导致的 native 崩溃。
- 不应混用多个 OpenNI2.dll 路径。
- `depth_display.png` 或其他 uint8 显示图不能当作真实深度图使用；真实深度应使用 `.npy` 或 `depth_raw.png`。
- 当前 2D 栅格图尚未加入 Y 方向高度过滤和地面去除。
- 当前 occupancy grid 只根据每格点数判断 occupied，还没有标准的 free / unknown 建模。
- 当前还不是完整 SLAM，只是单帧局部建图。

## 下一步计划

- 继续修复和规避 OpenNI 退出崩溃问题。
- 增加 Y 方向高度过滤，只保留可能属于障碍物高度范围的点。
- 增加地面去除，避免地面被误判为障碍物。
- 优化 occupancy grid，区分 occupied / free / unknown。
- 加入射线投影或 ray casting，建立自由空间。
- 尝试多帧地图融合。
- 学习并加入位姿估计。
- 后续迁移到 ROS2。
- 后续接入 Nav2、RTAB-Map 或其他 SLAM 框架。
