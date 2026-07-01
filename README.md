# Orbbec 深度相机避障感知与局部建图模块

## 项目目标

本项目基于 Orbbec 深度相机，面向 ROS2 移动平台构建避障感知与局部建图模块。当前阶段不急于接入 ROS2，而是先把单帧离线局部栅格地图质量调好，确保点云到二维占据地图的结果稳定、方向正确、噪声可控，后续再封装为 ROS2 节点。

当前主线流程：

```text
Orbbec 深度相机
→ 深度图采集
→ depth image → gridmap
→ realtime depth frame → gridmap
→ 多帧融合
→ ROS2 节点封装
```

离线点云验证链路 `.ply → gridmap` 已完成，`depth.npy → gridmap` 已验证通过；当前阶段重点是 `realtime depth frame → gridmap`，为下一步 `ROS2 depth topic → OccupancyGrid` 做准备。

## 多帧局部栅格地图融合

`fuse_gridmaps.py` 用于对多个单帧 `gridmap*.npy` 做轻量级时间窗口投票融合，减少单帧深度噪声、孤立误检和障碍物闪烁。

它读取二维 0/1 栅格地图：

```text
0 = 非障碍
1 = 障碍物
```

默认从 `grid_data/` 读取 `gridmap*.npy`，使用前 5 帧，投票阈值为 3：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" fuse_gridmaps.py
```

也可以手动指定输入目录、窗口大小和投票阈值：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" fuse_gridmaps.py --input_dir grid_data --window_size 5 --vote_threshold 3
```

或者手动指定多个 `.npy` 文件：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" fuse_gridmaps.py --inputs grid_data/gridmap_0001.npy grid_data/gridmap_0002.npy grid_data/gridmap_0003.npy grid_data/gridmap_0004.npy grid_data/gridmap_0005.npy
```

默认输出到：

```text
grid_data/fused/
```

输出文件：

```text
fused_gridmap.npy
fused_gridmap.png
fusion_config.json
fusion_summary.csv
```

融合规则：

```text
对每个栅格统计 N 帧中被判定为障碍物的次数
如果次数 >= vote_threshold，则输出障碍物 1
否则输出非障碍 0
```

这个方法是轻量级 temporal voting，不是完整 SLAM。适合：

- 相机静止；
- 机器人低速移动；
- 局部避障地图稳定化；
- 减少深度噪声导致的障碍物闪烁。

局限性：

- 未使用 odometry；
- 未使用 TF 坐标变换；
- 未做多帧位姿对齐；
- 相机快速移动时可能产生拖影；
- 后续 ROS2 版本需要结合机器人位姿进行改进。

## 深度图直接生成栅格地图

`depth_to_gridmap.py` 是从离线点云流程走向 ROS2 实时流程的关键中间步骤。它直接读取保存好的深度图，不再依赖 `.ply` 中间文件：

```text
depth_data/depth_0001.npy
→ 3D camera points
→ X-Z projection
→ data/gridmap_from_depth/gridmap.npy
→ data/gridmap_from_depth/gridmap.png
```

基础运行：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" depth_to_gridmap.py
```

指定输入深度图和输出目录：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" depth_to_gridmap.py --input_depth depth_data/depth_0001.npy --output_dir data/gridmap_from_depth
```

指定相机内参：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" depth_to_gridmap.py --input_depth depth_data/depth_0001.npy --fx 574.961 --fy 574.961 --cx 320 --cy 240
```

主要参数：

```text
--input_depth：输入 .npy 或 16-bit .png 深度图
--depth_unit：深度单位，mm 或 m，默认 mm
--fx / --fy / --cx / --cy：相机内参
--y_axis_up / --no-y_axis_up：将图像向下的 v 轴转换为项目约定的 Y 向上
--min-x / --max-x：左右过滤范围
--min-y / --max-y：高度过滤范围
--min-z / --max-z：深度过滤范围
--resolution：栅格分辨率
--obstacle-threshold：障碍物点数阈值
--remove_ground：可选地面去除
--ground_y_threshold：地面高度阈值
--min_component_size：小连通区域过滤
```

坐标转换模型：

```text
Z = depth
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

注意：图像坐标 `v` 向下增加，而项目约定 `Y` 向上为正，因此默认启用 `--y_axis_up`，内部会把 Y 方向翻转。当前如果不手动指定内参，脚本会使用估计值并在终端提示；后续 ROS2 版本应从 `/camera/depth/camera_info` 读取真实内参。

默认输出：

```text
data/gridmap_from_depth/gridmap.npy
data/gridmap_from_depth/gridmap.png
data/gridmap_from_depth/config.json
```

## 实时深度流生成局部栅格地图

`realtime_depth_to_gridmap.py` 是 ROS2 接入前的 Python 实时验证版本。它复用当前 OpenNI 深度帧读取代码，从 Orbbec 相机实时读取 `uint16/mm` 深度帧，并调用 `depth_to_gridmap.py` 中的 `generate_gridmap_from_depth()` 生成局部栅格地图。

当前阶段链路：

```text
Orbbec realtime depth frame
→ generate_gridmap_from_depth()
→ realtime gridmap preview
→ optional depth/gridmap/metrics samples
```

基础运行：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" realtime_depth_to_gridmap.py
```

降低处理频率，例如每 3 帧处理 1 次：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" realtime_depth_to_gridmap.py --process_every_n_frames 3
```

周期性保存样本，默认不保存大量数据：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" realtime_depth_to_gridmap.py --save_every_n_frames 30 --output_dir data/realtime_gridmap
```

保存文件格式：

```text
depth_0001.npy
gridmap_0001.npy
gridmap_0001.png
metrics_0001.json
```

主要参数与 `depth_to_gridmap.py` 保持一致：

```text
--fx / --fy / --cx / --cy
--depth_unit
--min_x / --max_x
--min_y / --max_y
--min_z / --max_z
--resolution
--obstacle_threshold
--remove_ground
--ground_y_threshold
--min_component_size
```

实时窗口会显示 depth image 和 gridmap，按 `q` 或 ESC 安全退出。终端会周期性打印 frame index、有效深度像素数、过滤后有效点数、占据栅格数量、占据比例、处理耗时和估计 FPS。如果平均处理速度低于 10 FPS，脚本会提示可尝试降低分辨率、增大 `--process_every_n_frames`、缩小地图范围、增大 `--resolution` 或继续优化 numpy 向量化。

项目路线更新：

```text
depth.npy → gridmap：已完成
realtime depth frame → gridmap：当前阶段
ROS2 depth topic → OccupancyGrid：下一阶段
```

## 当前进度

| 模块 | 状态 | 主要文件 |
|---|---|---|
| 深度相机接入 | 已完成初版 | `orbbec_openni_utils.py`, `test_depth_minimal.py` |
| 深度帧保存 | 已完成初版 | `save_depth_frame.py`, `depth_data/` |
| 深度像素转 3D 坐标 | 已完成初版 | `click_depth_point.py` |
| 深度图转点云 | 已完成初版 | `depth_to_pointcloud.py` |
| 点云预处理 | 已完成初版 | `process_pointcloud.py` |
| 单帧点云转局部栅格地图 | 已完成初版 | `pointcloud_to_gridmap.py` |
| 深度图直接转局部栅格地图 | 已完成初版 | `depth_to_gridmap.py` |
| 实时深度流转局部栅格地图 | 当前阶段 | `realtime_depth_to_gridmap.py` |
| 多帧局部栅格地图融合 | 已完成初版 | `fuse_gridmaps.py` |
| 参数批量实验和结果记录 | 已完成初版 | `data/experiments/` |
| 地面去除和噪声连通域过滤 | 已完成初版 | `pointcloud_to_gridmap.py` |
| free / unknown / ray casting | 预留中 | 后续实现 |
| ROS2 / Nav2 接入 | 暂不进行 | 单帧地图质量稳定后再做 |

## 坐标系和地图方向

项目约定点云坐标系为：

```text
X：左右方向，右为正
Y：上下方向，上为正
Z：前方深度方向，前方为正
```

`pointcloud_to_gridmap.py` 使用 X-Z 平面生成俯视栅格地图：

- 图像底部中间是 camera；
- 图像上方是 forward，也就是 +Z；
- 图像左侧是 left，也就是 -X；
- 图像右侧是 right，也就是 +X；
- 如果地图方向不正确，使用 `--flip-y` 或 `--flip-z` 修正输入点云坐标，不在代码里写死方向。

## 单帧局部地图质量优化

主线脚本：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" pointcloud_to_gridmap.py
```

指定实验名称后，输出会保存到：

```text
data/experiments/<experiment_name>/
```

每次运行会生成：

```text
gridmap.npy
gridmap.png
config.json
```

`gridmap.npy` 当前仍保持二值格式：

```text
0 = 非障碍
1 = 障碍物
```

代码结构和文档已经为后续扩展预留：

```text
-1 = unknown 未知
0  = free 空闲
1 或 100 = occupied 障碍物
```

后续会通过 ray casting 根据相机视线推断 free 区域，再区分 free 和 unknown。

## 参数说明

常用参数：

```text
--min-x / --max-x：左右范围，默认 -2.5 到 2.5 m
--min-z / --max-z：前方深度范围，默认 0.3 到 5.0 m
--min-y / --max-y：高度范围，默认 -0.5 到 1.5 m
--resolution：栅格分辨率，默认 0.05 m/cell
--obstacle-threshold：单个格子达到多少点判定为障碍物，默认 3
--remove_ground：开启简单地面去除
--ground_y_threshold：地面高度阈值，低于该 Y 值的点会被移除
--min_component_size：过滤小连通障碍区域
--flip-y / --flip-z：修正输入点云坐标方向
```

`min_y` 用于过滤过低的点，例如地面噪声。`max_y` 用于过滤过高的点，例如墙上物体、天花板区域或不影响底盘避障的高处结构。

## 单次实验示例

基础运行：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" pointcloud_to_gridmap.py --experiment_name quality_single
```

开启地面去除和小连通区域过滤：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" pointcloud_to_gridmap.py --experiment_name quality_single --remove_ground --ground_y_threshold -0.45 --min_component_size 3
```

指定点云和参数：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" pointcloud_to_gridmap.py astra_pointcloud_raw.ply --min-y -0.4 --max-y 1.2 --resolution 0.05 --obstacle-threshold 3
```

终端会输出原始点数、有效点数、X/Z 过滤点数、高度过滤点数、地面去除前后点数、地图尺寸、障碍物格子数量、障碍物占比和输出路径。

## 批量参数实验

默认批量实验会测试：

```text
resolution: 0.03, 0.05, 0.10
obstacle_threshold: 1, 3, 5
```

运行：

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" pointcloud_to_gridmap.py --batch --experiment_name quality_batch --remove_ground --min_component_size 3
```

输出目录示例：

```text
data/experiments/quality_batch/res_0.05_thr_3/
```

每组参数都会保存：

```text
gridmap.npy
gridmap.png
config.json
```

批量实验还会生成：

```text
data/experiments/quality_batch/summary.csv
```

`summary.csv` 包含：

```text
resolution
obstacle_threshold
min_y
max_y
valid_points
occupied_cells
occupied_ratio
output_dir
```

## 推荐调参顺序

1. 先确认地图方向：camera 在底部中间，forward 朝上，left/right 正确。
2. 调整 `min_y/max_y`，去掉地面、天花板和高处无关物体。
3. 使用 `--remove_ground` 和 `--ground_y_threshold` 去掉明显地面点。
4. 在 `resolution = 0.03, 0.05, 0.10` 中比较地图细节和稳定性。
5. 在 `obstacle_threshold = 1, 3, 5` 中比较障碍物稀疏度和噪声。
6. 使用 `--min_component_size 3` 或更大值过滤孤立噪声格子。
7. 根据 `occupied_ratio` 和 `gridmap.png` 选择适合避障的参数。

如果障碍物太稀疏，降低 `obstacle_threshold` 或增大 `resolution`。如果噪点太多，提高 `obstacle_threshold`、缩小 `min_y/max_y`，或增大 `min_component_size`。

## 为什么当前暂不接入 ROS2

ROS2 节点封装会引入实时数据流、消息格式、TF、坐标变换和调试复杂度。如果单帧离线地图质量还不稳定，接入 ROS2 后问题会更难定位。因此当前先在离线 `.ply` 点云上验证地图方向、过滤策略、障碍物密度和噪声处理。

## 后续路线

```text
单帧地图优化
→ 多帧融合
→ 深度图直接生成地图
→ 实时深度帧生成地图
→ ROS2 节点封装
→ ROS2 topic → OccupancyGrid
→ Nav2 / SLAM 集成
```
