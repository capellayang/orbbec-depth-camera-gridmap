# Orbbec Depth Camera Obstacle Perception and Local Mapping Module

## Project Goal

This project uses an Orbbec depth camera to build an obstacle perception and local mapping module for a future ROS2 mobile platform. The current stage does not focus on ROS2 integration yet. The priority is to optimize and validate single-frame offline local grid maps so that point-cloud-to-grid results are stable, correctly oriented, and useful for obstacle avoidance.

Current main pipeline:

```text
Orbbec depth camera
-> depth frame capture
-> depth image -> gridmap
-> realtime depth frame -> gridmap
-> multi-frame fusion
-> ROS2 node packaging
```

The offline `.ply -> gridmap` path is complete, and `depth.npy -> gridmap` has been validated. The current stage focuses on `realtime depth frame -> gridmap` as preparation for the next `ROS2 depth topic -> OccupancyGrid` step.

## Multi-Frame Local Grid Map Fusion

`fuse_gridmaps.py` fuses multiple single-frame `gridmap*.npy` files with lightweight temporal voting. The goal is to reduce single-frame depth noise, isolated false positives, and obstacle flicker.

It reads 2D binary grids:

```text
0 = non-obstacle
1 = occupied obstacle
```

By default, it reads `gridmap*.npy` from `grid_data/`, uses the first 5 sorted frames, and applies a vote threshold of 3:

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" fuse_gridmaps.py
```

Specify input directory, window size, and vote threshold:

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" fuse_gridmaps.py --input_dir grid_data --window_size 5 --vote_threshold 3
```

Or manually specify multiple `.npy` files:

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" fuse_gridmaps.py --inputs grid_data/gridmap_0001.npy grid_data/gridmap_0002.npy grid_data/gridmap_0003.npy grid_data/gridmap_0004.npy grid_data/gridmap_0005.npy
```

Default output directory:

```text
grid_data/fused/
```

Output files:

```text
fused_gridmap.npy
fused_gridmap.png
fusion_config.json
fusion_summary.csv
```

Fusion rule:

```text
For each cell, count how many frames marked it as occupied.
If count >= vote_threshold, output occupied 1.
Otherwise output non-obstacle 0.
```

This method is lightweight temporal voting, not full SLAM. It is suitable for:

- static camera tests;
- slow robot motion;
- local obstacle-map stabilization;
- reducing depth-noise flicker.

Limitations:

- no odometry;
- no TF transform;
- no multi-frame pose alignment;
- fast camera motion may create ghosting or smear;
- a later ROS2 version should fuse maps with robot pose.

## Direct Depth Image to Grid Map

`depth_to_gridmap.py` is the key intermediate step from the offline point-cloud workflow toward a real-time ROS2 workflow. It reads a saved depth image directly and removes the `.ply` intermediate dependency:

```text
depth_data/depth_0001.npy
-> 3D camera points
-> X-Z projection
-> data/gridmap_from_depth/gridmap.npy
-> data/gridmap_from_depth/gridmap.png
```

Basic run:

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" depth_to_gridmap.py
```

Specify input depth image and output directory:

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" depth_to_gridmap.py --input_depth depth_data/depth_0001.npy --output_dir data/gridmap_from_depth
```

Specify camera intrinsics:

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" depth_to_gridmap.py --input_depth depth_data/depth_0001.npy --fx 574.961 --fy 574.961 --cx 320 --cy 240
```

Main parameters:

```text
--input_depth: input .npy or 16-bit .png depth image
--depth_unit: depth unit, mm or m, default mm
--fx / --fy / --cx / --cy: camera intrinsics
--y_axis_up / --no-y_axis_up: convert image v-down coordinates into project Y-up coordinates
--min-x / --max-x: left-right filter range
--min-y / --max-y: height filter range
--min-z / --max-z: depth filter range
--resolution: grid resolution
--obstacle-threshold: occupied-cell point threshold
--remove_ground: optional ground removal
--ground_y_threshold: ground height threshold
--min_component_size: small connected-component filtering
```

Camera projection model:

```text
Z = depth
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

Image coordinate `v` increases downward, while the project convention uses positive Y upward. Therefore `--y_axis_up` is enabled by default and flips the Y direction internally. If intrinsics are not manually provided, the script uses estimated values and prints a warning. A later ROS2 version should read real intrinsics from `/camera/depth/camera_info`.

Default outputs:

```text
data/gridmap_from_depth/gridmap.npy
data/gridmap_from_depth/gridmap.png
data/gridmap_from_depth/config.json
```

## Real-Time Depth Stream to Local Grid Map

`realtime_depth_to_gridmap.py` is the Python validation version before ROS2 integration. It reuses the current OpenNI depth-frame reader, reads live `uint16/mm` depth frames from the Orbbec camera, and calls `generate_gridmap_from_depth()` from `depth_to_gridmap.py` to generate a local grid map.

Current stage pipeline:

```text
Orbbec realtime depth frame
-> generate_gridmap_from_depth()
-> realtime gridmap preview
-> optional depth/gridmap/metrics samples
```

Basic run:

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" realtime_depth_to_gridmap.py
```

Process one frame every 3 camera frames:

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" realtime_depth_to_gridmap.py --process_every_n_frames 3
```

Save periodic samples. Saving is disabled by default to avoid growing the data directory too quickly:

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" realtime_depth_to_gridmap.py --save_every_n_frames 30 --output_dir data/realtime_gridmap
```

Saved file format:

```text
depth_0001.npy
gridmap_0001.npy
gridmap_0001.png
metrics_0001.json
```

Main parameters match `depth_to_gridmap.py` where possible:

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

The realtime windows show the depth image and gridmap. Press `q` or ESC to exit safely. The terminal periodically prints frame index, valid depth pixels, valid points after filtering, occupied cells, occupied ratio, processing time, and estimated FPS. If average processing speed falls below 10 FPS, the script suggests lowering resolution, increasing `--process_every_n_frames`, shrinking the map range, increasing `--resolution`, or further optimizing numpy vectorization.

Project route update:

```text
depth.npy -> gridmap: complete
realtime depth frame -> gridmap: current stage
ROS2 depth topic -> OccupancyGrid: next stage
```

## Current Status

| Module | Status | Main Files |
|---|---|---|
| Depth camera access | First version done | `orbbec_openni_utils.py`, `test_depth_minimal.py` |
| Depth frame saving | First version done | `save_depth_frame.py`, `depth_data/` |
| Depth pixel to 3D coordinate | First version done | `click_depth_point.py` |
| Depth image to point cloud | First version done | `depth_to_pointcloud.py` |
| Point cloud preprocessing | First version done | `process_pointcloud.py` |
| Single-frame point cloud to local grid map | First version done | `pointcloud_to_gridmap.py` |
| Direct depth image to local grid map | First version done | `depth_to_gridmap.py` |
| Realtime depth stream to local grid map | Current stage | `realtime_depth_to_gridmap.py` |
| Multi-frame local grid fusion | First version done | `fuse_gridmaps.py` |
| Batch parameter experiments and result logging | First version done | `data/experiments/` |
| Ground removal and connected-component denoising | First version done | `pointcloud_to_gridmap.py` |
| free / unknown / ray casting | Reserved for later | Planned |
| ROS2 / Nav2 integration | Not current stage | After grid quality is stable |

## Coordinate Convention and Map Direction

The project uses this point cloud convention:

```text
X: left-right, positive to the right
Y: vertical, positive upward
Z: forward depth, positive forward
```

`pointcloud_to_gridmap.py` generates a top-down grid map on the X-Z plane:

- camera is at the bottom center of the image;
- forward is the top of the image, meaning +Z;
- left is -X;
- right is +X;
- if the map orientation is wrong, fix the input convention with `--flip-y` or `--flip-z` instead of hard-coding direction changes.

## Single-Frame Local Map Quality Optimization

Main script:

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" pointcloud_to_gridmap.py
```

With an experiment name, outputs go to:

```text
data/experiments/<experiment_name>/
```

Each run saves:

```text
gridmap.npy
gridmap.png
config.json
```

The current `gridmap.npy` is still binary:

```text
0 = non-obstacle
1 = occupied obstacle
```

The code and documentation are prepared for a later occupancy representation:

```text
-1 = unknown
0  = free
1 or 100 = occupied
```

Future ray casting will use camera line-of-sight to infer free space and separate free from unknown.

## Parameters

Common parameters:

```text
--min-x / --max-x: left-right range, default -2.5 to 2.5 m
--min-z / --max-z: forward depth range, default 0.3 to 5.0 m
--min-y / --max-y: height range, default -0.5 to 1.5 m
--resolution: grid resolution, default 0.05 m/cell
--obstacle-threshold: point count needed to mark a cell occupied, default 3
--remove_ground: enable simple ground removal
--ground_y_threshold: remove points below this Y value when ground removal is enabled
--min_component_size: remove small connected occupied components
--flip-y / --flip-z: fix source point cloud axis direction
```

`min_y` removes points that are too low, such as ground noise. `max_y` removes points that are too high, such as wall-mounted objects, ceiling regions, or structures that should not affect mobile-base obstacle avoidance.

## Single-Run Examples

Basic run:

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" pointcloud_to_gridmap.py --experiment_name quality_single
```

Enable ground removal and small-component filtering:

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" pointcloud_to_gridmap.py --experiment_name quality_single --remove_ground --ground_y_threshold -0.45 --min_component_size 3
```

Specify cloud and map parameters:

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" pointcloud_to_gridmap.py astra_pointcloud_raw.ply --min-y -0.4 --max-y 1.2 --resolution 0.05 --obstacle-threshold 3
```

The terminal prints raw point count, valid point count, X/Z filtered count, height filtered count, ground-removal point changes, map size, occupied cell count, occupied ratio, and output paths.

## Batch Parameter Experiments

The default batch experiment tests:

```text
resolution: 0.03, 0.05, 0.10
obstacle_threshold: 1, 3, 5
```

Run:

```powershell
& "C:\Users\YYF\miniconda3\envs\orbbec\python.exe" pointcloud_to_gridmap.py --batch --experiment_name quality_batch --remove_ground --min_component_size 3
```

Output directory example:

```text
data/experiments/quality_batch/res_0.05_thr_3/
```

Each parameter case saves:

```text
gridmap.npy
gridmap.png
config.json
```

Batch mode also writes:

```text
data/experiments/quality_batch/summary.csv
```

`summary.csv` includes:

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

## Recommended Tuning Order

1. Confirm map direction first: camera at bottom center, forward upward, left/right correct.
2. Tune `min_y/max_y` to remove ground, ceiling, and high irrelevant objects.
3. Enable `--remove_ground` and adjust `--ground_y_threshold` for obvious floor points.
4. Compare `resolution = 0.03, 0.05, 0.10`.
5. Compare `obstacle_threshold = 1, 3, 5`.
6. Use `--min_component_size 3` or larger to remove isolated noise cells.
7. Choose parameters by comparing `occupied_ratio` and `gridmap.png`.

If obstacles are too sparse, lower `obstacle_threshold` or increase `resolution`. If there is too much noise, increase `obstacle_threshold`, narrow `min_y/max_y`, or increase `min_component_size`.

## Why ROS2 Is Not the Current Step

ROS2 packaging adds live data streams, message types, TF, coordinate transforms, and runtime debugging complexity. If the single-frame offline map is not stable yet, ROS2 integration makes problems harder to isolate. The current stage therefore validates map direction, filtering strategy, obstacle density, and noise handling on offline `.ply` files first.

## Roadmap

```text
single-frame map optimization
-> multi-frame fusion
-> direct depth-to-grid mapping
-> realtime depth-frame mapping
-> ROS2 node packaging
-> ROS2 topic -> OccupancyGrid
-> Nav2 / SLAM integration
```
