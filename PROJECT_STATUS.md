# Project Status and File Inventory

This document records the current state of the Orbbec Astra Pro Plus depth camera mapping project. It is intentionally more detailed than the README files and includes file-by-file classification.

## Current Main Goal

Learn the mapping pipeline step by step:

```text
Depth acquisition -> Depth display -> 3D projection -> Point cloud -> Point cloud filtering -> X-Z projection -> 2D local occupancy grid -> SLAM foundations
```

The project has reached the early stage of generating a single-frame 2D local occupancy grid from a saved `.ply` point cloud. The next best step is to add height filtering, ground removal, and a more standard occupancy grid model with occupied / free / unknown cells.

## Completed

- Connected Orbbec Astra Pro Plus through OpenNI2.
- Read live depth frames with Python.
- Displayed pseudo-colored depth images.
- Saved raw depth frames as `.npy`.
- Estimated camera intrinsics from OpenNI FOV:
  - `fx = 574.9614356132867`
  - `fy = 574.9614061048316`
  - `cx = 320.0`
  - `cy = 240.0`
- Implemented pixel-to-3D projection:
  - `Z = depth[v, u] / 1000.0`
  - `X = (u - cx) * Z / fx`
  - `Y = (v - cy) * Z / fy`
- Implemented click-to-print `u/v/depth/X/Y/Z`.
- Converted saved depth images into `.ply` point clouds.
- Viewed and filtered point clouds with Open3D.
- Generated a first 2D local occupancy grid from a single `.ply` point cloud.
- Centralized OpenNI path and cleanup helpers in `orbbec_openni_utils.py`.

## In Progress

- Stabilizing OpenNI2 startup and shutdown on Windows.
- Cleaning the project direction away from person extraction and toward robot mapping.
- Improving the occupancy grid pipeline.

## Next Steps

1. Add Y-axis height filtering to remove points that are too high or too low for obstacle mapping.
2. Add ground removal so the floor is not treated as an obstacle.
3. Update occupancy grid logic to distinguish `occupied`, `free`, and `unknown`.
4. Add ray casting from the camera origin to mark free space.
5. Try multi-frame fusion after adding pose or odometry.
6. Study pose estimation.
7. Move the pipeline into ROS2.
8. Integrate with Nav2 or RTAB-Map.

## Paused or Deprecated Experiments

The wall/person extraction experiments are useful learning material but are not part of the current mainline. Do not continue expanding person recognition for now.

Related files:

- `extract_person_near_wall.py`
- `wall.ply`
- `non_wall.ply`
- `foreground.ply`
- `person.ply`
- `person_main_cluster.ply`
- `other_foreground.ply`
- `candidate_clusters_info.txt`

## File Inventory

| File or Directory | Type | Main Purpose | Stage | Mainline? | Recommendation |
|---|---|---|---|---|---|
| `.git/` | Directory | Git repository metadata | Project management | Yes | Keep |
| `.agents/` | Directory | Local agent/tooling state | Tooling | No | Keep if tools need it |
| `__pycache__/` | Directory | Python bytecode cache | Temporary | No | Safe to delete |
| `depth_data/` | Directory | Saved depth frames and display images | Depth capture | Yes | Keep sample frames; archive older captures later |
| `depth_data/depth_0001.npy` | NPY | Latest raw uint16 depth frame | Depth capture | Yes | Keep |
| `depth_data/depth_0002.npy` | NPY | Raw uint16 depth sample | Depth capture | Yes, sample | Keep or archive |
| `depth_data/depth_0003.npy` | NPY | Raw uint16 depth sample | Depth capture | Yes, sample | Keep or archive |
| `depth_data/depth_0004.npy` | NPY | Raw uint16 depth sample | Depth capture | Yes, sample | Keep or archive |
| `depth_data/depth_0005.npy` | NPY | Raw uint16 depth sample | Depth capture | Yes, sample | Keep or archive |
| `depth_data/depth_0006.npy` | NPY | Raw uint16 depth sample | Depth capture | Yes, sample | Keep or archive |
| `depth_data/depth_0007.npy` | NPY | Raw uint16 depth sample; currently referenced by `depth_to_pointcloud.py` | Depth to point cloud | Yes | Keep |
| `depth_data/depth_show_0001.png` | PNG | Display image for depth frame | Debug display | No | Keep or archive |
| `depth_data/depth_show_0002.png` | PNG | Display image for depth frame | Debug display | No | Keep or archive |
| `depth_data/depth_show_0003.png` | PNG | Display image for depth frame | Debug display | No | Keep or archive |
| `depth_data/depth_show_0004.png` | PNG | Display image for depth frame | Debug display | No | Keep or archive |
| `depth_data/depth_show_0005.png` | PNG | Display image for depth frame | Debug display | No | Keep or archive |
| `depth_data/depth_show_0006.png` | PNG | Display image for depth frame | Debug display | No | Keep or archive |
| `depth_data/depth_show_0007.png` | PNG | Display image for depth frame | Debug display | No | Keep or archive |
| `orbbec_openni_utils.py` | Python | Shared OpenNI initialization, frame reading, and cleanup | Depth acquisition | Yes | Keep |
| `test_depth_minimal.py` | Python | Minimal live depth stream test | Depth display | Yes | Keep |
| `save_depth_frame.py` | Python | Save raw depth frames and display images | Depth capture | Yes | Keep |
| `click_depth_point.py` | Python | Click depth image and print `u/v/depth/X/Y/Z` | Intrinsics / 3D projection | Yes | Keep |
| `depth_click.py` | Python | Compatibility wrapper for `click_depth_point.py` | Compatibility | Partial | Keep or archive |
| `print.py` | Python | Compatibility wrapper for old click-depth test | Temporary | No | Rename or remove later |
| `depth_to_pointcloud.py` | Python | Convert saved depth `.npy` to `.ply` point cloud | Depth to point cloud | Yes | Keep |
| `view_ply.py` | Python | View a `.ply` file with Open3D | Point cloud viewing | Yes | Keep |
| `process_pointcloud.py` | Python | Downsample, denoise, crop, and save processed point cloud | Point cloud preprocessing | Yes | Keep |
| `pointcloud_to_occupancy_grid.py` | Python | Generate X-Z 2D local occupancy grid from `.ply` | Mapping | Yes | Keep |
| `depth_obstacle_detector.py` | Python | Simple ROI-based obstacle detector from live depth | Obstacle detection | Related | Keep |
| `extract_person_near_wall.py` | Python | Wall segmentation, foreground extraction, clustering, person-like extraction | Old person/wall experiment | No | Archive |
| `astra_pointcloud_raw.ply` | PLY | Raw point cloud generated from depth | Depth to point cloud | Yes | Keep |
| `astra_pointcloud.ply` | PLY | Earlier point cloud sample | Point cloud data | Sample | Keep |
| `astra_pointcloud_processed.ply` | PLY | Processed point cloud from filtering | Point cloud preprocessing | Yes | Keep |
| `filtered_for_grid.ply` | PLY | Filtered point cloud used for occupancy grid | Mapping | Yes | Keep |
| `occupancy_grid.png` | PNG | Generated 2D local occupancy grid image | Mapping | Yes | Keep |
| `wall.ply` | PLY | Wall plane extraction result | Old person/wall experiment | No | Archive |
| `non_wall.ply` | PLY | Non-wall extraction result | Old person/wall experiment | No | Archive |
| `foreground.ply` | PLY | Foreground points near wall | Old person/wall experiment | No | Archive |
| `person.ply` | PLY | Person-like extracted point cloud | Old person/wall experiment | No | Archive |
| `person_main_cluster.ply` | PLY | Main person-like cluster | Old person/wall experiment | No | Archive |
| `other_foreground.ply` | PLY | Other foreground clusters | Old person/wall experiment | No | Archive |
| `candidate_clusters_info.txt` | TXT | Cluster statistics from old extraction experiment | Debug / old experiment | No | Archive |
| `obstacle_result.json` | JSON | Latest obstacle detector output; currently empty | Obstacle detection output | Related | Keep |

## Mainline Running Order

```text
test_depth_minimal.py
  -> save_depth_frame.py
  -> click_depth_point.py
  -> depth_to_pointcloud.py
  -> view_ply.py
  -> process_pointcloud.py
  -> pointcloud_to_occupancy_grid.py
```

## Technical Roadmap

| Step | Goal | Status |
|---|---|---|
| 1 | Reliable depth acquisition | Mostly done, stability issues remain |
| 2 | Depth frame saving | Done |
| 3 | Pixel-to-3D projection | Done |
| 4 | Point cloud generation | Done |
| 5 | Point cloud filtering | First version done |
| 6 | Single-frame local occupancy grid | First version done |
| 7 | Height filtering | Next |
| 8 | Ground removal | Next |
| 9 | Free-space ray casting | Planned |
| 10 | Multi-frame fusion | Planned |
| 11 | Pose estimation | Planned |
| 12 | SLAM | Planned |
| 13 | ROS2 / Nav2 integration | Planned |

## Important Notes

- Current project status: single-frame point cloud to 2D local occupancy grid has been implemented at a beginner level.
- The next priority is height filtering, ground removal, and a more standard occupancy grid.
- `depth_display.png` or `depth_show_*.png` images are visualization outputs only and must not be used as raw depth.
- Avoid using `person.ply` as the current mapping input unless intentionally reviewing the old person extraction experiment.
- Do not delete old experiment files yet; archive them when the main mapping pipeline is cleaner.
