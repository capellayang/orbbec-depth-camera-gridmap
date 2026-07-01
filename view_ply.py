import open3d as o3d
import os

PLY_PATH = r"C:\Users\YYF\Documents\python\camera\astra_pointcloud_raw.ply"

if not os.path.exists(PLY_PATH):
    raise FileNotFoundError(f"找不到点云文件: {PLY_PATH}")

pcd = o3d.io.read_point_cloud(PLY_PATH)

print("点云读取成功")
print(pcd)
print("点数量:", len(pcd.points))

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

o3d.visualization.draw_geometries([pcd, axis])