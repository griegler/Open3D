# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/surface_reconstruction_poisson.py

import open3d as o3d
import numpy as np
import os

import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../Misc"))
import meshes


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    pcd = meshes.eagle()
    # o3d.visualization.draw_geometries([pcd])

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
    __import__("ipdb").set_trace()
