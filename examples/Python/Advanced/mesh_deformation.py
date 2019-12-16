# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_subdivision.py

import numpy as np
import open3d as o3d
import time
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../Misc"))
import meshes


def problem0():
    mesh = meshes.plane(height=1, width=1)
    mesh = mesh.subdivide_midpoint(3)
    vertices = np.asarray(mesh.vertices)
    # fmt: off
    static_ids = [
        1, 46, 47, 48, 16, 51, 49, 50, 6, 31, 33, 32, 11, 26, 27, 25, 0, 64, 65,
        20, 66, 68, 67, 7, 69, 71, 70, 22, 72, 74, 73, 3, 15, 44, 43, 45, 5, 41,
        40, 42, 13, 39, 37, 38, 2, 56, 55, 19, 61, 60, 59, 8, 76, 75, 77, 23
    ]
    # fmt: on
    static_positions = []
    for id in static_ids:
        static_positions.append(vertices[id])
    handle_ids = [4]
    handle_positions = [vertices[4] + np.array((0, 0, 0.4))]

    return mesh, static_ids + handle_ids, static_positions + handle_positions


def problem1():
    mesh = meshes.plane(height=1, width=1)
    mesh = mesh.subdivide_midpoint(3)
    vertices = np.asarray(mesh.vertices)
    # fmt: off
    static_ids = [
        1, 46, 15, 43, 5, 40, 13, 38, 2, 56, 37, 39, 42, 41, 45, 44, 48, 47
    ]
    # fmt: on
    static_positions = []
    for id in static_ids:
        static_positions.append(vertices[id])
    handle_ids = [21]
    handle_positions = [vertices[21] + np.array((0, 0, 0.4))]

    return mesh, static_ids + handle_ids, static_positions + handle_positions


def problem2():
    mesh = meshes.armadillo()
    vertices = np.asarray(mesh.vertices)
    static_ids = [idx for idx in np.where(vertices[:, 1] < -30)[0]]
    static_positions = []
    for id in static_ids:
        static_positions.append(vertices[id])
    handle_ids = [2490]
    handle_positions = [vertices[2490] + np.array((-40, -40, -40))]

    return mesh, static_ids + handle_ids, static_positions + handle_positions


def select_vertices(mesh):
    vertices = np.asarray(mesh.vertices)
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window("Select Vertices")
    vis.add_geometry(mesh)
    vis.run()
    ids = [idx for idx in vis.get_picked_points()]
    vis.destroy_window()
    pos = [vertices[idx] for idx in ids]
    return ids, pos


class TransformPoints(object):

    def __init__(self, mesh, static_ids, static_positions, handle_ids, handle_positions, deform=False):
        self.mesh = mesh
        self.static_ids = static_ids
        self.static_positions = static_positions
        self.handle_ids = handle_ids
        self.handle_positions = handle_positions
        self.deform = deform

        self.constraint_ids = np.array(self.static_ids + self.handle_ids, dtype=np.int32)

        self.handle_points =  o3d.geometry.PointCloud()
        self.handle_points.points = o3d.utility.Vector3dVector(handle_positions)
        self.handle_points.paint_uniform_color((1, 0, 0))

        self.static_points =  o3d.geometry.PointCloud()
        self.static_points.points = o3d.utility.Vector3dVector(static_positions)
        self.static_points.paint_uniform_color((0, 1, 0))

        verts = np.asarray(self.mesh.vertices)
        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        self.delta_t = 0.02 * max(maxs - mins)

    def deform_arap(self):
        self.handle_positions = [v for v in np.asarray(self.handle_points.points)]
        constraint_pos = o3d.utility.Vector3dVector(self.static_positions + self.handle_positions)
        mesh = self.mesh.deform_as_rigid_as_possible(
            o3d.utility.IntVector(self.constraint_ids), constraint_pos, max_iter=50)
        self.mesh.vertices = mesh.vertices

    def escape_callback(self, vis):
        self.run = False
        return False

    def translate(self, dim, dir):
        t = np.zeros((3,))
        t[dim] = dir * self.delta_t
        self.handle_points.translate(t)
        if self.deform:
            self.deform_arap()
        return False

    def rotate(self, dim, dir):
        r = np.zeros((3,))
        r[dim] = dir * np.pi / 180 * 5
        R = o3d.geometry.get_rotation_matrix_from_xyz(r)
        self.handle_points.rotate(R)
        if self.deform:
            self.deform_arap()
        return False

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("Transform Vertices")
        vis.add_geometry(self.mesh)
        vis.add_geometry(self.static_points)
        vis.add_geometry(self.handle_points)

        vis.register_key_callback(256, self.escape_callback)

        vis.register_key_callback(ord("A"), lambda x: self.translate(0, -1))
        vis.register_key_callback(ord("D"), lambda x: self.translate(0, +1))
        vis.register_key_callback(ord("S"), lambda x: self.translate(1, -1))
        vis.register_key_callback(ord("W"), lambda x: self.translate(1, +1))
        vis.register_key_callback(ord("Q"), lambda x: self.translate(2, -1))
        vis.register_key_callback(ord("E"), lambda x: self.translate(2, +1))

        vis.register_key_callback(ord("G"), lambda x: self.rotate(0, -1))
        vis.register_key_callback(ord("J"), lambda x: self.rotate(0, +1))
        vis.register_key_callback(ord("H"), lambda x: self.rotate(1, -1))
        vis.register_key_callback(ord("Y"), lambda x: self.rotate(1, +1))
        vis.register_key_callback(ord("T"), lambda x: self.rotate(2, -1))
        vis.register_key_callback(ord("U"), lambda x: self.rotate(2, +1))
        print("Press A,D,S,W,Q,E to translate the points")
        print("Press G,J,Y,H,T,U to rotate the points")
        print("Press ESC to finish")

        self.run = True
        while self.run:
            vis.update_geometry(self.mesh)
            vis.update_geometry(self.handle_points)
            vis.poll_events()
            vis.update_renderer()

        self.handle_positions = [v for v in np.asarray(self.handle_points.points)]
        return self.handle_positions


def problem3():
    mesh = o3d.io.read_triangle_mesh(
        # "/home/griegler/Desktop/Open3D/armadillo_1k.off"
        # "/home/griegler/Desktop/Open3D/cactus_highres.off"
        # '/Users/griegler/Desktop/ballerina_10k.ply'
        '/Users/griegler/Desktop/simple_hand.ply'
        # '/Users/griegler/Desktop/cleopatra-s-needle-at-embankment-london-1.stl'
    )
    mesh.compute_vertex_normals()
    # mesh = meshes.armadillo()
    print("select static points")
    static_ids, static_positions = select_vertices(mesh)
    print("select moving points")
    handle_ids, handle_positions = select_vertices(mesh)
    print("transform the moving points")
    handle_positions = TransformPoints(mesh, static_ids, static_positions, handle_ids, handle_positions, deform=True).run()
    return mesh, static_ids + handle_ids, static_positions + handle_positions


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.Debug)

    for mesh, constraint_ids, constraint_pos in [
            # problem0(),
            # problem1(),
            # problem2(),
            problem3()
    ]:
        constraint_ids = np.array(constraint_ids, dtype=np.int32)
        constraint_pos = o3d.utility.Vector3dVector(constraint_pos)
        tic = time.time()
        mesh_prime = mesh.deform_as_rigid_as_possible(
            o3d.utility.IntVector(constraint_ids), constraint_pos, max_iter=50)
        print("deform took {}[s]".format(time.time() - tic))
        mesh_prime.compute_vertex_normals()

        mesh.paint_uniform_color((1, 0, 0))
        handles = o3d.geometry.PointCloud()
        handles.points = constraint_pos
        handles.paint_uniform_color((0, 1, 0))
        o3d.visualization.draw_geometries([mesh, mesh_prime, handles])
