import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import torch

import os, time

from envs.models.panda.panda import Panda

def visualize_with_scene_widget(mesh_paths, transform_matrices, colors):
    """
    mesh_paths: list of str - paths to mesh files
    transform_matrices: list of torch.Tensor - 4x4 SE(3) transformation matrices
    colors: list of [r,g,b] lists (optional) - RGB colors for each mesh
    """
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Mesh Viewer", 1920, 1080)
    
    # Create SceneWidget
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(window.renderer)
    window.add_child(widget3d)
    
    # Load and add meshes
    bounds = o3d.geometry.AxisAlignedBoundingBox()
    for i, (mesh_path, transform) in enumerate(zip(mesh_paths, transform_matrices)):
        # Load mesh
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        
        # Set color if provided
        if colors is not None and i < len(colors):
            if isinstance(colors[i], (list, np.ndarray)):
                mesh.paint_uniform_color(colors[i])
        
        # Apply transformation
        if isinstance(transform, torch.Tensor):
            transform = transform.cpu().numpy()
        mesh.transform(transform)
        
        # Add to scene
        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        
        # Add mesh to scene with a unique name
        widget3d.scene.add_geometry(f"mesh_{i}", mesh, material)
        
        # Update bounding box
        # bounds.extend(mesh.get_axis_aligned_bounding_box())
    
    # Set camera view
    center = bounds.get_center()
    eye = center + np.array([1.0, 1.0, 1.0])  # Camera position
    up = np.array([0.0, 0.0, 1.0])            # Up vector
    
    # Setup camera view
    
    widget3d.scene.camera.look_at(
        [0, 0, 0.5], # camera lookat
        [0.5, -1, 0.6], # camera position
        [0, 0, 1] # fixed
    )
    widget3d.scene.set_lighting(widget3d.scene.LightingProfile.SOFT_SHADOWS, (-0.3, 0.3, -0.9))
    
    # Enable mouse controls
    widget3d.enable_scene_caching(True)
    widget3d.set_view_controls(gui.SceneWidget.Controls.FLY)
    
    # Save screenshot
    # img_o3d = widget3d.scene.scene.render_to_image()
    # o3d.io.write_image(save_path, img_o3d, 9)
    
    # Start the application
    gui.Application.instance.run()

# 사용 예시
if __name__ == "__main__":
    # 예시 transformation matrices
    robot = Panda()
    _, T_links = robot.solveForwardKinematics(torch.tensor([0.000,  -1.000,  0.000, -2.071,  0.000,  1.3,  -1.000]), return_T_link=True)
    
    transforms = [robot.T_base] + [T_links[i] for i in range(7)]
    
    # 예시 mesh 파일들
    mesh_paths = [
        "envs/models/panda/meshes/visual/link0.obj", 
        "envs/models/panda/meshes/visual/link1.obj",
        "envs/models/panda/meshes/visual/link2.obj",
        "envs/models/panda/meshes/visual/link3.obj",
        "envs/models/panda/meshes/visual/link4.obj",
        "envs/models/panda/meshes/visual/link5.obj",
        "envs/models/panda/meshes/visual/link6.obj",
        "envs/models/panda/meshes/visual/link7.obj",
    ]
    
    visualize_with_scene_widget(
        mesh_paths, 
        transforms,
        colors=[[0.5, 0.5, 0.5]]*8,
    )