import json
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
import pyrender
from tqdm import tqdm
import trimesh

import time
from geometryhints.tools.mesh_renderer import Renderer, camera_marker, get_cam_pose_from_lookat_and_loc, create_light_array
from geometryhints.utils.cropping_utils import (
    find_image_collection_bounding_box,
    tightly_crop_images,
)
from geometryhints.utils.generic_utils import readlines
from geometryhints.utils.geometry_utils import rotx
from geometryhints.utils.rendering_utils import load_and_preprocess_mesh_for_rendering
from geometryhints.utils.visualization_utils import save_viz_video_frames, tile_images
import xml.etree.ElementTree as ET

import copy

def get_cam_settings_from_meshlab_dict(meshlab_camera_info: dict, image_width_target: int = None):
    """Returns cam pose and K from meshlab camera info and scaled image size.
    
    If scale_image_size is None, then the meshlab reported viewport size is returned 
    and the K is not scaled. Otherwise, the K is scaled to the new image size.
    
    """
    translation_vector_3 = np.array(meshlab_camera_info["TranslationVector"].strip().split(" ")).astype(float)[:3]
    rotation_matrix_33 = np.array(meshlab_camera_info["RotationMatrix"].strip().split(" ")).astype(float).reshape(4, 4)[:3,:3]
    focal_mm = float(meshlab_camera_info["FocalMm"].strip())
    center_px = np.array(meshlab_camera_info["CenterPx"].strip().split(" ")).astype(int)
    viewport_px = np.array(meshlab_camera_info["ViewportPx"].strip().split(" ")).astype(int)
    pixel_size_mm = np.array(meshlab_camera_info["PixelSizeMm"].strip().split(" ")).astype(float)
    
    # create K
    K_44 = np.eye(4)
    K_44[0, 2] = center_px[0] / viewport_px[0]
    K_44[1, 2] = center_px[1] / viewport_px[1]
    K_44[0, 0] = focal_mm / (pixel_size_mm[0] * viewport_px[0])
    K_44[1, 1] = focal_mm / (pixel_size_mm[0] * viewport_px[0])

    
    if image_width_target is not None:
        image_width = image_width_target
        image_height = int(image_width_target * (viewport_px[1] / viewport_px[0]))
    else:
        image_width = viewport_px[0]
        image_height = viewport_px[1]
        
    # scale K
    K_44[0] *= image_width
    K_44[1,2] *= image_height
    K_44[1,1] *= image_width
        
    # create cam pose
    # meshlab's trackball is fun! 
    cam_T_world_44 = np.identity(4)
    translate_44 = np.eye(4)
    translate_44[:3, 3] = translation_vector_3
    cam_T_world_44 = translate_44 @ cam_T_world_44
    rotation_matrix_44 = np.eye(4)
    rotation_matrix_44[:3, :3] = rotation_matrix_33
    cam_T_world_44 = rotation_matrix_44 @ cam_T_world_44
    world_T_cam_44 = np.linalg.inv(cam_T_world_44)
    
    # rotate 180 degrees around x
    rot = np.eye(4)
    rot[:3, :3] = rotx(np.pi)
    world_T_cam_44 =  world_T_cam_44 @ rot
    
    if False:
        fpv_camera = trimesh.scene.Camera(
            resolution=(200, 300), focal=(200, 200)
        )

        cam_marker_size = 0.7
        cam_marker_mesh = camera_marker(fpv_camera, cam_marker_size=cam_marker_size)[1]

        np_vertices = np.array(cam_marker_mesh.vertices)

        np_vertices = (
            world_T_cam_44
            @ np.concatenate([np_vertices, np.ones((np_vertices.shape[0], 1))], 1).T
        ).T

        np_vertices = np_vertices / np_vertices[:, 3][:, None]
        cam_marker_mesh = trimesh.Trimesh(
            vertices=np_vertices[:, :3], faces=cam_marker_mesh.faces
        )
        cam_marker_mesh.export("cam_marker.ply")

        print(f"K_44: {K_44}")
        print(f"world_T_cam_44: {world_T_cam_44}")
        
    return world_T_cam_44, K_44, image_width, image_height
    

def render_views(
    mesh_load_dirs: list[str],
    run_names: list[str],
    render_save_dir: Path,
    color_with: str,
    all_views_dict: dict, 
    render_width: int = 640 * 2,
    render_height: int = 480 * 2,
):
    """Renders top down views of each mesh and saves to render_save_dir as a video."""
    assert len(mesh_load_dirs) == len(run_names)

    combined_dir_name = "combined"
    individual_dir_name = "individual"
    if color_with == "normals":
        combined_dir_name += "_normals"
        individual_dir_name += "_normals"
    
    (render_save_dir / "fixed_views" / combined_dir_name).mkdir(exist_ok=True, parents=True)
    (render_save_dir / "fixed_views" / individual_dir_name).mkdir(exist_ok=True, parents=True)

    
    for scan, scan_views in tqdm(all_views_dict.items(), desc="Looping over scans: "):        
        # load a mesh for each method we care about
        try:
            meshes = [
                load_and_preprocess_mesh_for_rendering(
                    mesh_load_path=load_path.replace("SCAN_NAME", scan),
                    scan=scan,
                    color_with=color_with,
                )
                for load_path in mesh_load_dirs
            ]
        except IndexError as e:
            # It seems that IndexError is thrown by trimesh when a ply file is bad?
            print(f"Failed to load scene {scan} – {e}")
            continue

        for view_name, view_info in scan_views.items():
            cam_mat_44, K_44, image_width, image_height = get_cam_settings_from_meshlab_dict(
                view_info["meshlab_camera_info"], 
                image_width_target=None if view_info.get("use_original_res", "false") == "true" else render_width,
            )
        

            light_position = cam_mat_44[:3,3].copy()
            light_position[2] += float(view_info.get("light_vertical_shift", "1"))
            
            # shift lights in x and y direction of the camera
            # get the lookat of the fpv camera
            fpv_cam_transform = np.linalg.inv(cam_mat_44[:3, :3])
            z_vec = np.zeros((3,))
            # z is y axis in ScanNet
            z_vec[1] = -1
            current_fpv_look_at = fpv_cam_transform @ z_vec
            # shift lights by the lookat in x and y
            light_position[:2] += current_fpv_look_at[:2] * float(view_info.get("light_cam_x_y_shift", 0))
            
            light_world_T_cam_44 = np.eye(4)
            light_world_T_cam_44[:3, 3] = light_position
            # light_world_T_cam_44 = light_world_T_cam_44 @ shift
            lights = create_light_array(
                pyrender.SpotLight(
                    intensity=float(view_info.get("light_intensity", 20.0)),
                    outerConeAngle=np.pi/2
                ),
                light_world_T_cam_44,
                x_length=float(view_info.get("light_grid_spread", 1.0)),
                y_length=float(view_info.get("light_grid_spread", 1.0)),
                num_x=int(view_info.get("light_grid_count", 2)),
                num_y=int(view_info.get("light_grid_count", 2)),
            )

            # light debug
            # light_position = cam_mat_44[:3,3].copy()
            # light_position[2] += 1
            # light_pose = np.eye(4)
            # light_pose[:3, 3] = light_position
            # lights = []
            # lights.append([pyrender.SpotLight(intensity=20, outerConeAngle=np.pi/2), light_pose])
            # fpv_camera = trimesh.scene.Camera(
            #     resolution=(200, 300), focal=(200, 200)
            # )

            # cam_marker_size = 0.7
            # cam_marker_mesh = camera_marker(fpv_camera, cam_marker_size=cam_marker_size)[1]

            # np_vertices = np.array(cam_marker_mesh.vertices)

            # debug_pose_cam = np.eye(4)
            # np_vertices = (
            #     light_world_T_cam_44
            #     @ np.concatenate([np_vertices, np.ones((np_vertices.shape[0], 1))], 1).T
            # ).T

            # np_vertices = np_vertices / np_vertices[:, 3][:, None]
            # cam_marker_mesh = trimesh.Trimesh(
            #     vertices=np_vertices[:, :3], faces=cam_marker_mesh.faces
            # )
            
            renders = []
            names = []
            # Render the meshes for each method in turn
            for mesh, run_name in zip(meshes, run_names):
                renderer = Renderer(height=image_height, width=image_width, ambient_light=0.1)

                
                render_flags = (
                    pyrender.RenderFlags.NONE
                    # | pyrender.RenderFlags.SKIP_CULL_FACES
                    # | pyrender.RenderFlags.ALL_WIREFRAME
                )
                
                if view_info.get("cull_faces", "false") == "false":
                    render_flags = render_flags | pyrender.RenderFlags.SKIP_CULL_FACES
                
                if color_with == "normals":
                    render_flags = render_flags | pyrender.RenderFlags.FLAT
                else:
                    render_flags = render_flags | pyrender.RenderFlags.SHADOWS_ALL
                    
                render_color = renderer.render_mesh(
                    [mesh],
                    image_height,
                    image_width,
                    cam_mat_44,
                    K_44,
                    lights=copy.deepcopy(lights),
                    get_colour=True,
                    render_flags=render_flags,
                    znear=view_info["meshlab_camera_info"]["NearPlane"],
                )
                renderer.delete()
                if render_color is None:
                    print(f"Failed to render scene {scan} for method {run_name}")
                    continue

                renders.append(render_color)
                names.append(run_name)

                # if True:#export_meshes:
                    # mesh.export(render_save_dir / "individual_meshes" / f"{scan}_{run_name}.ply")

            # crop and save images
            # renders = tightly_crop_images(renders)

            if len(renders) > 0:
                # also save a combined image
                combined = np.hstack(renders)
                im_save_path = render_save_dir / "fixed_views" / combined_dir_name / f"{scan}_{view_name}.png"
                cv2.imwrite(str(im_save_path), combined[:, :, ::-1])

            for run_name, render in zip(names, renders):
                im_save_path = render_save_dir / "fixed_views" / individual_dir_name / f"{scan}_{view_name}_{run_name}.png"
                cv2.imwrite(str(im_save_path), render[:, :, ::-1])


def parse_view_xml(view_xml_file_path: Path):
    xml_tree = ET.parse(view_xml_file_path)
    all_views_dict = {}
    
    for scene_xml in xml_tree.getroot():
        scene_dict = {}
        for view_xml in scene_xml:
            view_dict = {}
            view_dict["meshlab_camera_info"] = {}
            
            for view_child in view_xml:
                if view_child.tag == "VCGCamera":
                    view_dict["meshlab_camera_info"].update(view_child.attrib)
                elif view_child.tag == "ViewSettings":
                    view_dict["meshlab_camera_info"].update(view_child.attrib)
                else:
                    view_dict.update(view_child.attrib)
            
            scene_dict[view_xml.attrib["name"]] = view_dict
            
        all_views_dict[scene_xml.attrib["name"]] = scene_dict
    
    
    return all_views_dict

@click.command()
@click.option(
    "--save_dir",
    type=Path,
    required=True,
)
@click.option(
    "--load_dir",
    type=str,
    help=(
        "Folders to load the meshes from, typically one folder for each method. "
        "Can be specified multiple times."
    ),
    required=True,
    multiple=True,
)
@click.option(
    "--run_name",
    type=str,
    help=(
        "The names of each method pointed to with the -load-dir arguments. "
        "Must have exactly the same number of -run-name options as -load-dir options."
    ),
    multiple=True,
    required=True,
)
@click.option(
    "--color_with",
    default="raw",
    type=click.Choice(["raw", "normals"]),
    help="What to color the vertices with",
)
@click.option(
    "--view_xml_path",
    type=Path,
    help="View dictironary path.",
)
@click.option(
    "--turntable",
    type=bool,
    default=False,
    is_flag=True,
    help="Whether to render a turntable video instead",
)
@click.option(
    "--turntable_interval",
    default=200,
    type=int,
    help="Number of intervals in a 360 spin video.",
)
def cli(
    load_dir: list[Path], save_dir: str, run_name: list[str], color_with: str, view_xml_path: Path, turntable: bool, turntable_interval: int
):  # click Paths are strings
    
    # copy past meshlab camera settings via Window -> "Copy camera settings to clipboard"
    # use copilot to help you parse the string
    # <?xml version="1.0"?>
    # <data>
    #     <scan name="scene0711_00">
    #         <view name="desk">
    #             <!-- your meshlab cam data goes here FIRST: -->
    #             <VCGCamera CameraType="0" LensDistortion="0 0" BinaryData="0" ViewportPx="1585 1099" CenterPx="792 549" TranslationVector="-3.25087 -2.48397 -1.64679 1" FocalMm="35.135296" PixelSizeMm="0.0369161 0.0369161" RotationMatrix="-0.959945 0.280115 0.00635567 0 -0.1541 -0.546768 0.822981 0 0.234005 0.789038 0.568032 0 0 0 0 1 "/>
    #             <ViewSettings FarPlane="40.799702" NearPlane="0.30310887" TrackScale="1.8665501"/>
    #             <RenderAttrib normals="true"/>
    #         </view>
    #     </scan>
    # </data>
    all_views_dict = parse_view_xml(view_xml_path)

    render_views(
        mesh_load_dirs=load_dir,
        render_save_dir=Path(save_dir),
        run_names=run_name,
        color_with=color_with,
        all_views_dict=all_views_dict,
    )


if __name__ == "__main__":
    cli()
