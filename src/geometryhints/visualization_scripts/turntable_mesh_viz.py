from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np
import pyrender
from tqdm import tqdm

from geometryhints.tools.mesh_renderer import (
    Renderer,
    create_lights_above_mesh,
    get_cam_pose_from_lookat_and_loc,
)
from geometryhints.utils.cropping_utils import (
    find_image_collection_bounding_box,
    tightly_crop_images,
)
from geometryhints.utils.generic_utils import readlines
from geometryhints.utils.rendering_utils import load_and_preprocess_mesh_for_rendering
from geometryhints.utils.visualization_utils import save_viz_video_frames, tile_images


def create_birdseye_camera_viewpoint(
    scene_middle: np.ndarray,
    height_above_mesh: float = 8.0,
    pullback_dist: float = 5.0,
    cam_angle: float = 0,
) -> np.ndarray:
    """create the camera somewhere above the mesh

    Args:
        scene_middle (np.ndarray): the middle of the scene
        height_above_mesh (float): how high above the mesh the camera should be
        pullback_dist (float): is the amount 'off-centre' the camera is. if set to zero,
            camera will be directly above the mesh looking down
        cam_angle (float): angle of rotation around the z axis

    Returns:
        np.ndarray: the camera viewpoint as a 4x4 array
    """

    # get camera position in x and y based on the current angle.
    # cam_angle=0 gives us x_dist at 0 and y_dist at pullback_dist
    x_dist = pullback_dist * np.sin(cam_angle)
    y_dist = pullback_dist * np.cos(cam_angle)

    camera_position = scene_middle + np.array([x_dist, y_dist, height_above_mesh])
    look_at_vec = scene_middle - camera_position
    look_at_vec /= np.linalg.norm(look_at_vec)

    camera_viewpoint = get_cam_pose_from_lookat_and_loc(
        cam_location=camera_position, look_at_vec=look_at_vec
    )

    return camera_viewpoint


def render_top_down(
    mesh_load_dirs: list[str],
    run_names: list[str],
    render_save_dir: Path,
    color_with: str,
    scans: list[str],
    render_width: int = 640 * 2,
    render_height: int = 480 * 2,
):
    """Renders a top down view of each mesh and saves to render_save_dir"""
    assert len(mesh_load_dirs) == len(run_names)

    (render_save_dir / "combined").mkdir(exist_ok=True, parents=True)
    (render_save_dir / "individual").mkdir(exist_ok=True, parents=True)
    (render_save_dir / "individual_meshes").mkdir(exist_ok=True, parents=True)

    renderer = Renderer(height=render_height, width=render_width, ambient_light=0.2)

    # create intrinsics for all the renders
    K = np.eye(4)
    K[0, 2] = 0.5
    K[1, 2] = 0.5
    K[0, :] *= render_width
    K[1, :] *= render_height
    # after fixing the function that computes cam pose this is necessary.
    K[0, 0] *= 0.5
    K[1, 1] *= 0.5

    for scan in tqdm(scans):
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

        # currently the lights are positioned relative to the 'zeroth' mesh – this is fine?
        lights = create_lights_above_mesh(meshes[0])

        scene_middle = (meshes[0].vertices.max(0) + meshes[0].vertices.min(0)) / 2
        camera_viewpoint = create_birdseye_camera_viewpoint(scene_middle)

        renders = []
        names = []

        # Render the meshes for each method in turn
        for mesh, run_name in zip(meshes, run_names):
            render_color, _ = renderer.render_mesh(
                [mesh],
                render_height,
                render_width,
                camera_viewpoint,
                K,
                lights=lights,
                render_flags=pyrender.RenderFlags.SKIP_CULL_FACES,
                # | pyrender.constants.RenderFlags.SHADOWS_ALL,
            )

            if render_color is None:
                print(f"Failed to render scene {scan} for method {run_name}")
                continue

            renders.append(render_color)
            names.append(run_name)

            if True:  # export_meshes:
                mesh.export(render_save_dir / "individual_meshes" / f"{scan}_{run_name}.ply")

        # crop and save images
        renders = tightly_crop_images(renders)

        for run_name, render in zip(names, renders):
            im_save_path = render_save_dir / "individual" / f"{scan}_{run_name}.png"
            cv2.imwrite(str(im_save_path), render[:, :, ::-1])

        if len(renders) > 0:
            # also save a combined image
            combined = np.hstack(renders)
            im_save_path = render_save_dir / "combined" / f"{scan}.png"
            cv2.imwrite(str(im_save_path), combined[:, :, ::-1])


def render_top_down_turntable(
    mesh_load_dirs: list[str],
    run_names: list[str],
    render_save_dir: Path,
    color_with: str,
    scans: list[str],
    render_width: int = 640 * 2,
    render_height: int = 480 * 2,
    turntable_interval: int = 200,
):
    """Renders top down views of each mesh and saves to render_save_dir as a video."""
    assert len(mesh_load_dirs) == len(run_names)

    (render_save_dir / "combined_turntable").mkdir(exist_ok=True, parents=True)
    (render_save_dir / "individual_turntable").mkdir(exist_ok=True, parents=True)

    renderer = Renderer(height=render_height, width=render_width, ambient_light=0.2)

    print(mesh_load_dirs)

    for scan in tqdm(scans, desc="Looping over scans: "):
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

        # currently the lights are positioned relative to the 'zeroth' mesh – this is fine?
        lights = create_lights_above_mesh(meshes[0])

        scene_middle = meshes[0].vertices.mean(0)

        # find the optimal K for this scan. Get min and max bounds
        min_extent = 0
        max_extent = 0
        for mesh in meshes:
            min_extent = np.minimum(min_extent, mesh.vertices.min(0))
            max_extent = np.maximum(max_extent, mesh.vertices.max(0))

        diff = (max_extent - min_extent).max()

        focal_length_scale = 0.5
        if diff < 4:
            focal_length_scale *= 2.0

        # create intrinsics for all the renders based on diff
        K = np.eye(4)
        K[0, 2] = 0.5
        K[1, 2] = 0.5
        K[0] *= render_width
        K[1] *= render_height
        K[0, 0] *= focal_length_scale
        K[1, 1] *= focal_length_scale

        renders = {}
        names = []
        # Render the meshes for each method in turn
        for mesh, run_name in tqdm(zip(meshes, run_names), desc="Looping over methods: "):
            renders[run_name] = []
            names.append(run_name)

            cam_angles = np.linspace(
                start=0, stop=2 * np.pi, num=turntable_interval, endpoint=False
            )
            for cam_angle in cam_angles:
                camera_viewpoint = create_birdseye_camera_viewpoint(
                    scene_middle, cam_angle=cam_angle, height_above_mesh=6.0, pullback_dist=6.0
                )

                render_color = renderer.render_mesh(
                    [mesh],
                    render_height,
                    render_width,
                    camera_viewpoint,
                    K,
                    lights=lights,
                    get_colour=True,
                    # render_flags=pyrender.RenderFlags.SKIP_CULL_FACES,
                    # | pyrender.constants.RenderFlags.SHADOWS_ALL,
                )

                if render_color is None:
                    print(f"Failed to render scene {scan} for method {run_name}")
                    continue

                renders[run_name].append(render_color)

        # crop all images
        # unravel dicts
        all_renders = []
        for name in names:
            all_renders.extend(renders[name])
        # get bounds
        left, top, bottom, right = find_image_collection_bounding_box(all_renders)
        padding_ratio = 0.05

        bottom = min(render_height, round(bottom + render_height * padding_ratio))
        top = max(0, round(top - render_height * padding_ratio))

        right = min(render_width, round(right + render_width * padding_ratio))
        left = max(0, round(left - render_width * padding_ratio))

        for name in names:
            # crop
            renders[name] = [im[top:bottom, left:right] for im in renders[name]]

        for run_name in names:
            video_save_path = render_save_dir / "individual_turntable" / f"{scan}_{run_name}.mp4"
            # cv2.imwrite(str(im_save_path), render[:, :, ::-1])
            save_viz_video_frames(renders[run_name], str(video_save_path), fps=20)

        # tile images and save a combined video
        video_frames = []
        for frame_num in range(turntable_interval):
            frames = []
            for run_name in names:
                frames.append(renders[run_name][frame_num])
            frames = tile_images(frames)
            video_frames.append(frames)
        save_path = render_save_dir / "combined_turntable" / f"{scan}.mp4"
        save_viz_video_frames(video_frames, str(save_path), fps=20)


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
    type=click.Choice(["raw"]),
    help="What to color the vertices with",
)
@click.option(
    "--scan",
    default=[],
    type=str,
    multiple=True,
    help="Specified scans – otherwise will run all test scans",
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
    load_dir: list[Path],
    save_dir: str,
    run_name: list[str],
    color_with: str,
    scan: list[str],
    turntable: bool,
    turntable_interval: int,
):  # click Paths are strings
    if len(scan) == 0:
        scan = readlines(
            "/home/mohameds/code/geometryhints/data_splits/ScanNetv2/standard_split/scannetv2_test.txt"
        )

    if turntable:
        render_top_down_turntable(
            mesh_load_dirs=load_dir,
            render_save_dir=Path(save_dir),
            run_names=run_name,
            color_with=color_with,
            scans=scan,
            turntable_interval=turntable_interval,
        )
    else:
        render_top_down(
            mesh_load_dirs=load_dir,
            render_save_dir=Path(save_dir),
            run_names=run_name,
            color_with=color_with,
            scans=scan,
        )


if __name__ == "__main__":
    cli()
