"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""
import argparse
import json
import math
import os
import random
import sys
import time
import glob
import urllib.request
import uuid
from typing import Tuple
from mathutils import Vector, Matrix
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import numpy as np
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

import bpy
from mathutils import Vector

# import OpenEXR
# import Imath
from PIL import Image

# import blenderproc as bproc

bpy.app.debug_value=256

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="/views_whole_sphere-test2")
parser.add_argument(
    "--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--num_images", type=int, default=16)
parser.add_argument("--random_images", type=int, default=4)
parser.add_argument("--fix_images", type=int, default=16)
parser.add_argument("--random_ortho", type=int, default=2)
parser.add_argument("--device", type=str, default="CUDA")

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)



print('===================', args.engine, '===================')

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 1.
cam.data.lens = 35
cam.data.sensor_height = 32
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

# setup lighting
# bpy.ops.object.light_add(type="AREA")
# light2 = bpy.data.lights["Area"]
# light2.energy = 3000
# bpy.data.objects["Area"].location[2] = 0.5
# bpy.data.objects["Area"].scale[0] = 100
# bpy.data.objects["Area"].scale[1] = 100
# bpy.data.objects["Area"].scale[2] = 100

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100
render.threads_mode = 'FIXED'  # 使用固定线程数模式
render.threads = 32  # 设置线程数

scene.cycles.device = "GPU"
scene.cycles.samples = 128   # 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3  # 3
scene.cycles.transmission_bounces = 3   # 3
# scene.cycles.filter_width = 0.01
bpy.context.scene.cycles.adaptive_threshold = 0
scene.cycles.use_denoising = True
scene.render.film_transparent = True

bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = 'CUDA' # or "OPENCL"
bpy.context.scene.cycles.tile_size = 8192


# eevee = scene.eevee
# eevee.use_soft_shadows = True
# eevee.use_ssr = True
# eevee.use_ssr_refraction = True
# eevee.taa_render_samples = 64
# eevee.use_gtao = True
# eevee.gtao_distance = 1
# eevee.use_volumetric_shadows = True
# eevee.volumetric_tile_size = '2'
# eevee.gi_diffuse_bounces = 1
# eevee.gi_cubemap_resolution = '128'
# eevee.gi_visibility_resolution = '16'
# eevee.gi_irradiance_smoothing = 0


# for depth & normal
context.view_layer.use_pass_normal = True
context.view_layer.use_pass_z = True
context.scene.use_nodes = True


tree = bpy.context.scene.node_tree
nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links

# Clear default nodes
for n in nodes:
    nodes.remove(n)

# # Create input render layer node.
render_layers = nodes.new('CompositorNodeRLayers')

scale_normal = nodes.new(type="CompositorNodeMixRGB")
scale_normal.blend_type = 'MULTIPLY'
scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])
bias_normal = nodes.new(type="CompositorNodeMixRGB")
bias_normal.blend_type = 'ADD'
bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
links.new(scale_normal.outputs[0], bias_normal.inputs[1])
normal_file_output = nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = 'Normal Output'
links.new(bias_normal.outputs[0], normal_file_output.inputs[0])

normal_file_output.format.file_format = "OPEN_EXR" # default is "PNG"
normal_file_output.format.color_mode = "RGB"  # default is "BW"

# depth_file_output = nodes.new(type="CompositorNodeOutputFile")
# depth_file_output.label = 'Depth Output'
# links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
# depth_file_output.format.file_format = "OPEN_EXR" # default is "PNG"
# depth_file_output.format.color_mode = "RGB"  # default is "BW"

def prepare_depth_outputs():
    tree = bpy.context.scene.node_tree
    links = tree.links
    render_node = tree.nodes['Render Layers']
    depth_out_node = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_map_node = tree.nodes.new(type="CompositorNodeMapRange")
    depth_out_node.base_path = ''
    depth_out_node.format.file_format = 'OPEN_EXR'
    depth_out_node.format.color_depth = '32'

    depth_map_node.inputs[1].default_value = 0.54
    depth_map_node.inputs[2].default_value = 1.96
    depth_map_node.inputs[3].default_value = 0
    depth_map_node.inputs[4].default_value = 1
    depth_map_node.use_clamp = True
    links.new(render_node.outputs[2],depth_map_node.inputs[0])
    links.new(depth_map_node.outputs[0], depth_out_node.inputs[0])
    return depth_out_node, depth_map_node

# depth_file_output, depth_map_node = prepare_depth_outputs()


def exr_to_png(exr_path):
    depth_path = exr_path.replace('.exr', '.png')
    exr_image = OpenEXR.InputFile(exr_path)
    dw = exr_image.header()['dataWindow']
    (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    def read_exr(s, width, height):
        mat = np.fromstring(s, dtype=np.float32)
        mat = mat.reshape(height, width)
        return mat

    dmap, _, _ = [read_exr(s, width, height) for s in exr_image.channels('BGR', Imath.PixelType(Imath.PixelType.FLOAT))]
    dmap = np.clip(np.asarray(dmap,np.float64),a_max=1.0, a_min=0.0) * 65535
    dmap = Image.fromarray(dmap.astype(np.uint16))
    dmap.save(depth_path)
    exr_image.close()
    # os.system('rm {}'.format(exr_path))

def extract_depth(directory):
    fns = glob.glob(f'{directory}/*.exr')
    for fn in fns: exr_to_png(fn)
    os.system(f'rm {directory}/*.exr')

def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )

def sample_spherical(radius=3.0, maxz=3.0, minz=0.):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        vec[2] = np.abs(vec[2])
        vec = vec / np.linalg.norm(vec, axis=0) * radius
        if maxz > vec[2] > minz:
            correct = True
    return vec

def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
#         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec

def randomize_camera():
    elevation = random.uniform(0., 90.)
    azimuth = random.uniform(0., 360)
    distance = random.uniform(0.8, 1.6)
    return set_camera_location(elevation, azimuth, distance)

def set_camera_location(elevation, azimuth, distance):
    # from https://blender.stackexchange.com/questions/18530/
    x, y, z = sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera

def set_camera_mvdream(azimuth, elevation, distance):
    # theta, phi = np.deg2rad(azimuth), np.deg2rad(elevation)
    azimuth, elevation = np.deg2rad(azimuth), np.deg2rad(elevation)
    point = (
        distance * math.cos(azimuth) * math.cos(elevation),
        distance * math.sin(azimuth) * math.cos(elevation),
        distance * math.sin(elevation),
    )
    camera = bpy.data.objects["Camera"]
    camera.location = point

    direction = -camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera

def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K


def get_calibration_matrix_K_from_blender_for_ortho(camd, ortho_scale):
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    fx = resolution_x_in_px / ortho_scale
    fy = resolution_y_in_px / ortho_scale / pixel_aspect_ratio

    cx = resolution_x_in_px / 2
    cy = resolution_y_in_px / 2

    K = Matrix(
        ((fx, 0, cx),
        (0, fy, cy),
        (0 , 0, 1)))
    return K


def get_3x4_RT_matrix_from_blender(cam):
    bpy.context.view_layer.update()
    location, rotation = cam.matrix_world.decompose()[0:2]
    R = np.asarray(rotation.to_matrix())
    t = np.asarray(location)

    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    R = R.T
    t = -R @ t
    R_world2cv = cam_rec @ R
    t_world2cv = cam_rec @ t

    RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)
    return RT

def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    for obj in scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def normalize_scene():
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        None
    """
    if len(list(get_scene_root_objects())) > 1:
        print('we have more than one root objects!!')
        # create an empty object to be used as a parent for all root objects
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)

        # parent all root objects to the empty object
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    dxyz = bbox_max - bbox_min
    dist = np.sqrt(dxyz[0]**2+ dxyz[1]**2+dxyz[2]**2)
    scale = 1 / dist
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    # unparent the camera
    bpy.data.objects["Camera"].parent = None
    return scale, offset

def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


def render_and_save(view_id, object_uid, len_val, azimuth, elevation, distance, ortho=False):
    # print(view_id)
    # render the image
    render_path = os.path.join(args.output_dir, object_uid, 'image', f"{view_id:03d}.png")
    scene.render.filepath = render_path

    if not ortho:
        cam.data.lens = len_val

    # depth_map_node.inputs[1].default_value = distance - 1
    # depth_map_node.inputs[2].default_value = distance + 1
    # depth_file_output.base_path = os.path.join(args.output_dir, object_uid, 'depth')

    # depth_file_output.file_slots[0].path = f"{view_id:03d}"
    normal_file_output.file_slots[0].path = f"{view_id:03d}"

    if not os.path.exists(os.path.join(args.output_dir, object_uid, 'image', f"{view_id:03d}.png")):
        bpy.ops.render.render(write_still=True)

    # if os.path.exists(os.path.join(args.output_dir, object_uid, 'depth', f"{view_id:03d}0001.exr")):
    #     os.rename(os.path.join(args.output_dir, object_uid, 'depth', f"{view_id:03d}0001.exr"),
    #               os.path.join(args.output_dir, object_uid, 'depth', f"{view_id:03d}.exr"))

    if os.path.exists(os.path.join(args.output_dir, object_uid, 'normal', f"{view_id:03d}0001.exr")):
        normal = cv2.imread(os.path.join(args.output_dir, object_uid, 'normal', f"{view_id:03d}0001.exr"), cv2.IMREAD_UNCHANGED)
        normal_unit16 = (normal * 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(args.output_dir, object_uid, 'normal', f"{view_id:03d}.png"), normal_unit16)
        os.remove(os.path.join(args.output_dir, object_uid, 'normal', f"{view_id:03d}0001.exr"))

    # save camera KRT matrix
    if ortho:
        K = get_calibration_matrix_K_from_blender_for_ortho(cam.data, ortho_scale=cam.data.ortho_scale)
    else:
        K = get_calibration_matrix_K_from_blender(cam.data)

    RT = get_3x4_RT_matrix_from_blender(cam)
    para_path = os.path.join(args.output_dir, object_uid, 'camera', f"{view_id:03d}.npy")
    # np.save(RT_path, RT)
    paras = {}
    paras['intrinsic'] = np.array(K, np.float32)
    paras['extrinsic'] = np.array(RT, np.float32)
    paras['fov'] = cam.data.angle
    paras['azimuth'] = azimuth
    paras['elevation'] = elevation
    paras['distance'] = distance
    paras['focal'] = cam.data.lens
    paras['sensor_width'] = cam.data.sensor_width
    paras['near'] = distance - 1
    paras['far'] = distance + 1
    paras['camera'] = 'persp' if not ortho else 'ortho'
    np.save(para_path, paras)

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    object_uid = os.path.basename(object_file).split(".")[0]
    # if we already render this object, we skip it
    if os.path.exists(os.path.join(args.output_dir, object_uid, 'camera', '64.npy')): return
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, object_uid, 'camera'), exist_ok=True)

    reset_scene()
    load_object(object_file)
    
    lights = [obj for obj in bpy.context.scene.objects if obj.type == 'LIGHT']
    for light in lights:
        bpy.data.objects.remove(light, do_unlink=True)
    
#     bproc.init()
    
    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes['Background']
    env_light = 0.5
    back_node.inputs['Color'].default_value = Vector([env_light, env_light, env_light, 1.0])
    back_node.inputs['Strength'].default_value = 1.0
    
    #Make light just directional, disable shadows.
    light_data = bpy.data.lights.new(name=f'Light', type='SUN')
    light = bpy.data.objects.new(name=f'Light', object_data=light_data)
    bpy.context.collection.objects.link(light)
    light = bpy.data.lights['Light']
    light.use_shadow = False
    # Possibly disable specular shading:
    light.specular_factor = 1.0
    light.energy = 5.0

    #Add another light source so stuff facing away from light is not completely dark
    light_data = bpy.data.lights.new(name=f'Light2', type='SUN')
    light = bpy.data.objects.new(name=f'Light2', object_data=light_data)
    bpy.context.collection.objects.link(light)
    light2 = bpy.data.lights['Light2']
    light2.use_shadow = False
    light2.specular_factor = 1.0
    light2.energy = 3 #0.015
    bpy.data.objects['Light2'].rotation_euler = bpy.data.objects['Light2'].rotation_euler
    bpy.data.objects['Light2'].rotation_euler[0] += 180

    #Add another light source so stuff facing away from light is not completely dark
    light_data = bpy.data.lights.new(name=f'Light3', type='SUN')
    light = bpy.data.objects.new(name=f'Light3', object_data=light_data)
    bpy.context.collection.objects.link(light)
    light3 = bpy.data.lights['Light3']
    light3.use_shadow = False
    light3.specular_factor = 1.0
    light3.energy = 3 #0.015
    bpy.data.objects['Light3'].rotation_euler = bpy.data.objects['Light3'].rotation_euler
    bpy.data.objects['Light3'].rotation_euler[0] += 90

    #Add another light source so stuff facing away from light is not completely dark
    light_data = bpy.data.lights.new(name=f'Light4', type='SUN')
    light = bpy.data.objects.new(name=f'Light4', object_data=light_data)
    bpy.context.collection.objects.link(light)
    light4 = bpy.data.lights['Light4']
    light4.use_shadow = False
    light4.specular_factor = 1.0
    light4.energy = 3 #0.015
    bpy.data.objects['Light4'].rotation_euler = bpy.data.objects['Light4'].rotation_euler
    bpy.data.objects['Light4'].rotation_euler[0] += -90
    
    scale, offset = normalize_scene()
    np.save(os.path.join(args.output_dir, object_uid, 'meta.npy'), np.asarray([scale, offset[0], offset[1], offset[1]],np.float32))

    try:
        # some objects' normals are affected by textures
        mesh_objects = [obj for obj in scene_meshes()]
        main_bsdf_name = 'BsdfPrincipled'
        normal_name = 'Normal'
        for obj in mesh_objects:
            for mat in obj.data.materials:
                for node in mat.node_tree.nodes:
                    if main_bsdf_name in node.bl_idname:
                        principled_bsdf = node
                        # remove links, we don't want add normal textures
                        if principled_bsdf.inputs[normal_name].links:
                            mat.node_tree.links.remove(principled_bsdf.inputs[normal_name].links[0])
    except:
        print("don't know why")
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    subject_width = 1.0
    lens = [35, 50, 85, 105, 135]
    
    normal_file_output.base_path = os.path.join(args.output_dir, object_uid, 'normal')
    
    elevation = random.uniform(-20, 40)
    for i in range(args.num_images):
        # change the camera to orthogonal
        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = subject_width
        distance = 3
        azimuth = i * 360 / args.num_images
        bpy.context.view_layer.update()
        set_camera_mvdream(azimuth, 0, distance)
        render_and_save(i * (args.random_images+1), object_uid, -1, azimuth, 0, distance, ortho=True)
        
       
        bpy.context.view_layer.update()
        _ = set_camera_mvdream(azimuth, elevation, distance)
        render_and_save(i * (args.random_images+1) + args.random_images, object_uid, -1, azimuth, elevation, distance, ortho=True)
        
        # change the camera to perspective
        cam.data.type = 'PERSP'
        for j in range(args.random_images-1):
            len_val = random.choice(lens)
            # elevation = random.uniform(elevation_range[0], elevation_range[1])
            distance = subject_width * len_val / cam.data.sensor_width

            bpy.context.view_layer.update()
            _ = set_camera_mvdream(azimuth, elevation, distance)
            render_and_save(i * (args.random_images+1) + j + 1, object_uid, len_val, azimuth, elevation, distance, ortho=False)
            

if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
