from __future__ import print_function

import argparse
import json
import os
import random
import sys

import numpy as np

from configs import *

"""
This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
    import bpy, bpy_extras
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import utils
    except ImportError as e:
        print("\nERROR")
        print("Running render_images.py from Blender and cannot import utils.py.")
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.78).")
        sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
                    help="Base blender file on which all scenes are based; includes " +
                         "ground plane, lights, and camera.")
parser.add_argument('--shape_dir', default='data/shapes',
                    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
                    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
                    help="Optional path to a JSON file mapping shape names to a list of " +
                         "allowed color names for that shape. This allows rendering images " +
                         "for CLEVR-CoGenT.")


# Rendering options
parser.add_argument('--use_gpu', default=1, type=int,
                    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
                         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
                         "to work.")
parser.add_argument('--width', default=128, type=int,
                    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=128, type=int,
                    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
                    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
                    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
                    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
                    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
                    help="The number of samples to use when rendering. Larger values will " +
                         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
                    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
                    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
                    help="The tile size to use for rendering. This should not affect the " +
                         "quality of the rendered image but may affect the speed; CPU-based " +
                         "rendering may achieve better performance using smaller tile sizes " +
                         "while larger tile sizes may be optimal for GPU-based rendering.")

# Object options
parser.add_argument('--min_objects', default=2, type=int,
                    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=2, type=int,
                    help="The maximum number of objects to place in each scene")

# Output options
parser.add_argument('--save_path', default='output/')
parser.add_argument('--bind_path', default='precomputed_binds/binds.json')

parser.add_argument('--num_samples', type=int, default=100)

parser.add_argument('--rule', default='xshift')

argv = utils.extract_args()
args = parser.parse_args(argv)


def render_scene(num_objects, positions, rotations, colors, shapes, sizes, materials,
                 args,
                 output_image='render.png',
                 output_scene='render.json',
                 ):

    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    utils.load_materials(args.material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    if args.use_gpu == 1:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        else:
            cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu == 1:
        bpy.context.scene.cycles.device = 'GPU'

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        'image_filename': os.path.basename(output_image),
        'objects': [],
    }

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to camera position
    if args.camera_jitter > 0:
        for i in range(3):
            bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

    camera = bpy.data.objects['Camera']

    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

    # Now add objects
    objects, blender_objects = add_objects(num_objects, positions, rotations, colors, shapes, sizes, materials, args, camera)

    # Render the scene and dump the scene data structure
    scene_struct['objects'] = objects
    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print(e)

    with open(output_scene, 'w') as f:
        json.dump(scene_struct, f, indent=2)


def add_objects(num_objects, positions, rotations, colors, shapes, sizes, materials, args, camera):
    """
    Add objects with specific attributes to the current blender scene
    """

    objects = []
    blender_objects = []
    for i in range(num_objects):

        x, y = positions[i]

        theta = rotations[i]

        size = sizes[i]

        shape = shapes[i]

        material = materials[i]

        color = colors[i]

        utils.add_object(args.shape_dir, shape, size, (x, y), theta=theta)

        obj = bpy.context.object
        blender_objects.append(obj)

        utils.add_material(material, Color=color)

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
            'shape': shape,
            'size': size,
            'material': material,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'color': color,
        })

    return objects, blender_objects


def checker(positions, sizes):
    N = positions.shape[0]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dist = np.sqrt(np.sum((positions[i] - positions[j]) * (positions[i] - positions[j])))
            min_dist = sizes[i] + sizes[j]
            if dist < min_dist:
                return False
    return True


if __name__ == '__main__':

    # Load binds
    with open(args.bind_path) as binds_file:
        binds = json.load(binds_file)
        binds = binds["binds"]

    # Dump Set
    data_path = args.save_path
    os.makedirs(data_path, exist_ok=True)

    for run in range(args.num_samples):
        sample_label = random.randint(0, 10 ** 16)
        sample_path = os.path.join(data_path, "{:018d}".format(sample_label))
        os.makedirs(sample_path, exist_ok=True)

        N = np.random.choice(np.arange(args.min_objects, args.max_objects + 1))
        object_binds = [random.choice(binds) for _ in range(N)]

        x = np.random.uniform(-3, 3, (N, 2))
        while True:
            x = np.random.uniform(-3, 3, (N, 2))
            if checker(x, [SIZES[object_binds[i][2]] for i in range(N)]):
                break

        angle = np.random.random(size=(N)) * 360.

        obj_positions = []
        obj_rotations = []
        obj_colors = []
        obj_shapes = []
        obj_sizes = []
        obj_materials = []

        for i in range(N):
            obj_positions.append((x[i, 0], x[i, 1]))
            obj_rotations.append(angle[i])
            obj_colors.append(COLORS[object_binds[i][0]])
            obj_shapes.append(SHAPES[object_binds[i][1]])
            obj_sizes.append(SIZES[object_binds[i][2]])
            obj_materials.append(MATERIALS[object_binds[i][3]])

        render_scene(N, obj_positions, obj_rotations, obj_colors, obj_shapes, obj_sizes, obj_materials,
                     args,
                     output_image=os.path.join(sample_path, "source.png"),
                     output_scene=os.path.join(sample_path, "source.json"))

        if args.rule == 'xshift':
            obj_positions = [(x + SHIFT_AMOUNT, y) for x, y in obj_positions]

        render_scene(N, obj_positions, obj_rotations, obj_colors, obj_shapes, obj_sizes, obj_materials,
                     args,
                     output_image=os.path.join(sample_path, "target.png"),
                     output_scene=os.path.join(sample_path, "target.json"))

    print('Dataset saved at : {}'.format(data_path))
