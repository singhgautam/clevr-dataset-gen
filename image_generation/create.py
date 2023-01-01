from __future__ import print_function

import numpy as np
import h5py
from tqdm import tqdm
import random
import argparse

from spriteworld import renderers as spriteworld_renderers
from spriteworld.sprite import Sprite
import os
import cv2

import math
import json


import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter

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

# Settings for objects
parser.add_argument('--min_objects', default=2, type=int,
                    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=2, type=int,
                    help="The maximum number of objects to place in each scene")


# Output settings
parser.add_argument('--save_path', default='/research/projects/object_centric/gs790/sysvim/datasets/clevr-xshift-001')

parser.add_argument('--num_train', type=int, default=64000)
parser.add_argument('--num_test', type=int, default=12800)

parser.add_argument('--train_ratio_increments', type=float, default=0.2)
parser.add_argument('--test_ratio', type=float, default=0.2)


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

args = parser.parse_args()


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



def generate_binds(colors_, shapes_, sizes_, ratio=0.8):
    must = max([len(colors_), len(shapes_), len(sizes_)])

    colors = np.append(np.arange(len(colors_)), np.random.randint(low=len(colors_), size=must - len(colors_)))
    shapes = np.append(np.arange(len(shapes_)), np.random.randint(low=len(shapes_), size=must - len(shapes_)))
    sizes = np.append(np.arange(len(sizes_)), np.random.randint(low=len(sizes_), size=must - len(sizes_)))

    N = len(colors_) * len(shapes_) * len(sizes_)
    N_sample = math.floor((N - must) * ratio)

    color = np.random.choice(colors, (must, 1), replace=False)
    shape = np.random.choice(shapes, (must, 1), replace=False)
    size = np.random.choice(sizes, (must, 1), replace=False)

    binds = np.concatenate((color, shape, size), axis=-1)
    cores = binds[:]

    pool = generate_unseen_binds(colors_, shapes_, sizes_, binds, 1)

    for i in range(N_sample):
        idx = np.random.choice(np.arange(pool.shape[0]))
        sample = pool[idx]
        pool = np.delete(pool, idx, axis=0)
        binds = np.concatenate([binds, [sample]], axis=0)

    return binds, cores


def generate_unseen_binds(colors_, shapes_, sizes_, binds_, ratio=0.2):
    colors = np.arange(len(colors_))
    shapes = np.arange(len(shapes_))
    sizes = np.arange(len(sizes_))

    must = max([len(colors_), len(shapes_), len(sizes_)])

    N = len(colors_) * len(shapes_) * len(sizes_)
    N_sample = N - math.floor((N - must) * (1 - ratio)) - must

    if N_sample > N - len(binds_):  # switch to i.i.d
        print("switch to i.i.d")
        return binds_[:]

    unseen_binds_ = []

    unseen_binds_ = np.array(np.meshgrid(colors, shapes, sizes)).T.reshape(-1, 3)
    binds_ = np.array(binds_)

    rows1 = unseen_binds_.view([('', unseen_binds_.dtype)] * unseen_binds_.shape[1])
    rows2 = binds_.view([('', binds_.dtype)] * binds_.shape[1])
    unseen_binds_ = np.setdiff1d(rows1, rows2).view(unseen_binds_.dtype).reshape(-1, unseen_binds_.shape[1])

    binds = []

    for i in range(N_sample):
        idx = np.random.choice(np.arange(unseen_binds_.shape[0]))
        sample = unseen_binds_[idx]
        unseen_binds_ = np.delete(unseen_binds_, idx, axis=0)
        binds += [sample]

    return np.array(binds)


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

    # PARAMETERS
    C, H, W = 3, args.image_size, args.image_size

    RULES = ["xshift", "xswap", "colorchange"]
    SHAPES = ['SmoothCube_v2', 'Sphere', 'SmoothCylinder']
    COLORS = [
        (1., 0., 0., 1.),
        (0., 1., 0., 1.),
        (0., 0., 1., 1.),
    ]
    SIZES = [
        1.5,
        2.0,
        2.5,
    ]
    XSHIFT = 2.0

    num_shapes = len(SHAPES)

    # Composition space ratio
    test_ratio = args.test_ratio
    train_ratio = args.train_ratio
    assert train_ratio <= 1 - test_ratio, "Improper train-test split ratio."

    # PARAMETERS per TASK
    R = RULES[args.rule]

    # Binding Pairs (TRAIN)
    BINDS, CORES = generate_binds(COLORS, SHAPES, SIZES, ratio=train_ratio)  # train
    UNSEEN_BINDS = generate_unseen_binds(COLORS, SHAPES, SIZES, BINDS, ratio=test_ratio)  # test

    # print(f"Train Comp: {len(BINDS)}, Test Comp: {len(UNSEEN_BINDS)}")

    data_path = os.path.join(args.save_path, "clevr-{}-alpha-{}".format(R, train_ratio))
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    train_info = {
        "colors": COLORS,
        "shapes": SHAPES,
        "sizes": SIZES,
        "ratio": train_ratio,
        "binds": BINDS.tolist(),
        "cores": CORES.tolist(),
    }
    with open(os.path.join(train_path, "info.json"), 'w') as outfile:
        json.dump(train_info, outfile)

    test_info = {
        "colors": COLORS,
        "shapes": SHAPES,
        "sizes": SIZES,
        "ratio": test_ratio,
        "unseen_binds": UNSEEN_BINDS.tolist(),
        "cores": CORES.tolist(),
    }
    with open(os.path.join(test_path, "info.json"), 'w') as outfile:
        json.dump(test_info, outfile)

    print("Train Set : {}".format(args.num_train))
    for run in tqdm(range(args.num_train)):
        sample_path = os.path.join(train_path, "{:08d}".format(run))
        os.makedirs(sample_path, exist_ok=True)

        N = np.random.choice(np.arange(args.min_objects, args.max_objects + 1))
        bind = np.random.choice(np.arange(len(BINDS)), size=(N))
        bind = BINDS[bind]

        color = bind[:, 0]
        shape = bind[:, 1]
        size = bind[:, 2]

        x = np.random.uniform(-4, 4, (N, 2))
        while True:
            x = np.random.uniform(-4, 4, (N, 2))
            if checker(x, [SIZES[size[i]] for i in range(N)]):
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
            obj_colors.append(COLORS[color[i]])
            obj_shapes.append(SHAPES[shape[i]])
            obj_sizes.append(SIZES[size[i]])
            obj_materials.append('Rubber')

        render_scene(N, obj_positions, obj_rotations, obj_colors, obj_shapes, obj_sizes, obj_materials,
                     args,
                     output_image=os.path.join(sample_path, "source.png"),
                     output_scene=os.path.join(sample_path, "source.json"))

        if R == 'xshift':
            obj_positions = [(x + XSHIFT, y) for x,y in obj_positions]

        render_scene(N, obj_positions, obj_rotations, obj_colors, obj_shapes, obj_sizes, obj_materials,
                     args,
                     output_image=os.path.join(sample_path, "target.png"),
                     output_scene=os.path.join(sample_path, "target.json"))


    print("Test Set : {}".format(args.num_test))

    for run in tqdm(range(args.num_test)):
        sample_path = os.path.join(test_path, "{:08d}".format(run))
        os.makedirs(sample_path, exist_ok=True)

        N = np.random.choice(np.arange(args.min_objects, args.max_objects + 1))
        bind = np.random.choice(np.arange(len(UNSEEN_BINDS)), size=(N))
        bind = UNSEEN_BINDS[bind]

        color = bind[:, 0]
        shape = bind[:, 1]
        size = bind[:, 2]

        x = np.random.uniform(-4, 4, (N, 2))
        while True:
            x = np.random.uniform(-4, 4, (N, 2))
            if checker(x, [SIZES[size[i]] for i in range(N)]):
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
            obj_colors.append(COLORS[color[i]])
            obj_shapes.append(SHAPES[shape[i]])
            obj_sizes.append(SIZES[size[i]])
            obj_materials.append('rubber')

        render_scene(N, obj_positions, obj_rotations, obj_colors, obj_shapes, obj_sizes, obj_materials,
                     args,
                     output_image=os.path.join(sample_path, "source.png"),
                     output_scene=os.path.join(sample_path, "source.json"))

        if R == 'xshift':
            obj_positions = [(x + XSHIFT, y) for x, y in obj_positions]

        render_scene(N, obj_positions, obj_rotations, obj_colors, obj_shapes, obj_sizes, obj_materials,
                     args,
                     output_image=os.path.join(sample_path, "target.png"),
                     output_scene=os.path.join(sample_path, "target.json"))

    print('Dataset saved at : {}'.format(data_path))
