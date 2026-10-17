"""Run with blender -b --python tools/blender_scene.py -- [options]."""

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


_SPHERE_MESHES = {}


def material(name, color, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = 0.35
    return mat


def unit_sphere_mesh(mat):
    """Reuse one low-poly sphere mesh per material instead of spawning operators."""
    if mat.name in _SPHERE_MESHES:
        return _SPHERE_MESHES[mat.name]
    segments, rings = 16, 10
    vertices = []
    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        for segment in range(segments):
            theta = 2 * math.pi * segment / segments
            vertices.append(
                (
                    math.sin(phi) * math.cos(theta),
                    math.sin(phi) * math.sin(theta),
                    math.cos(phi),
                )
            )
    faces = []
    for ring in range(rings):
        for segment in range(segments):
            a = ring * segments + segment
            b = ring * segments + (segment + 1) % segments
            c = (ring + 1) * segments + (segment + 1) % segments
            d = (ring + 1) * segments + segment
            faces.append((a, b, c, d))
    mesh = bpy.data.meshes.new(f"SharedSphere_{mat.name}")
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.append(mat)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    mesh.update()
    _SPHERE_MESHES[mat.name] = mesh
    return mesh


def sphere_object(name, scale, mat):
    obj = bpy.data.objects.new(name, unit_sphere_mesh(mat))
    bpy.context.collection.objects.link(obj)
    obj.scale = scale
    return obj


def sphere(name, radius, mat):
    return sphere_object(name, (radius, radius, radius), mat)


def ellipsoid(name, scale, mat):
    return sphere_object(name, scale, mat)


def orient_segment(obj, start, end, radius, length_factor=0.58):
    delta = Vector(end) - Vector(start)
    obj.location = (Vector(start) + Vector(end)) / 2
    obj.scale = (radius[0], radius[1], max(delta.length * length_factor, radius[0]))
    if delta.length > 1e-7:
        obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(delta)


def mannequin(points, parents, mat, prefix):
    """Build a smooth, stylized human figure from the 22 LAFAN joints."""
    parts = []
    for j, point in enumerate(points):
        radius = 0.092 if j == 13 else (0.068 if j in (0, 9, 10, 11, 12) else 0.042)
        obj = sphere(f"{prefix}_joint_{j}", radius, mat)
        obj.location = point
        if j == 13:
            obj.scale = (radius * 0.88, radius * 0.9, radius * 1.2)
        parts.append((obj, "point", j, None, None))
        if parents[j] >= 0:
            radii = {
                1: (0.073, 0.069), 2: (0.052, 0.050),
                3: (0.052, 0.068), 4: (0.046, 0.035),
                5: (0.073, 0.069), 6: (0.052, 0.050),
                7: (0.052, 0.068), 8: (0.046, 0.035),
                9: (0.105, 0.092), 10: (0.125, 0.105),
                11: (0.145, 0.115), 12: (0.045, 0.045),
                13: (0.078, 0.082), 14: (0.068, 0.065),
                15: (0.049, 0.046), 16: (0.040, 0.038),
                17: (0.040, 0.034), 18: (0.068, 0.065),
                19: (0.049, 0.046), 20: (0.040, 0.038),
                21: (0.040, 0.034),
            }
            radius = radii[j]
            length_factor = 0.72 if j in (3, 4, 7, 8, 13) else 0.58
            bone = ellipsoid(f"{prefix}_bone_{j}", (*radius, 0.1), mat)
            bone.rotation_mode = "QUATERNION"
            orient_segment(
                bone, points[parents[j]], point, radius, length_factor
            )
            parts.append((bone, "segment", parents[j], j, (radius, length_factor)))

    # Fill the gaps in the sparse LAFAN rig with a pelvis, rib cage and chest.
    # These stay attached to the same joints as the benchmark skeleton.
    pelvis = ellipsoid(f"{prefix}_pelvis", (0.145, 0.105, 0.105), mat)
    pelvis.location = points[0]
    parts.append((pelvis, "point", 0, None, None))
    torso = ellipsoid(f"{prefix}_torso", (0.125, 0.095, 0.1), mat)
    torso.rotation_mode = "QUATERNION"
    orient_segment(torso, points[0], points[11], (0.125, 0.095), 0.72)
    parts.append((torso, "segment", 0, 11, ((0.125, 0.095), 0.72)))
    chest = ellipsoid(f"{prefix}_chest", (0.155, 0.105, 0.1), mat)
    chest.location = points[11]
    parts.append((chest, "point", 11, None, None))
    return parts


def pose_mannequin(parts, points):
    for obj, kind, start, end, shape in parts:
        if kind == "point":
            obj.location = points[start]
        else:
            radius, length_factor = shape
            orient_segment(obj, points[start], points[end], radius, length_factor)


def aim(obj, target):
    obj.rotation_euler = (
        (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--panel", type=int, default=0, choices=range(4))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument(
        "--engine", choices=["CYCLES", "BLENDER_EEVEE_NEXT"], default="CYCLES"
    )
    p.add_argument(
        "--ghosts",
        type=int,
        default=0,
        help="Static sampled poses, e.g. 7; 0 = animated mannequin",
    )
    p.add_argument(
        "--max-frames", type=int, default=None, help="Preview only: truncate animation"
    )
    p.add_argument("--save-only", action="store_true")
    args = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    with np.load(args.motion, allow_pickle=False) as data:
        pos = data["positions"].copy() * float(data["unit_scale"])
        parents = data["parents"].astype(int)
        fps = int(data["fps"])
        anchor = int(data["anchor"])
    if (
        pos.ndim != 4
        or pos.shape[0] != 4
        or pos.shape[2:] != (22, 3)
        or not np.isfinite(pos).all()
    ):
        raise ValueError("Expected finite positions[4, frames, 22, 3]")
    if (
        len(parents) != 22
        or parents[0] != -1
        or any(not 0 <= parents[j] < j for j in range(1, 22))
    ):
        raise ValueError("Invalid skeleton hierarchy")
    # LAFAN Y-up centimeters -> Blender Z-up meters; preserve handedness.
    pos = pos[..., [0, 2, 1]]
    pos[..., 1] *= -1
    # One fixed ground offset, based on GT: no per-frame grounding or foot lock.
    pos[..., 2] -= pos[0, ..., 2].min() - 0.045
    # Shared framing computed across all panels, so movement remains comparable.
    bounds_min, bounds_max = pos.min(axis=(0, 1, 2)), pos.max(axis=(0, 1, 2))
    center = (bounds_min + bounds_max) / 2
    radius = max(float(np.linalg.norm(bounds_max - bounds_min)) / 2, 1.4)
    points = pos[args.panel]
    start_pose, target_pose = points[0].copy(), points[-1].copy()
    anchor_pose = points[anchor].copy()
    if args.max_frames:
        if args.max_frames < 1:
            p.error("--max-frames must be positive")
        points = points[: args.max_frames]
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = args.engine
    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = (
            False  # portable to Blender builds without OpenImageDenoise
        )
        scene.cycles.device = "CPU"
    scene.render.resolution_x, scene.render.resolution_y = args.width, args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = fps
    scene.frame_start, scene.frame_end = 1, len(points)
    scene.world.color = (0.25, 0.25, 0.25)
    scene.view_settings.view_transform = "AgX"
    body = material("Graphite", (0.15, 0.19, 0.24), metallic=0.3)
    red = material("Start", (0.85, 0.045, 0.04), metallic=0.15)
    blue = material("Target", (0.035, 0.11, 0.85), metallic=0.15)
    green = material("Pose anchor", (0.08, 0.65, 0.10), metallic=0.15)
    # Infinite-looking checker floor, with real shadows.
    floor = bpy.data.materials.new("Checker floor")
    floor.use_nodes = True
    nodes, links = floor.node_tree.nodes, floor.node_tree.links
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Color1"].default_value = (0.62, 0.67, 0.72, 1)
    checker.inputs["Color2"].default_value = (0.9, 0.92, 0.94, 1)
    checker.inputs["Scale"].default_value = 1.0
    tex = nodes.new("ShaderNodeTexCoord")
    links.new(tex.outputs["Object"], checker.inputs["Vector"])
    links.new(
        checker.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"]
    )
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
    bpy.context.object.data.materials.append(floor)
    bpy.ops.object.light_add(type="AREA", location=center + np.array([3, -4, 7]))
    bpy.context.object.data.energy = 1600
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = 5
    aim(bpy.context.object, center)
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 6))
    bpy.context.object.rotation_euler = (0.35, -0.4, -0.5)
    bpy.context.object.data.energy = 1.7
    bpy.ops.object.camera_add(location=center + np.array([1.2, -2.5, 1.0]) * radius)
    camera = bpy.context.object
    aim(camera, center)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = radius * 2.5 * max(1.0, args.width / args.height)
    camera.data.lens = 50
    scene.camera = camera
    # Keep start/target ghosts visible throughout the clip. Green marks the supplied anchor.
    mannequin(start_pose, parents, red, "START")
    mannequin(target_pose, parents, blue, "TARGET")
    if args.panel == 2:
        mannequin(anchor_pose, parents, green, "ANCHOR")
    if args.ghosts:
        if args.ghosts < 2:
            p.error("--ghosts must be at least 2")
        for i in sorted(
            set(np.linspace(0, len(points) - 1, args.ghosts).astype(int)[1:-1])
        ):
            if args.panel == 2 and i == anchor:
                continue
            mannequin(points[i], parents, body, f"TRAIL_{i}")
        scene.frame_end = 1
    else:
        parts = mannequin(points[0], parents, body, "MOTION")
        for t, pose in enumerate(points, 1):
            pose_mannequin(parts, pose)
            for obj, kind, _, _, _ in parts:
                obj.keyframe_insert(data_path="location", frame=t)
                obj.hide_render = t in (1, pos.shape[1]) or (
                    args.panel == 2 and t == anchor + 1
                )
                obj.keyframe_insert(data_path="hide_render", frame=t)
                if kind == "segment":
                    for prop in ("rotation_quaternion", "scale"):
                        obj.keyframe_insert(data_path=prop, frame=t)
        for obj, _, _, _, _ in parts:
            if obj.animation_data and obj.animation_data.action:
                for curve in obj.animation_data.action.fcurves:
                    for key in curve.keyframe_points:
                        key.interpolation = (
                            "CONSTANT"
                            if curve.data_path == "hide_render"
                            else "LINEAR"
                        )
    scene.frame_set(1)
    scene.render.filepath = str((args.output / "frame_").resolve())
    bpy.ops.wm.save_as_mainfile(filepath=str((args.output / "scene.blend").resolve()))
    if not args.save_only:
        bpy.ops.render.render(animation=True)
    print("CMIB Blender export complete:", args.output)


if __name__ == "__main__":
    main()
