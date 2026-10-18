"""Adapt CMIB LAFAN positions for Diffusion-Motion-Inbetweening SMPLify."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np


# CMIB's standard 22-joint LAFAN order (see mib/model/skeleton.py).
LAFAN_JOINTS = [
    "Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToe", "Spine",
    "Spine1", "Spine2", "Neck", "Head", "LeftShoulder", "LeftArm",
    "LeftForeArm", "LeftHand", "RightShoulder", "RightArm",
    "RightForeArm", "RightHand",
]


def humanml3d_joints(lafan_positions):
    """Map [frames, 22, 3] LAFAN XYZ (cm, Y-up) to HumanML3D joint order."""
    if lafan_positions.ndim != 3 or lafan_positions.shape[1:] != (22, 3):
        raise ValueError("Expected LAFAN positions[frames, 22, 3]")
    p = lafan_positions
    # HumanML3D order: pelvis, hips, spine chain, legs/feet, neck/collars,
    # head, shoulders, elbows, and wrists. Collar joints are interpolated.
    mapped = np.stack(
        [
            p[:, 0],                         # 0 pelvis
            p[:, 1],                         # 1 left hip
            p[:, 5],                         # 2 right hip
            p[:, 9],                         # 3 spine
            p[:, 2],                         # 4 left knee
            p[:, 6],                         # 5 right knee
            p[:, 10],                        # 6 upper spine
            p[:, 3],                         # 7 left ankle
            p[:, 7],                         # 8 right ankle
            p[:, 11],                        # 9 chest
            p[:, 4],                         # 10 left toe
            p[:, 8],                         # 11 right toe
            p[:, 12],                        # 12 neck
            (p[:, 12] + p[:, 14]) * 0.5,    # 13 left collar
            (p[:, 12] + p[:, 18]) * 0.5,    # 14 right collar
            p[:, 13],                        # 15 head
            p[:, 14],                        # 16 left shoulder
            p[:, 18],                        # 17 right shoulder
            p[:, 15],                        # 18 left elbow
            p[:, 19],                        # 19 right elbow
            p[:, 16],                        # 20 left wrist
            p[:, 20],                        # 21 right wrist
        ],
        axis=1,
    )
    # CMIB exports centimeters; SMPLify's SMPL model and CondMDI use meters.
    return (mapped * 0.01).astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True, help="CMIB motion.npz")
    parser.add_argument(
        "--render-dir",
        type=Path,
        required=True,
        help="Output directory from tools/render_video.py",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for SMPLify inputs"
    )
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)

    with np.load(args.motion, allow_pickle=False) as data:
        positions = data["positions"].copy()
        unit_scale = float(data["unit_scale"])
        fps = int(data["fps"])
        names = data["joint_names"].astype(str).tolist()
        panels = data["panels"].astype(str).tolist()
        fixture = bool(data["is_test_fixture"]) if "is_test_fixture" in data else False

    if names != LAFAN_JOINTS:
        raise ValueError("This adapter requires the standard CMIB 22-joint LAFAN order")
    if positions.ndim != 4 or positions.shape[0] != 4 or positions.shape[2:] != (22, 3):
        raise ValueError("Expected positions[4, frames, 22, 3]")
    if not np.isfinite(positions).all() or not np.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("Motion positions and unit_scale must be finite and valid")
    if len(panels) != 4:
        raise ValueError("Expected the four standard CMIB motion panels")
    panel_videos = [args.render_dir / f"panel_{i}" / f"panel_{i}.mp4" for i in range(4)]
    missing = [str(path) for path in panel_videos if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing rendered CMIB panel videos: {missing}")

    # Keep the same fixed-ground transform used by the Blender preview. Input
    # remains Y-up for joints2smpl; only scale centimeters to SMPL meters.
    xyz = humanml3d_joints(positions.reshape(-1, 22, 3) * unit_scale).reshape(
        positions.shape[0], positions.shape[1], 22, 3
    )
    lengths = np.full(4, xyz.shape[1], dtype=np.int64)
    results = {
        "motion": xyz.transpose(0, 2, 3, 1),  # [samples=4, 22, 3, T]
        "text": np.asarray([f"CMIB {name}" for name in panels], dtype=object),
        "lengths": lengths,
        "num_samples": 4,
        "num_repetitions": 1,
    }
    np.save(args.output / "results.npy", results)
    (args.output / "README.txt").write_text(
        "Generated for setarehc/diffusion-motion-inbetweening visualize.render_mesh.\n"
        "Joint positions were mapped from CMIB LAFAN 22-joint order to HumanML3D order.\n"
        "Coordinates are Y-up and converted from centimeters to meters.\n"
        f"fps={fps}; frames={xyz.shape[1]}; synthetic_test_fixture={fixture}\n"
        "Use the sampleXX_rep00.mp4 files below as --input_path. Keep each file\n"
        "beside results.npy so render_mesh can find the corresponding motion array.\n"
    )

    for i, source in enumerate(panel_videos):
        shutil.copy2(source, args.output / f"sample{i:02d}_rep00.mp4")

    metadata = {
        "source_motion": str(args.motion.resolve()),
        "source_render_dir": str(args.render_dir.resolve()),
        "input_joint_order": LAFAN_JOINTS,
        "output_joint_order": [
            "pelvis", "left_hip", "right_hip", "spine", "left_knee", "right_knee",
            "spine1", "left_ankle", "right_ankle", "spine2", "left_toe", "right_toe",
            "neck", "left_collar", "right_collar", "head", "left_shoulder",
            "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        ],
        "coordinate_scale_to_meters": unit_scale * 0.01,
        "fps": fps,
        "frames": int(xyz.shape[1]),
        "panels": panels,
        "synthetic_test_fixture": fixture,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote SMPLify-compatible input: {args.output / 'results.npy'}")
    print("Run visualize.render_mesh from the Diffusion-Motion-Inbetweening checkout.")


if __name__ == "__main__":
    main()
