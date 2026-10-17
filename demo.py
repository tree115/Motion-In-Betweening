"""Export a reproducible four-way CMIB comparison for Blender (no renderer needed)."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cmib.lafan1 import extract, utils
from cmib.model.network import TransformerModel
from cmib.model.preprocess import lerp_input_repr, slerp_input_repr, replace_constant
from cmib.model.skeleton import (
    Skeleton,
    joint_names,
    sk_offsets,
    sk_parents,
    sk_joints_to_remove,
)


def skeleton():
    sk = Skeleton(sk_offsets, sk_parents, device="cpu")
    sk.remove_joints(sk_joints_to_remove)
    return sk


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_motion(args, ckpt, sk):
    first, last = int(ckpt["from_idx"]), int(ckpt["target_idx"])
    if first < 9 or last <= first:
        raise ValueError(
            "Expected a LAFAN checkpoint with from_idx >= 9 and target_idx > from_idx"
        )
    if args.bvh:
        anim = extract.read_bvh(str(args.bvh))
        # Match official window preprocessing: center XZ and face +X at frame 9.
        window = args.window or (int(ckpt["horizon"]) - 1 + 10)
        if window < last + 1:
            raise ValueError("--window must include checkpoint target_idx")
        pos = anim.pos[args.start : args.start + window].copy()[None]
        q = anim.quats[args.start : args.start + window].copy()[None]
        if pos.shape[1] != window:
            raise ValueError("BVH is too short for --start and checkpoint target_idx")
        pos[:, :, 0, 0] -= pos[:, :, 0, 0].mean(axis=1, keepdims=True)
        pos[:, :, 0, 2] -= pos[:, :, 0, 2].mean(axis=1, keepdims=True)
        pos, q = utils.rotate_at_frame(pos, q, anim.parents, n_past=10)
        if q.shape[2] == 27:
            keep = [i for i in range(27) if i not in sk_joints_to_remove]
            q = q[:, :, keep]
            offsets = anim.offsets[keep]
            if list(anim.parents) != sk_parents:
                raise ValueError(
                    "27-joint BVH hierarchy is not the expected LAFAN skeleton"
                )
        elif q.shape[2] != 22 or not np.array_equal(anim.parents, sk.parents()):
            raise ValueError(
                "Only the standard 22/27-joint LAFAN hierarchy is supported"
            )
        else:
            offsets = anim.offsets
        if not np.allclose(offsets[1:], sk.offsets().numpy()[1:], atol=0.1):
            raise ValueError(
                "BVH bone offsets do not match LAFAN centimeters; retarget/rescale first"
            )
        root = pos[:, first : last + 1, 0]
        q = q[:, first : last + 1]
    else:
        if not args.trust_pickle_data:
            raise ValueError(
                "Processed data is pickle: use --trust-pickle-data only for your own/trusted file"
            )
        with open(args.processed, "rb") as f:
            data = pickle.load(f)
        if not 0 <= args.sample < len(data["local_q"]):
            raise ValueError("--sample is outside this processed dataset")
        q = data["local_q"][args.sample : args.sample + 1, first : last + 1]
        root = data["root_p"][args.sample : args.sample + 1, first : last + 1]
    q, root = torch.as_tensor(q, dtype=torch.float32), torch.as_tensor(
        root, dtype=torch.float32
    )
    if q.shape != (1, last - first + 1, 22, 4) or root.shape != (
        1,
        last - first + 1,
        3,
    ):
        raise ValueError(
            "Motion shape does not match checkpoint; use the matching processed window"
        )
    if (
        not torch.isfinite(q).all()
        or not torch.isfinite(root).all()
        or (q.norm(dim=-1) < 1e-8).any()
    ):
        raise ValueError("Input motion contains non-finite values or zero quaternions")
    return sk.forward_kinematics_with_rotation(F.normalize(q, dim=-1), root)


def interpolate(pose, anchor, mode):
    if mode == "constant":
        return replace_constant(pose, anchor)
    if mode != "slerp":
        raise ValueError(f"Unsupported checkpoint interpolation: {mode}")
    return torch.cat(
        [
            lerp_input_repr(pose[..., :66], anchor),
            slerp_input_repr(pose[..., 66:].clone(), anchor),
        ],
        dim=-1,
    )


def predict(model, pose, label, anchor, mode, sk, device):
    x = interpolate(pose, anchor, mode).permute(1, 0, 2).to(device)
    mask = torch.zeros(x.shape[0] + 1, x.shape[0] + 1, dtype=torch.bool, device=device)
    code = torch.tensor([[label]], dtype=torch.long, device=device)
    with torch.inference_mode():
        out, _ = model(x, mask, code)
        out = out[1:].permute(1, 0, 2).cpu()
    if not torch.isfinite(out).all():
        raise ValueError("Checkpoint produced non-finite predictions")
    pos = out[..., :66].reshape(1, -1, 22, 3)
    # Same bone-length projection as the official run_cmib.py.
    pos = sk.convert_to_global_pos(sk.convert_to_unit_offset_mat(pos))
    rot = out[..., 66:].reshape(1, -1, 22, 4)
    if (rot.norm(dim=-1) < 1e-8).any():
        raise ValueError("Checkpoint produced zero quaternions")
    rot = F.normalize(rot, dim=-1)
    # These are explicit visualization constraints, not raw benchmark predictions.
    for t in sorted({0, pose.shape[1] - 1, anchor}):
        pos[:, t] = pose[:, t, :66].reshape(1, 22, 3)
        rot[:, t] = pose[:, t, 66:].reshape(1, 22, 4)
    return pos[0].numpy(), rot[0].numpy()


def export_bvh(path, pos, global_rot, sk, fps):
    """Rotation-driven BVH; its FK may differ from the model's position head."""
    from scipy.spatial.transform import Rotation

    parents = sk.parents()
    local_q, _ = utils.quat_ik(global_rot, pos, parents)
    # scipy expects XYZW; the model and LAFAN use WXYZ.
    euler = Rotation.from_quat(local_q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_euler(
        "ZYX", degrees=True
    )
    euler = euler.reshape(len(pos), 22, 3)
    lines, order = ["HIERARCHY"], []
    offsets = sk.offsets().numpy()

    def joint(i, depth):
        tab = "  " * depth
        lines.extend(
            [f'{tab}{"ROOT" if i == 0 else "JOINT"} {joint_names[i]}', tab + "{"]
        )
        off = np.zeros(3) if i == 0 else offsets[i]
        lines.append(tab + "  OFFSET " + " ".join(f"{v:.6f}" for v in off))
        lines.append(
            tab
            + (
                "  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation"
                if i == 0
                else "  CHANNELS 3 Zrotation Yrotation Xrotation"
            )
        )
        order.append(i)
        children = np.flatnonzero(parents == i)
        for child in children:
            joint(int(child), depth + 1)
        if len(children) == 0:
            lines.extend(
                [tab + "  End Site", tab + "  {", tab + "    OFFSET 0 0 0", tab + "  }"]
            )
        lines.append(tab + "}")

    joint(0, 0)
    lines.extend(["MOTION", f"Frames: {len(pos)}", f"Frame Time: {1 / fps:.9f}"])
    for t in range(len(pos)):
        values = list(pos[t, 0]) + list(euler[t, order].flatten())
        lines.append(" ".join(f"{v:.6f}" for v in values))
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--classes",
        type=Path,
        required=True,
        help="Matching le_classes_.npy from training",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--bvh", type=Path, help="One standard LAFAN BVH, centimeters, Y up"
    )
    src.add_argument("--processed", type=Path, help="processed_test_data.pkl")
    p.add_argument("--trust-pickle-data", action="store_true")
    p.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow legacy pickle checkpoint loading (trusted files only)",
    )
    p.add_argument("--start", type=int, default=0, help="BVH window start (0-based)")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument(
        "--window",
        type=int,
        help="BVH preprocessing window; default matches official inference (motion horizon + 10)",
    )
    p.add_argument(
        "--source-label",
        required=True,
        help="Actual action of input clip; used for base and pose panels",
    )
    p.add_argument(
        "--semantic-label",
        required=True,
        help="Target action in le_classes_.npy, e.g. dance",
    )
    p.add_argument(
        "--anchor",
        type=int,
        default=None,
        help="0-based inside predicted horizon; default midpoint",
    )
    p.add_argument(
        "--anchor-offset",
        type=float,
        nargs=3,
        default=[0, 0, 0],
        metavar=("X", "Y", "Z"),
        help="Translate entire anchor pose in cm; e.g. 0 -15 0",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--fps", type=int, default=30, help="LAFAN1 native playback rate")
    p.add_argument("--output", type=Path, default=Path("outputs/demo"))
    p.add_argument("--export-bvh", action="store_true")
    a = p.parse_args()
    if a.start < 0 or a.sample < 0 or a.fps <= 0:
        p.error("start/sample must be nonnegative and fps positive")
    torch.set_num_threads(4)
    ckpt = torch.load(
        a.checkpoint, map_location="cpu", weights_only=not a.trust_checkpoint
    )
    state = ckpt["transformer_encoder_state_dict"]
    classes = np.load(a.classes, allow_pickle=False).astype(str).tolist()
    num_labels, width = state["cond_emb.weight"].shape
    if len(classes) != num_labels or len(set(classes)) != len(classes):
        raise ValueError("classes must exactly match the checkpoint embedding table")
    for label in (a.source_label, a.semantic_label):
        if label not in classes:
            p.error(f"Unknown action {label!r}; available: {classes}")
    sk = skeleton()
    pos, rot = load_motion(a, ckpt, sk)
    steps = pos.shape[1]
    if int(ckpt["horizon"]) != steps + 1 or width != 154:
        raise ValueError(
            "Checkpoint horizon/d_model is incompatible with this LAFAN motion"
        )
    anchor = steps // 2 if a.anchor is None else a.anchor
    if not 0 < anchor < steps - 1:
        p.error("anchor must lie strictly between start and end poses")
    model = TransformerModel(
        seq_len=ckpt["horizon"],
        d_model=width,
        nhead=ckpt["nhead"],
        d_hid=ckpt["d_hid"],
        nlayers=ckpt["nlayers"],
        dropout=0.05,
        out_dim=154,
        num_labels=num_labels,
    ).to(a.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    pose = torch.cat([pos.flatten(2), rot.flatten(2)], dim=-1)
    constrained = pose.clone()
    constrained[:, anchor, :66] += torch.tensor(a.anchor_offset).repeat(22)
    mode = ckpt["interpolation"]
    base = predict(model, pose, classes.index(a.source_label), 0, mode, sk, a.device)
    guided = predict(
        model, constrained, classes.index(a.source_label), anchor, mode, sk, a.device
    )
    semantic = predict(
        model, pose, classes.index(a.semantic_label), 0, mode, sk, a.device
    )
    names = ["ground_truth", "inbetween", "pose_conditioned", "semantic_conditioned"]
    positions = np.stack([pos[0].numpy(), base[0], guided[0], semantic[0]])
    rotations = np.stack([rot[0].numpy(), base[1], guided[1], semantic[1]])
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.output / "motion.npz",
        positions=positions,
        rotations=rotations,
        parents=sk.parents(),
        joint_names=np.array(joint_names),
        panels=np.array(names),
        fps=a.fps,
        anchor=anchor,
        unit_scale=0.01,
        source_label=a.source_label,
        semantic_label=a.semantic_label,
        is_test_fixture=bool(ckpt.get("is_test_fixture", False)),
    )
    metadata = dict(
        checkpoint=str(a.checkpoint.resolve()),
        checkpoint_sha256=sha256(a.checkpoint),
        classes=classes,
        classes_sha256=sha256(a.classes),
        source_label=a.source_label,
        semantic_label=a.semantic_label,
        input=str((a.bvh or a.processed).resolve()),
        input_sha256=sha256(a.bvh or a.processed),
        start=a.start,
        sample=a.sample,
        bvh_window=a.window or (int(ckpt["horizon"]) - 1 + 10),
        anchor=anchor,
        anchor_offset=a.anchor_offset,
        from_idx=ckpt["from_idx"],
        target_idx=ckpt["target_idx"],
        frames=steps,
        fps=a.fps,
        panels=names,
        is_test_fixture=bool(ckpt.get("is_test_fixture", False)),
        device=a.device,
        torch_version=str(torch.__version__),
        postprocessing="official bone-length projection; clamp start/end and supplied pose anchor; no smoothing/foot-lock",
        bvh_note="BVH uses rotation-head FK, which can differ from position-head NPZ rendering. Not benchmark metrics.",
    )
    (a.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if a.export_bvh:
        for i, name in enumerate(names):
            export_bvh(a.output / f"{name}.bvh", positions[i], rotations[i], sk, a.fps)
    print(f'Exported {steps} frames x 4 panels to {a.output / "motion.npz"}')


if __name__ == "__main__":
    main()
