"""Regression tests for inference/export; checkpoints here are deliberately untrained."""

import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from cmib.lafan1.extract import read_bvh
from cmib.model.network import TransformerModel
from cmib.model.preprocess import slerp_input_repr
from demo import skeleton, export_bvh

ROOT = Path(__file__).resolve().parents[1]


def make_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    steps = 12
    model = TransformerModel(
        steps + 1, 154, 7, 64, 1, dropout=0, out_dim=154, num_labels=2
    )
    torch.save(
        dict(
            transformer_encoder_state_dict=model.state_dict(),
            horizon=steps + 1,
            from_idx=9,
            target_idx=steps + 8,
            nhead=7,
            d_hid=64,
            nlayers=1,
            interpolation="slerp",
            is_test_fixture=True,
        ),
        directory / "untrained.pt",
    )
    np.save(directory / "le_classes_.npy", np.array(["walk", "dance"]))
    q = np.zeros((1, steps + 9, 22, 4), dtype=np.float32)
    q[..., 0] = 1
    # Orient the otherwise X-aligned LAFAN rest skeleton upright.
    q[:, :, 0] = [0.5, 0.5, 0.5, 0.5]
    for i in range(steps + 9):
        phase = i * 0.3
        for j, angle in [
            (1, np.pi + 0.35 * np.sin(phase)),
            (5, np.pi - 0.35 * np.sin(phase)),
            (2, -0.25 * (1 + np.sin(phase))),
            (6, -0.25 * (1 - np.sin(phase))),
        ]:
            q[:, i, j] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        for j, angle in [(14, -2.1), (18, 2.1)]:
            q[:, i, j] = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
    root = np.zeros((1, steps + 9, 3), dtype=np.float32)
    root[..., 1] = 100
    root[0, :, 2] = np.linspace(0, 60, steps + 9)
    with open(directory / "fixture.pkl", "wb") as f:
        pickle.dump(dict(local_q=q, root_p=root), f)
    return [
        sys.executable,
        str(ROOT / "demo.py"),
        "--checkpoint",
        str(directory / "untrained.pt"),
        "--classes",
        str(directory / "le_classes_.npy"),
        "--processed",
        str(directory / "fixture.pkl"),
        "--trust-pickle-data",
        "--source-label",
        "walk",
        "--semantic-label",
        "dance",
        "--output",
        str(directory / "export"),
        "--export-bvh",
    ]


class DemoTests(unittest.TestCase):
    def test_slerp_shortest_path_preserves_source(self):
        q = torch.zeros(1, 5, 8)
        q[..., 0] = 1
        q[..., 4] = 1
        q[:, -1] *= -1  # same rotation, opposite quaternion hemisphere
        saved = q.clone()
        out = slerp_input_repr(q, 0).reshape(1, 5, 2, 4)
        torch.testing.assert_close(q, saved)
        torch.testing.assert_close(out.norm(dim=-1), torch.ones(1, 5, 2))
        self.assertTrue(torch.isfinite(out).all())

    def test_bvh_source(self):
        with tempfile.TemporaryDirectory() as d:
            cmd = make_fixture(d)
            with open(Path(d) / "fixture.pkl", "rb") as f:
                data = pickle.load(f)
            q = torch.from_numpy(data["local_q"])
            root = torch.from_numpy(data["root_p"])
            # Official inference uses horizon+10 frames, one more than target_idx+1.
            q = torch.cat([q, q[:, -1:]], dim=1)
            root = torch.cat([root, root[:, -1:]], dim=1)
            sk = skeleton()
            pos, rot = sk.forward_kinematics_with_rotation(q, root)
            raw = Path(d) / "input.bvh"
            export_bvh(raw, pos[0].numpy(), rot[0].numpy(), sk, 30)
            i = cmd.index("--processed")
            del cmd[i : i + 3]  # --processed path --trust-pickle-data
            subprocess.run(cmd + ["--bvh", str(raw)], cwd=ROOT, check=True)
            with np.load(Path(d) / "export/motion.npz") as exported:
                self.assertEqual(exported["positions"].shape, (4, 12, 22, 3))
                self.assertTrue(exported["is_test_fixture"])

    def test_export_constraints_and_bvh_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            cmd = make_fixture(d)
            subprocess.run(
                cmd + ["--anchor", "5", "--anchor-offset", "0", "-10", "0"],
                check=True,
                cwd=ROOT,
            )
            output = Path(d) / "export"
            with np.load(output / "motion.npz") as data:
                pos, rot = data["positions"], data["rotations"]
                self.assertEqual(pos.shape, (4, 12, 22, 3))
                self.assertTrue(np.isfinite(pos).all())
                np.testing.assert_allclose(np.linalg.norm(rot, axis=-1), 1, atol=1e-5)
                for panel in range(1, 4):
                    np.testing.assert_allclose(pos[panel, [0, -1]], pos[0, [0, -1]])
                np.testing.assert_allclose(
                    pos[2, 5], pos[0, 5] + [0, -10, 0], atol=1e-5
                )
                # Bone projection must preserve the canonical lengths, including the translated anchor.
                sk = skeleton()
                for j, parent in enumerate(sk.parents()[1:], 1):
                    np.testing.assert_allclose(
                        np.linalg.norm(pos[:, :, j] - pos[:, :, parent], axis=-1),
                        np.linalg.norm(sk.offsets()[j]),
                        atol=1e-3,
                    )
                anim = read_bvh(str(output / "ground_truth.bvh"))
                np.testing.assert_allclose(anim.pos[:, 0], pos[0, :, 0], atol=1e-4)
                self.assertEqual(anim.quats.shape, (12, 22, 4))
                # Validate quaternion rotation round trip, including Euler order.
                from cmib.lafan1.utils import quat_fk

                local_pos = np.broadcast_to(sk.offsets().numpy(), (12, 22, 3)).copy()
                local_pos[:, 0] = pos[0, :, 0]
                global_q, global_p = quat_fk(anim.quats, local_pos, sk.parents())
                np.testing.assert_allclose(
                    np.abs((global_q * rot[0]).sum(-1)), 1, atol=1e-5
                )
                np.testing.assert_allclose(global_p, pos[0], atol=1e-3)
            meta = json.loads((output / "metadata.json").read_text())
            self.assertEqual(meta["frames"], 12)
            invalid = cmd.copy()
            invalid[invalid.index("dance")] = "unknown"
            result = subprocess.run(invalid, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unknown action", result.stderr)


if __name__ == "__main__":
    unittest.main()
