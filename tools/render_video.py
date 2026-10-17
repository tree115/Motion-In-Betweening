"""Render four Blender panels and encode a labeled comparison MP4 or ghost still."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


def run(cmd, log):
    with open(log, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}); see {log}\n{Path(log).read_text()[-2500:]}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--blender", default="blender")
    p.add_argument(
        "--width",
        type=int,
        default=640,
        help="Width per panel; final video is twice this",
    )
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--engine", choices=["CYCLES", "BLENDER_EEVEE_NEXT"], default="CYCLES"
    )
    p.add_argument("--max-frames", type=int)
    p.add_argument("--ghosts", type=int, default=0)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output only in a dedicated existing render directory",
    )
    a = p.parse_args()
    if min(a.width, a.height, a.samples, a.threads) < 1 or a.width % 2 or a.height % 2:
        p.error(
            "sizes must be positive even numbers; samples and threads must be positive"
        )
    if not shutil.which(a.blender) or not shutil.which("ffmpeg"):
        p.error("Blender and FFmpeg must be installed and on PATH")
    if a.output.exists() and any(a.output.iterdir()) and not a.overwrite:
        p.error("Output directory is not empty; choose a new directory or --overwrite")
    a.output.mkdir(parents=True, exist_ok=True)
    a.motion = a.motion.resolve()
    with np.load(a.motion, allow_pickle=False) as data:
        fps = int(data["fps"])
        count = int(data["positions"].shape[1])
        fixture = bool(data["is_test_fixture"]) if "is_test_fixture" in data else False
        semantic = str(data["semantic_label"]) if "semantic_label" in data else ""
    count = min(count, a.max_frames) if a.max_frames else count
    if count < 1:
        p.error("No frames to render")
    labels = [
        "Ground truth",
        "In-betweening",
        "Pose-conditioned",
        "Semantic-conditioned",
    ]
    if fixture:
        labels = [
            "PIPELINE TEST - synthetic reference",
            "PIPELINE TEST - untrained model",
            "PIPELINE TEST - pose constraint",
            "PIPELINE TEST - semantic token",
        ]
    if semantic and not fixture:
        labels[-1] += f" ({semantic})"
    script = Path(__file__).with_name("blender_scene.py").resolve()
    videos = []
    for i, label in enumerate(labels):
        directory = a.output / f"panel_{i}"
        directory.mkdir(exist_ok=True)
        # Known generated frame paths only; never recursively remove user files.
        if a.overwrite:
            for frame in directory.glob("frame_*.png"):
                frame.unlink()
        cmd = [
            a.blender,
            "-b",
            "-t",
            str(a.threads),
            "--python-exit-code",
            "1",
            "--python",
            str(script),
            "--",
            "--motion",
            str(a.motion),
            "--panel",
            str(i),
            "--output",
            str(directory.resolve()),
            "--width",
            str(a.width),
            "--height",
            str(a.height),
            "--samples",
            str(a.samples),
            "--engine",
            a.engine,
        ]
        if a.max_frames:
            cmd += ["--max-frames", str(a.max_frames)]
        if a.ghosts:
            cmd += ["--ghosts", str(a.ghosts)]
        print(f"Rendering {label}", flush=True)
        run(cmd, a.output / f"blender_{i}.log")
        expected = 1 if a.ghosts else count
        if not all(
            (directory / f"frame_{j:04d}.png").is_file() for j in range(1, expected + 1)
        ):
            raise RuntimeError(f"Missing frames for {label}")
        caption = directory / "label.txt"
        caption.write_text(label)
        video = a.output / f"panel_{i}.mp4"
        # Filter uses a relative constant filename, avoiding escaping user paths in FFmpeg expressions.
        filter_text = f"pad=iw:ih+40:0:40:white,drawtext=textfile=label.txt:fontcolor=black:fontsize=16:x=(w-tw)/2:y=8"
        with open(a.output / f"ffmpeg_{i}.log", "w") as log:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-threads",
                    str(a.threads),
                    "-framerate",
                    str(fps),
                    "-start_number",
                    "1",
                    "-i",
                    "frame_%04d.png",
                    "-frames:v",
                    str(expected),
                    "-vf",
                    filter_text,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    "-threads",
                    str(a.threads),
                    str(video.resolve()),
                ],
                cwd=directory,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        videos.append(video)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-filter_complex_threads", "1"]
    for video in videos:
        cmd += ["-i", str(video)]
    filters = "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]"
    cmd += [
        "-filter_complex",
        filters,
        "-map",
        "[v]",
        "-frames:v",
        str(1 if a.ghosts else count),
    ]
    target = a.output / ("comparison.png" if a.ghosts else "comparison.mp4")
    cmd += ["-threads", str(a.threads)]
    if not a.ghosts:
        cmd += [
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    cmd += [str(target)]
    run(cmd, a.output / "compose.log")
    (a.output / "render.json").write_text(
        json.dumps(
            dict(
                motion=str(a.motion),
                frames=1 if a.ghosts else count,
                fps=fps,
                width=2 * a.width,
                height=2 * (a.height + 40),
                engine=a.engine,
                samples=a.samples,
                ghosts=a.ghosts,
                output=str(target),
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Done: {target}")


if __name__ == "__main__":
    main()
