# Motion-In-Betweening

Fork of [Conditional Motion In-Betweening (CMIB)](https://github.com/jihoonerd/Conditional-Motion-In-Betweening), by Jihoon Kim et al., Pattern Recognition (2022).

**[Hướng dẫn chạy lại thí nghiệm và dựng demo Blender → DEMO_VI.md](DEMO_VI.md)**

**[Mở notebook Google Colab để chạy inference và xuất demo Blender](notebooks/CMIB_Colab_Blender_Demo.ipynb)**

- `trainer.py`: train CMIB with seeded runs and the paper's AdamW settings.
- `demo.py`: checkpoint + LAFAN BVH/processed sample → four motion panels, metadata, optional BVH.
- `tools/render_video.py`: Blender + FFmpeg → four-panel comparison video or sampled-pose still, with editable `.blend` scenes and a smoother procedural mannequin.
- `tests/test_demo.py`: synthetic regression tests for export and motion constraints; these do not establish pretrained model quality.

The original paper rendered in Unity. This Blender implementation uses stylized procedural mannequins; the original character mesh/rig is not included. A paper-like human character requires a licensed skinned mesh and matching rig, as well as trained weights, the matching label table and motion data.
