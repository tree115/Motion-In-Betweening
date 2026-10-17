# Tái hiện CMIB và dựng demo Blender

Repo này có pipeline `demo.py -> motion.npz/BVH -> Blender -> MP4`.
Paper **Conditional Motion In-betweening** (Kim et al., 2022, mục 5.2) dùng Unity để trình diễn; Blender thay được khâu trình diễn, không thay mô hình. Cảnh mặc định ở đây là mannequin procedural có thân và tay chân bo tròn, sàn ô vuông, bóng đổ, cùng camera cho bốn ô. Cấu hình render kiểm chứng trong cloud là Blender 4.3.2 + Cycles CPU; EEVEE cần backend đồ họa phù hợp và không dùng được trong cloud headless hiện tại. Không kèm nhân vật/mesh của tác giả.

## 1. Môi trường

Trong môi trường cloud hiện tại:

```bash
source /workspace/.motion-env/activate.sh
cd /workspace/Motion-In-Betweening
```

Ở máy khác, cài Python 3.12, PyTorch bản CPU/CUDA phù hợp từ https://pytorch.org/get-started/locally/, rồi:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch  # bỏ qua nếu đã cài PyTorch CUDA phù hợp
blender --version
ffmpeg -version
```

Đã kiểm thử Python 3.12 / PyTorch 2.14.1 CPU / Blender 4.3.2 / FFmpeg trong cloud. Script Blender nhắm API Blender 4.2–4.3; phiên bản 4.4+ đổi API animation có thể cần điều chỉnh. Mã gốc công bố PyTorch 1.10.1 / CUDA 11.3; môi trường hiện tại kiểm chứng pipeline trên runtime mới, chưa chứng minh metric tương đương runtime gốc. CPU đủ inference và render; nên dùng GPU cho huấn luyện toàn bộ. Không cần W&B key nếu `export WANDB_MODE=offline`. Chạy Python từ thư mục gốc repo; shim `cmib/` xử lý tên import của mã gốc dù mã nằm trong `mib/`.

## 2. Chuẩn bị checkpoint và dữ liệu thật

Cần:

- `train-XXXX.pt` từ mô hình đã huấn luyện, không phải trọng số ngẫu nhiên.
- `le_classes_.npy` **đúng của lần huấn luyện đó**, vì thứ tự label phải khớp bảng embedding. Không tự sắp xếp hay đoán thứ tự.
- Một BVH LAFAN1, hoặc `processed_test_data.pkl` tương ứng cửa sổ dữ liệu.

Nguồn checkpoint: README chính thức https://github.com/jihoonerd/Conditional-Motion-In-Betweening trỏ tới https://works.do/FCqKVjy (`trained_weights.zip`). Giải nén vào thư mục riêng, ví dụ `/workspace/assets/cmib_weights`.

Dữ liệu LAFAN1: https://github.com/ubisoft/ubisoft-laforge-animation-dataset. Theo hướng dẫn chính thức, tải bằng Git LFS; `lafan1/lafan1.zip` phải là ZIP thật, không phải pointer văn bản vài dòng. Giải nén vào thư mục dữ liệu riêng. Nếu proxy GitHub của cloud từ chối LFS, tải bằng trình duyệt ở máy bạn và đưa ZIP/BVH vào môi trường; không cần cung cấp GitHub token trong chat.

```bash
# Trên máy có Git LFS hoạt động:
git lfs install
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
# Giải nén lafan1/lafan1.zip để có thư mục BVH.
# Xem evaluate.py của dataset nếu cần chạy baseline/validation chính thức.
```

Kiểm tra label thực tế:

```bash
python -c "import numpy as np; print(np.load('/workspace/assets/cmib_weights/le_classes_.npy', allow_pickle=False))"
```

Các đường dẫn bên dưới là ví dụ; thay bằng file thật. `dance`, `walk`, `jumps` chỉ dùng nếu có trong danh sách label. Dùng Subject 5 để làm demo trên tập test LAFAN; huấn luyện dùng Subject 1–4. Dữ liệu LAFAN1 có điều khoản CC BY-NC-ND 4.0; xem license của nguồn trước khi chia sẻ tài sản/dữ liệu.

## 3. Chạy inference bốn trường hợp

```bash
python demo.py \
  --checkpoint /workspace/assets/cmib_weights/weights/train-5000.pt \
  --classes /workspace/assets/cmib_weights/le_classes_.npy \
  --bvh /workspace/assets/lafan/BVH/walk1_subject5.bvh \
  --start 400 \
  --source-label walk \
  --semantic-label dance \
  --anchor 40 \
  --output /workspace/outputs/cmib-walk \
  --export-bvh
```

- `--start` là frame đầu cửa sổ BVH (0-based). Script căn hướng ở frame 9 và trừ trung bình XZ như preprocessing gốc; mặc định cửa sổ là `checkpoint.horizon - 1 + 10`. Dùng `--window` khi checkpoint được huấn luyện với cửa sổ khác.
- `--anchor` là chỉ số **bên trong đoạn dự đoán**, không tính token semantic. Với checkpoint 80 frame, anchor 40 hợp lệ; với checkpoint 30 frame, chọn 15. Bỏ tùy chọn này để lấy điểm giữa.
- `--source-label` là action thật của clip, cho hai ô base và pose-conditioned; `--semantic-label` là action muốn điều kiện hóa. Base vẫn sử dụng semantic token của action nguồn theo kiến trúc gốc, không phải mô hình unconditional riêng.
- Anchor mặc định lấy từ GT. Có thể thêm `--anchor-offset 0 -15 0` để dịch toàn bộ pose anchor xuống 15 cm trong hệ Y-up. Đây là điều khiển thử nghiệm, không phải ground truth mới; offset quá lớn có thể vượt phân bố huấn luyện.
- Dùng `--device cuda:0` nếu có GPU; mặc định CPU.
- FPS mặc định 30 đúng LAFAN1. Muốn slow motion hãy đổi tốc độ phát ở bước dựng; không đổi FPS tùy ý khi so sánh tốc độ chuyển động.

Nếu đã có processed data:

```bash
python demo.py \
  --checkpoint /workspace/assets/cmib_weights/weights/train-5000.pt \
  --classes /workspace/assets/cmib_weights/le_classes_.npy \
  --processed /workspace/assets/lafan/processed_test_data.pkl \
  --trust-pickle-data --sample 950 \
  --source-label walk --semantic-label dance \
  --output /workspace/outputs/cmib-sample950 --export-bvh
```

`--sample` phải nằm trong tập dữ liệu và bạn cần chọn đúng `--source-label`. Processed pickle chỉ dùng với file tin cậy. PyTorch mặc định nạp checkpoint bằng `weights_only=True`; chỉ thêm `--trust-checkpoint` khi checkpoint cũ thật sự cần cơ chế pickle đầy đủ và bạn tin nguồn.

Kết quả:

- `motion.npz`: vị trí/quaternion global 4 × T × 22 khớp, WXYZ, centimeters, Y-up; file này dành cho render theo position head.
- `metadata.json`: hash checkpoint/dữ liệu/classes, label, sample/frame/anchor và các bước hậu xử lý.
- Bốn file `.bvh` nếu bật `--export-bvh`: local rotations, ZYX Euler, root translation, 30 fps.

**Lưu ý khi so sánh:** pipeline áp dụng projection độ dài xương như `run_cmib.py`, rồi gán lại start/end và anchor để minh họa ràng buộc. Không smoothing hoặc foot lock. Đây là video trình diễn có hậu xử lý; không lấy output đã clamp để công bố metric của paper. Vị trí và quaternion được dự đoán bởi hai phần output khác nhau; BVH dựa trên rotation head + FK có thể khác hình học position head trong NPZ. Foot sliding, xuyên sàn và pose thiếu tự nhiên cần được đánh giá trên checkpoint thật, không thể sửa chỉ bằng ánh sáng/render.

## 4. Render video và hình nhiều pose như paper

Render thử trước:

```bash
python tools/render_video.py \
  --motion /workspace/outputs/cmib-walk/motion.npz \
  --output /workspace/outputs/cmib-preview \
  --width 480 --height 360 --samples 16 --max-frames 12
```

Render đầy đủ, kích thước cuối 1920 × 1160 (bốn ô 960 × 540 + caption):

```bash
python tools/render_video.py \
  --motion /workspace/outputs/cmib-walk/motion.npz \
  --output /workspace/outputs/cmib-video \
  --width 960 --height 540 --samples 96 --threads 8
```

Output: `comparison.mp4`, bốn video riêng, PNG từng frame, `panel_0/scene.blend` … `panel_3/scene.blend`, và log render/encode. Mở `.blend` trong Blender để thay vật liệu, camera, ánh sáng hoặc kiểm tra keyframe. Start màu đỏ, target xanh dương; ô pose có anchor xanh lá. Mannequin procedural tạo được hình chuyển động dễ đọc nhưng không có mesh/quần áo của paper. Mặc định render Cycles CPU để chạy được headless; không yêu cầu OpenGL/GPU. `--engine BLENDER_EEVEE_NEXT` chỉ dùng khi máy có backend đồ họa tương thích. Dùng `--overwrite` khi chủ ý render lại vào thư mục đã có kết quả.

Hình tĩnh chồng nhiều thời điểm giống bố cục trong ảnh bạn gửi:

```bash
python tools/render_video.py \
  --motion /workspace/outputs/cmib-walk/motion.npz \
  --output /workspace/outputs/cmib-ghosts \
  --width 960 --height 540 --samples 96 --ghosts 7
```

Kết quả `comparison.png`. Không dùng `--max-frames` khi muốn hình tĩnh bao trùm cả chuyển động.

Muốn nhân vật người có mesh như ảnh: import BVH qua **File → Import → Motion Capture (.bvh)** (bật extension BVH nếu Blender chưa có), tỷ lệ cm→m `0.01`, forward `-Z`, up `Y`. Retarget rig nguồn sang rig nhân vật bằng mapping bone/rest pose và kiểm tra hướng trục, chiều dài chân, vị trí root. Rig Mixamo/Rigify không dùng chung rest pose với LAFAN nên không gán thẳng quaternion vào bone tương ứng. Kiểm tra bản BVH trước khi thêm foot IK hoặc contact cleanup; giữ video chưa cleanup để so sánh phương pháp. Hiện chưa có mesh/rig đích nên repo không tự động retarget một nhân vật bất kỳ. Mannequin mặc định dùng mesh segment có keyframe vị trí, không phải skinned character.

## 5. Huấn luyện lại thay vì dùng pretrained

Paper mục 4.5: 7 attention heads, 8 encoder layers, FFN 2048, dropout 0.05; AdamW lr=1e-4, betas=(0.9, 0.99), weight decay=0.01; batch 32; loss weights semantic/position/rotation = 1.5/0.05/2.0.

```bash
export WANDB_MODE=offline
python trainer.py \
  --data_path /workspace/assets/lafan/BVH \
  --processed_data_dir /workspace/outputs/cache-lafan-window90-train \
  --project /workspace/outputs/train --exp_name cmib80 \
  --window 90 --from_idx 9 --target_idx 88 \
  --batch_size 32 --epochs 5000 --save_interval 50 \
  --learning_rate 0.0001 --loss_cond_weight 1.5 \
  --loss_pos_weight 0.05 --loss_rot_weight 2.0 \
  --interpolation slerp --device 0 --seed 42
```

Dùng thư mục cache riêng cho mỗi window/dataset; loader cũ không tự kiểm tra cache có đúng cấu hình. Trainer có bước LR flip ghi BVH bổ sung, vì vậy dùng bản sao dữ liệu writable. Mã hiện nạp phần lớn tập dữ liệu lên device trước huấn luyện; cần cân nhắc RAM/VRAM. Seed hỗ trợ chạy lại nhưng không bảo đảm bitwise determinism trên mọi GPU/version. 5000 epoch theo hướng dẫn mã gốc, không phải cam kết hội tụ hay kết quả bằng paper. Môi trường cloud hiện tại không có GPU được xác nhận; chưa chạy huấn luyện đầy đủ hoặc toàn bộ benchmark.

Thay đổi đã làm: shim package `cmib`; sửa SLERP không mutate GT, chấp nhận quaternion q/-q tương đương; thêm seed, sửa AdamW beta2=0.99, detach loss logging, lưu checkpoint cuối; thêm inference/export, renderer Blender, regression tests. Không thay kiến trúc hoặc sửa metric để làm đẹp kết quả.

## 6. Kiểm thử

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m unittest discover -s tests -v
```

Các test dùng checkpoint **chưa huấn luyện**, chỉ xác minh pipeline, không chứng minh chất lượng CMIB. Kiểm tra ràng buộc start/end/anchor, bone lengths, quaternion, BVH round trip và label sai. Đã chạy một epoch synthetic với trainer, lưu checkpoint cuối và chạy Blender/FFmpeg xuất video 12 frame / 30 fps. Chưa xác nhận kết quả paper cho đến khi nạp được checkpoint thật, đúng label và dữ liệu test.
