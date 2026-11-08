# Chạy eval bundle trên vast.ai

Chạy `notebooks/run_eval_bundle.py` (kế hoạch: `docs/eval_plan_wikidata5m.md`) cho family, fb15k_237 và
wikidata5m trên một máy thuê. Máy vast là máy trắng và bị xóa khi hủy, nên mọi thứ được dựng lại từ GitHub cộng
một file dữ liệu.

## Cấu hình máy nên thuê

- RAM ≥ 64 GB. Đây là yêu cầu quan trọng nhất: chỉ riêng việc load wikidata5m (không dựng NetworkX) đã tốn khoảng
  19 GB, cộng thêm các đỉnh bộ nhớ của pandas và GNN.
- GPU ≥ 12 GB, driver hỗ trợ CUDA 12.4 (torch 2.6.0+cu124).
- Disk ≥ 50 GB. Env khoảng 10 GB, dữ liệu sau giải nén khoảng 1 GB, phần còn lại là cache PPR.
- Image Linux x86_64 bất kỳ có `bash`. `setup.sh` tự cài Miniforge, không dùng conda có sẵn trong image.
- Có internet, để tải gói pip và model `all-MiniLM-L6-v2` từ HuggingFace.

## Ở local, trước khi thuê máy

```bash
python -m vast.pack_data --out D:/gog_eval_data.tar.gz   # dùng C:/Users/MTBH/anaconda3/envs/hybkgqa/python.exe
```

Script ghi `vast/data_manifest.txt` (file này được commit) và tạo file tar chứa đúng các file trong manifest.
Nếu sửa manifest thì commit và push lại trước khi clone.

## Trên máy vast

```bash
# 1. upload dữ liệu (chạy ở local; PORT và HOST lấy từ nút SSH của instance)
scp -P <PORT> D:/gog_eval_data.tar.gz root@<HOST>:/workspace/

# 2-5. trên máy vast
cd /workspace
git clone -b feat/propose-collect-finish https://github.com/Neil-275/GoG_enhanced.git
cd GoG_enhanced
bash vast/setup.sh                               # cài Miniforge, env hybkgqa và model HF; khoảng 10-15 phút
tar -xzf /workspace/gog_eval_data.tar.gz         # giải nén vào root repo
tmux new -s eval                                 # để lần chạy không chết khi mất SSH
bash vast/run.sh                                 # preflight rồi chạy cả ba dataset
```

- `run.sh` kiểm tra trước khi chạy: đủ file và đúng kích thước theo manifest, RAM ≥ 64 GB, có GPU. Lần chạy
  không cần API key và không cần `.env`.
- Muốn chạy lại riêng một vài dataset: `bash vast/run.sh --datasets wikidata5m`.
- Note được ghi nối thêm sau mỗi bước, nên dù process chết giữa chừng, phần đã xong vẫn còn trong
  `results/eval_bundle/NOTE_<timestamp>.md`.

## Tải về trước khi hủy máy

```bash
scp -P <PORT> 'root@<HOST>:/workspace/GoG_enhanced/results/eval_bundle/*' ./results/eval_bundle/
scp -P <PORT> 'root@<HOST>:/workspace/GoG_enhanced/results/calibration_results/*_localpush.pkl' ./results/calibration_results/
```

- `results/eval_bundle/NOTE_*.md`: kết quả. Thư mục này còn chứa log `run_*.log`.
- `results/calibration_results/*_localpush.pkl`: chín file, mỗi dataset ba biến thể prior. Sau này chạy agent sẽ
  cần các file này; nếu mất thì phải thuê máy fit lại.
