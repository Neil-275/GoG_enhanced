# Kế hoạch đo đánh giá Wikidata5m (kèm Family và FB15K-237)

Chốt ngày 2026-11-08, sau buổi grilling. Code chưa được sửa theo kế hoạch này.

## 1. Mục tiêu

Đo các thành phần của Propose trên wikidata5m bằng cùng thước đo đã dùng cho Family và FB15K-237, chưa chạy agent.
Ba dataset được chạy chung trên một máy thuê ở vast.ai bằng một script. Mọi kết quả được ghi vào một file note
Markdown để tải về sau.

## 2. Những gì sẽ đo

Áp dụng cho cả ba dataset. Trừ khi ghi khác, mỗi phép đo chỉ tính các câu hỏi có đúng một cạnh bị drop nối
`q_entity` với `hard_answer`.

1. **Mô hình relation prior**: mô hình đoán slot `(relation, direction)` mà entity đang bị thiếu, chỉ dựa vào hồ sơ
   đếm relation của entity. Chỉ số là tỷ lệ slot đúng nằm trong top-3, đo cho bản pretrain và bản finetune. Phép đo
   này nhẹ (chỉ cần các cạnh quanh entity câu hỏi, không cần GNN). Chạy lại cho cả ba dataset để note có đủ số liệu.
   Số của wikidata5m (0.907 với finetune) bị lạc quan, vì `smoke_val` nằm trong val, và val là tập dùng để
   early-stopping khi finetune prior.

2. **GNN Predict khi biết sẵn slot đúng**: cho GNN biết đúng relation và direction của cạnh bị drop, rồi xem
   `hard_answer` đứng hạng mấy. Chỉ số: MRR, Hit@1/3/5/10. Đây là trần của bước Predict, tách riêng khỏi việc chọn slot.

3. **Độ phủ của subgraph** (đo cùng bước 2): sampler local push chỉ đưa cho GNN một subgraph quanh entity câu hỏi.
   Đo tỷ lệ `hard_answer` nằm trong subgraph, cùng trung vị số node và số cạnh. Số này phân biệt hai kiểu trượt:
   đáp án không có trong subgraph, và đáp án có trong subgraph nhưng GNN xếp hạng thấp.

4. **Propose xác suất rồi Predict, không dùng LLM**: chọn slot theo xác suất đã calibrate, gọi GNN trên các slot được
   chọn, rồi giữ các entity dự đoán vượt ngưỡng. Chỉ số:
   - tỷ lệ chọn đúng slot;
   - tỷ lệ `hard_answer` nằm trong các entity được dự đoán;
   - số slot và số entity dự đoán trung bình mỗi câu;
   - precision chuẩn hóa: `1 / số entity dự đoán` khi trúng, trung bình trên mọi câu.

5. **Ablation relation prior** cho bước 4: chạy ba biến thể là prior finetune, prior pretrain và không dùng prior
   (`lam = 0`, `k_p = 0`). Mỗi biến thể dùng calibration riêng.

## 3. Những gì không đo

- Baseline 1-hop không dùng agent (`run_baseline.py`).
- Đánh giá end-to-end Propose xác suất trên test (`eval_prob_propose.py`).
- Propose bằng LLM, Propose bằng LLM rồi Predict, và các biến thể type-check. Các phần này tốn chi phí gọi LLM, và
  chưa có kế hoạch so chế độ `llm` với chế độ `prob` trên wikidata5m.
- Agent đầy đủ (`GoG.GoG`, Hits@Hard). Để sau.
- Sampler matrix: không khả thi với 4.59M entity của wikidata5m. Với Family/FB15K, note chỉ dẫn lại số cũ trong cell
  tổng hợp của `notebooks/test_generate.ipynb` để đối chiếu, không chạy lại.

## 4. Thiết lập chung

- **Sampler**: local push cho cả ba dataset, giữ nguyên các tham số đang có trong `gnn_config.py`, không quét.
  - family: `epsilon 1e-7`, tối đa 1024 node, 10000 cạnh.
  - fb15k_237: `epsilon 1e-8`, tối đa 1454 node, 30000 cạnh.
  - wikidata5m: `epsilon 1e-6`, tối đa 1024 node, 10000 cạnh.

  Với wikidata5m, các giá trị này là mặc định của `one_shot_subgraph/train_auto.py`. Checkpoint không lưu tham số
  train, nên chưa xác nhận được đây đúng là cấu hình lúc train. Nếu MRR thấp bất thường thì ghi chú trong note và vẫn
  chạy tiếp, không dừng.
- **Tập câu hỏi**:
  - wikidata5m: fit calibration trên `val_calibration.tsv` (val trừ smoke, 2965 câu), báo cáo trên `smoke_val.tsv`
    (500 câu).
  - Family/FB15K: fit calibration và báo cáo trên cùng tập `val` (179 và 506 câu hợp lệ), giữ như trước. Calibration
    chỉ fit 2 tham số nhiệt độ nên mức rò rỉ nhỏ. Note sẽ ghi chú điểm này.

## 5. Calibration

Calibration đưa điểm của baseline Propose và điểm GNN về xác suất, bằng cách fit một nhiệt độ `T` cho mỗi loại điểm.
Cả hai bước Propose xác suất và ablation prior đều cần calibration.

- **Fit lại cho cả ba dataset với local push.** Calibration hiện tại của Family/FB15K được fit với sampler matrix, mà
  điểm GNN lại phụ thuộc vào subgraph. Wikidata5m thì chưa có calibration nào.
- **Các file cần fit cho mỗi dataset**: một nhiệt độ Predict (GNN không phụ thuộc prior, nên dùng chung), và một nhiệt
  độ Propose cho từng biến thể: finetune, pretrain, không prior.
- **Kiểm tra kết quả fit**: in đường NLL theo `T`, rồi tinh chỉnh bằng lưới mịn quanh cực tiểu. Lý do: hiện Family và
  FB15K ra đúng cùng một cặp `T` trên lưới 50 điểm, có thể chỉ do lưới quá thô.
- **Đặt tên file theo sampler**: `calibration_results_{dataset}_{mode}_{sampler}.pkl`, với sampler là `localpush` hoặc
  `matrix`. File hiện có được đổi tên thêm hậu tố `_matrix`, để số đang báo cáo vẫn tái lập được. Loader trong
  `GoG/prob_propose.py` chọn file theo giá trị `local_ppr` đang dùng.
- **Bật lại guard**: guard kiểm tra calibration cũ trong `ProbabilisticPropose.__init__` được bật lại, và bổ sung các
  tham số sampler (`local_ppr`, `epsilon`, cap node/cạnh, `alpha`) vào nhóm thông tin lưu trong file và được so khi
  load. Khi đó dùng calibration của một cấu hình khác sẽ báo lỗi, thay vì âm thầm cho ra số sai.

## 6. Thay đổi code

- `notebooks/eval_pipeline.py`: bổ sung số đo độ phủ subgraph vào bước GNN Predict.
- Chuyển code calibration wikidata5m (cell 34-38 của `relation_inference/baseline/calibrate.ipynb`) thành hàm Python
  dùng chung cho cả ba dataset. Notebook giữ nguyên.
- `GoG/prob_propose.py`: tên file calibration theo sampler, bật lại guard, có cách chọn biến thể prior bằng tham số
  thay vì sửa tay `PRIOR_PATHS`.
- Script mới `notebooks/run_eval_bundle.py` chạy toàn bộ (mục 7).

## 7. Script chạy và file note

- **Thứ tự dataset**: family, rồi fb15k_237, rồi wikidata5m. Dataset nhỏ chạy trước để lỗi code lộ ra sớm, trước khi
  tốn khoảng 19 GB RAM và 5 phút load wikidata5m.
- **Thứ tự bước trong mỗi dataset**: calibration, relation prior, GNN Predict kèm độ phủ, Propose xác suất, ablation
  prior. Mỗi dataset chỉ load KG và GNN một lần trong cùng process.
- **Xử lý lỗi**: bước nào lỗi thì ghi traceback vào note rồi chạy bước hoặc dataset tiếp theo.
- **Flag `--datasets`**: để chạy lại riêng một hoặc vài dataset.
- **File note**: `results/eval_bundle/NOTE_<timestamp>.md`, mỗi lần chạy một file mới. Script ghi nối thêm sau mỗi
  bước, nên process có chết giữa chừng thì phần đã xong vẫn còn. Nội dung gồm:
  - lệnh đã chạy, git commit và `git status`;
  - cấu hình sampler và đường dẫn prior của từng dataset;
  - kết quả calibration, kèm đường NLL theo `T`;
  - bảng số của từng bước;
  - số sampler matrix cũ của Family/FB15K để đối chiếu;
  - các ghi chú: cấu hình sampler wikidata5m chưa xác nhận, số prior wikidata5m lạc quan, Family/FB15K fit calibration
    và báo cáo trên cùng tập.

## 8. Git và chạy trên máy thuê vast.ai

Máy vast là máy trắng (không có repo, dữ liệu hay conda env) và bị xóa sạch khi hủy instance, nên mọi thứ phải tự
dựng lại được từ GitHub cộng một file dữ liệu.

**Ở local**
- **Commit 1**: các thay đổi cũ chưa commit (`GoG/kg_interface.py`, `GoG/prob_propose.py`, `gnn_config.py`, xóa
  `relation_inference/baseline/t.py`). Không commit `server_wd5m/`.
- **Commit 2**: code mới của kế hoạch này, kèm `setup.sh`, `run.sh` và script đóng gói dữ liệu.
- **Push** branch `feat/propose-collect-finish` lên `origin` sau commit 2. Lần push này đẩy luôn các commit trước đó
  chưa được push. `.env` đã nằm trong gitignore.
- **Đóng gói dữ liệu**: script tạo một file tar chứa đúng các file cần dùng, liệt kê trong một manifest, không tar
  nguyên thư mục. Ví dụ, `data_for_LP/id_processed/fb15k_237` nặng 4.9 GB, phần lớn là cache không cần. Dự kiến
  khoảng 2.5-3 GB. Các nhóm file cần có:
  - `brink_dataset/{family,fb15k_237,wikidata5m}/` (gồm `val_calibration.tsv` và `smoke_val.tsv` của wikidata5m);
  - `data_for_LP/weights/<dataset>/` (checkpoint GNN) và các file cần dùng trong `data_for_LP/id_processed/<dataset>/`;
  - `relation_inference/model/*_{pretrain,finetune}.pt` của cả ba dataset;
  - `sampled_args/sample_args_finetune_{family,fb15k_237}_2.pkl`;
  - `results/calibration_results/*.pkl` hiện có (để đổi tên thành `_matrix` và đối chiếu).

**Máy vast**
- Cấu hình: RAM ≥ 64 GB (quan trọng nhất; load wikidata5m không dựng NetworkX đã khoảng 19 GB), GPU ≥ 12 GB,
  driver hỗ trợ CUDA 12.4, disk ≥ 50 GB.
- Các bước:
  1. `git clone -b feat/propose-collect-finish https://github.com/Neil-275/GoG_enhanced.git`
  2. `setup.sh`: cài môi trường theo `GoG/hybkgqa.yaml` (torch 2.6.0 cu124, torch-scatter, torch-geometric). Phiên
     bản Python phải khớp env local (yaml ghi 3.10, nhưng `__pycache__` local là 3.11).
  3. Upload file tar dữ liệu và giải nén vào thư mục repo.
  4. `run.sh`: kiểm tra trước khi chạy (đủ file theo manifest, RAM, GPU, không cần API key), rồi gọi
     `notebooks/run_eval_bundle.py`.
- Máy cần có internet để tải model `all-MiniLM-L6-v2` từ HuggingFace.

**Tải về trước khi hủy máy**
- `results/eval_bundle/NOTE_*.md`
- `results/calibration_results/*_localpush.pkl` của cả ba dataset. Sau này chạy agent sẽ cần các file này; mất thì
  phải thuê máy fit lại.
