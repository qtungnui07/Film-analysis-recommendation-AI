# So sánh baseline và balanced trên dữ liệu export mới

Hai model dưới đây đã train trước đó từ `all-MiniLM-L6-v2` với 3 epoch, learning rate 2e-5 và batch size 16, cùng dữ liệu `minilm_training_export/train.csv`. Baseline dùng 197.333 cặp chưa cân bằng; balanced dùng 177.195 cặp sau min 2/max 100. ZIP bản gốc dùng cùng loss `MultipleNegativesRankingLoss` nhưng đặt batch size 64 và có pipeline tạo dữ liệu khác: baseline này là bản thích ứng với CSV mới, không phải chạy nguyên notebook gốc.

Chấm cùng `minilm_training_export_new/mini_data.db`, cùng evaluator `scripts/evaluate_export_hybrid_weights.py`, 598 seen queries, 143 unseen queries và 3.425 title queries. Không dùng reranker IMDb.

| Model | MiniLM/BM25 | Seen P@10 | Unseen P@10 | Title hit@1 | Title hit@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | 100/0 | 0.3925 | 0.3301 | 0.6797 | 0.9101 |
| Balanced | 100/0 | 0.3786 | 0.3217 | 0.7653 | 0.9577 |
| Baseline | 80/20 | 0.3983 | 0.3385 | 0.7796 | 0.9550 |
| Balanced | 80/20 | 0.3793 | 0.3350 | 0.8286 | 0.9766 |
| Baseline train lại, batch 64 | 100/0 | 0.3793 | 0.3189 | 0.7445 | 0.9527 |
| Baseline train lại, batch 64 | 80/20 | 0.3836 | 0.3322 | 0.8201 | 0.9734 |

Ở 80/20 baseline cao hơn Seen P@10 0.0190 và Unseen P@10 0.0035; balanced cao hơn title hit@1 0.0490 và hit@10 0.0216. Đây là kết quả cho một lần train của mỗi model, chưa kiểm tra độ ổn định theo nhiều seed. Test đã được xem để so sánh nên không còn là bộ test hoàn toàn chưa đụng đến cho các lần chọn cấu hình sau.

Kết quả máy: `data/original-baseline-new-split-sweep.json` và `data/final-hybrid-weight-sweep-balanced-100-2.json`.

Lần train baseline batch 64 chạy đủ 3 epoch trong 973,6 giây trên CUDA, lưu tại `data/original-retrain-batch64`; kết quả đánh giá tại `data/original-batch64-new-split-sweep.json`. So sánh batch 64 với balanced batch 16 thay đổi cả batch size và sampling, vì vậy không thể quy toàn bộ chênh lệch cho balancing.
