# Hướng dẫn train và tối ưu Movie Search

Tài liệu này ghi lại toàn bộ hai pipeline đã dùng trong project:

1. **Pipeline dữ liệu cũ**: database MovieLens/TMDB/Genome tự build; đã chạy đầy đủ baseline, title model và Hard Negatives.
2. **Pipeline dữ liệu export mới**: dùng `minilm_training_export/train.csv` làm dữ liệu train sạch, tách biệt với dữ liệu đánh giá.

Tài liệu liên quan:

- [SEARCH_EXPERIMENT_LOG_VI.md](SEARCH_EXPERIMENT_LOG_VI.md): kết quả chi tiết của pipeline cũ.
- [SEARCH_OPTIMIZATION_PLAN_VI.md](SEARCH_OPTIMIZATION_PLAN_VI.md): kế hoạch tối ưu ban đầu.

---

## A. Môi trường train local

Máy chạy: NVIDIA GeForce RTX 2060, 6 GB VRAM. Dùng Python virtual environment riêng và cache ở ổ E.

```cmd
set "UV_CACHE_DIR=E:\uv-cache"
set "TEMP=E:\temp"
set "TMP=E:\temp"

uv venv .venv-train
uv pip install --python .venv-train\Scripts\python.exe --torch-backend=auto torch "sentence-transformers==6.2.0" "transformers==5.22.2" "datasets==5.2.0" "accelerate==1.15.0" numpy
.venv-train\Scripts\python.exe scripts\train_search_local.py --check-device --device cuda
```

Khi CUDA nhận GPU, train bằng `--batch-size 16`. Script tự bật FP16. Batch 32 có thể nhanh hơn, nhưng batch 16 được giữ để so sánh thí nghiệm công bằng với các kết quả đã có.

---

## B. Pipeline dữ liệu cũ — lịch sử đầy đủ

### B.1. Dữ liệu và khả năng chạy

Pipeline cũ dùng các file local sau:

| Dữ liệu | Mục đích |
| --- | --- |
| `moviebot-drive/moviebot/moviebot.db` | SQLite catalogue, MovieLens Genome và TMDB overview/keywords |
| `moviebot-drive/moviebot/tags.csv`, `links.csv` | MovieLens community tags và ánh xạ ID |
| `data/moviebot.db` | Database app/backend sau khi copy/build |

Có thể train và đánh giá trực tiếp vì database có Genome relevance cho một phần catalogue. Các lệnh của pipeline cũ dùng `scripts/train_search_local.py` **không có** `--train-csv`.

> **Trạng thái:** `moviebot-drive/`, model outputs và validation outputs cũ đã được xóa để nhường dung lượng. Vì vậy các lệnh phần B chỉ là lịch sử tái lập khi khôi phục lại dữ liệu cũ từ backup/download; hiện tại không chạy được trong workspace này.

### B.2. First model: tag + keyword search

Nguồn pair train:

1. Genome tag có relevance `>= 0.80`;
2. MovieLens user tags có ít nhất 2 users;
3. TMDB keywords.

```cmd
.venv-train\Scripts\python.exe scripts\train_search_local.py --no-title-pairs --output data\minilm-movie-search --device cuda --batch-size 16
```

### B.3. Second model: thêm title pairs

```cmd
.venv-train\Scripts\python.exe scripts\train_search_local.py --previous-model data\minilm-movie-search --output data\minilm-movie-search-titles --device cuda --batch-size 16
```

First model tốt hơn cho tag search; Second model tốt hơn rõ rệt cho title lookup.

### B.4. Hard Negative Mining

Hard negative là phim được model retrieve gần tag nhưng có Genome relevance `<= 0.10`. Positive có relevance `>= 0.80`.

Quy trình:

1. Tách inner validation từ outer-train: 10% movies có Genome và 10% tag keys.
2. Dùng First model retrieve top-K chỉ trong inner-training movies.
3. Tạo triple `(tag, positive movie, hard-negative movie)`.
4. Fine-tune thêm 1 epoch, LR `5e-6`.
5. Chọn hyperparameter trên validation, sau đó train full outer-train và chấm frozen test.

Validation variants đã chạy:

| Variant | Top-K | Negatives / positive | Validation Seen P@10 | Validation Unseen P@10 |
| --- | ---: | ---: | ---: | ---: |
| HN-1 | 100 | 1 | **0.393** | **0.315** |
| HN-2 | 100 | 2 | 0.394 | 0.312 |
| HN-3 | 50 | 1 | 0.389 | 0.300 |

Chọn HN-1. Frozen-test kết quả:

| Setup | Seen P@10 | Unseen P@10 |
| --- | ---: | ---: |
| First + BM25 80/20 | 0.395 | 0.330 |
| First + Hard Negatives | 0.412 | **0.355** |
| **First + Hard Negatives + BM25 80/20** | **0.417** | 0.354 |

Kết luận pipeline cũ: **First model + HN-1 + BM25 80/20** là cấu hình deploy cân bằng nhất cho tag search.

---

## C. Pipeline dữ liệu export mới — đang dùng

### C.1. File nào được phép dùng

| File | Có dùng để train? | Ghi chú |
| --- | --- | --- |
| `minilm_training_export/train.csv` | **Có** | 197.333 cặp `anchor,positive`, không có nhãn test |
| `minilm_training_export/README.md` | Có | mô tả data protocol |
| `minilm_training_export/mini_data.db` | **Không** | chứa split, Genome/test metadata; không dùng để tune/training |
| `Movie_Search_Test_GroundTruth_Evaluation_GiaiThich.ipynb` | Chỉ sau khi chốt model | evaluator giữ test ground truth riêng |

`train.csv` đã bao gồm Genome tags, TMDB keywords và IMDb titles trong cột `anchor`; vì vậy model train từ file này là **title-aware** ngay từ đầu. Không còn cần tách First/Second model như pipeline cũ.

### C.2. Baseline mới

```cmd
.venv-train\Scripts\python.exe scripts\train_search_local.py --train-csv minilm_training_export\train.csv --output data\minilm-export-baseline --device cuda --batch-size 16
```

Kết quả train:

| Model | Training pairs | Runtime | Train loss |
| --- | ---: | ---: | ---: |
| Baseline export | 197.333 | 27 phút 50 giây | 1.447 |

Model được lưu ở `data/minilm-export-baseline/`. Script không tự đọc test data và không in P@10, đúng theo protocol export.

### C.3. Balanced Query Sampling — đang chạy

Từ `train.csv`:

| Thống kê raw data | Giá trị |
| --- | ---: |
| Training pairs | 197.333 |
| Unique query phrases | 33.996 |
| Query có 1 pair (được oversample lên 2) | 22.142 |
| Query bị cap ở cấu hình 100/2 | 334 |

Cấu hình thử nghiệm hiện tại:

- query có dưới 2 pairs được oversample lên 2 rows;
- query có trên 100 pairs chỉ giữ ngẫu nhiên, cố định theo seed, 100 rows;
- các query khác giữ nguyên;
- không query nào bị xóa hoàn toàn.

```cmd
.venv-train\Scripts\python.exe scripts\train_search_local.py --train-csv minilm_training_export\train.csv --balanced-query-sampling --min-pairs-per-query 2 --max-pairs-per-query 100 --output data\minilm-export-balanced-100-2 --device cuda --batch-size 16
```

Kết quả train:

| Model | Training rows | Runtime | Train loss |
| --- | ---: | ---: | ---: |
| Balanced 100/2 | 177.195 | 25 phút 47 giây | 1.266 |

Model được lưu ở `data/minilm-export-balanced-100-2/`. Loss của hai model không đủ để kết luận P@10 vì distribution train khác nhau. Cả hai candidate chưa đọc test data.

### C.4. Proxy validation chỉ dùng `train.csv`

Để chọn giữa baseline và balanced mà không dùng `mini_data.db`, pipeline tạo một **unseen-anchor proxy validation**:

1. Tách 10% normalized anchor keys bằng seed cố định, giữ toàn bộ pairs của các key đó khỏi training.
2. Train model trên 90% anchor keys còn lại.
3. Retrieve trên các anchor bị giữ lại, với movie texts và positive pairs chỉ lấy từ `train.csv`.
4. Báo cáo Proxy Precision@10, Recall@10 và MRR@10.

Đây không thay thế evaluation chính thức với Genome ground truth, nhưng là tín hiệu train-only hợp lệ để chọn baseline hay balanced. Chỉ candidate thắng proxy validation mới được đưa qua evaluator notebook chính thức.

### C.5. Kết quả Proxy Baseline và trạng thái Proxy Balanced

Proxy Baseline đã train với 10% normalized anchor keys bị giữ lại hoàn toàn:

| Hạng mục | Giá trị |
| --- | ---: |
| Held-out pairs | 18.868 |
| Held-out normalized anchor keys | 3.236 |
| Proxy queries (anchor text gốc) | 3.390 |
| Candidate movie documents | 13.698 |
| Proxy Precision@10 | 0.0755 |
| Proxy Recall@10 | 0.4392 |
| Proxy MRR@10 | 0.4083 |

Đây là baseline proxy để so với Balanced; các số này **không** so trực tiếp với P@10 của Genome evaluation cũ.

Proxy Balanced 100/2 chạy trên đúng proxy split và seed. Số training steps ít hơn baseline là chủ đích:

| Hạng mục | Proxy Baseline | Proxy Balanced 100/2 |
| --- | ---: | ---: |
| Input train pairs trước balancing | 178.465 | 178.465 |
| Output training rows | 178.465 | 159.769 |
| Unique training queries | 30.606 | 30.606 |
| Optimizer steps cho 3 epochs, batch 16 | 33.465 | 29.958 |
| Query groups bị cap | — | 308 |
| Query groups được oversample | — | 19.976 |

Balanced không xóa query nào. Nó cap các query có quá nhiều pairs và lặp thêm query hiếm. Số rows bị cap lớn hơn số duplicate rows được thêm vào nên tổng rows, và vì thế số optimizer steps, giảm khoảng 10%.

Kết quả proxy:

| Model | Proxy Precision@10 | Proxy Recall@10 | Proxy MRR@10 |
| --- | ---: | ---: | ---: |
| Proxy Baseline | 0.0755 | 0.4392 | 0.4083 |
| **Proxy Balanced 100/2** | **0.0782** | **0.4612** | **0.4416** |
| Balanced 100/2 + Query Augmentation (1 variant) | 0.0780 | **0.4637** | 0.4383 |

Balanced 100/2 tăng `+0.0027` Precision@10, `+0.0220` Recall@10 và `+0.0333` MRR@10 so với proxy baseline. Query Augmentation trial tăng Recall thêm `+0.0025`, nhưng giảm Precision `-0.0002` và MRR `-0.0033` so với Balanced. Vì MRR và Precision cùng giảm, trial augmentation bị loại. Candidate tốt nhất sau Phase 3 vẫn là `data/minilm-export-proxy-balanced-100-2/`.

---

## D. Bốn phase tối ưu

### Phase 1 — Hard Negative Mining

**Mục tiêu:** Dạy model loại các phim gần ngữ nghĩa nhưng không liên quan thật sự.

**Đã làm với dữ liệu cũ:** HN-1 thắng validation và frozen test, như phần B.4.

**Trạng thái với export mới:** Đã nhận `genome_relevance_train_all_scores.csv` chỉ gồm rows `split=train`, sau đó nén/kiểm tra bằng `scripts/prepare_genome_relevance.py` thành `data/genome-train-relevance.npz`.

| Kiểm tra compact artifact | Kết quả |
| --- | ---: |
| Train movies có Genome | 7.192 |
| Genome tags sau khi loại held-out tags | 905 |
| Relevance range | 0.0005–1.0000 |
| Candidate relevance `<= 0.1` | 4.254.840 |
| Positive relevance `>= 0.8` | 67.696 |
| Matrix tag key giao held-out tag key | 0 |

Artifact không chứa held-out Genome tag và bao phủ toàn bộ Genome anchors trong `train_with_source.csv`. HN-1 trên proxy split dùng base model Balanced đã train trên cùng 90% anchors:

```cmd
.venv-train\Scripts\python.exe scripts\train_search_local.py --train-with-source-csv minilm_training_export_new\train_with_source.csv --hard-negatives --hard-negative-relevance data\genome-train-relevance.npz --hard-negative-base-model data\minilm-export-proxy-balanced-100-2 --proxy-validation-share 0.1 --output data\minilm-export-proxy-balanced-100-2-hn1 --device cuda --batch-size 16
```

**HN-1 kết quả:** Mine được 34.656 triples từ 661 tags, nhưng proxy Precision/Recall/MRR lần lượt là `0.0739 / 0.4188 / 0.3858`, thấp hơn Base Balanced (`0.0782 / 0.4612 / 0.4416`). HN-1 không được chọn. Audit cho thấy nhiều negative có retrieval rank 70–100, không còn đủ hard và fine-tuning toàn bộ 1 epoch làm model nghiêng mạnh về Genome pairs.

HN-2 chỉ giữ negative rank `<= 20` và giảm learning rate để hạn chế catastrophic forgetting:

```cmd
.venv-train\Scripts\python.exe scripts\train_search_local.py --train-with-source-csv minilm_training_export_new\train_with_source.csv --hard-negatives --hard-negative-relevance data\genome-train-relevance.npz --hard-negative-base-model data\minilm-export-proxy-balanced-100-2 --hard-negative-max-rank 20 --hard-negative-learning-rate 2e-6 --proxy-validation-share 0.1 --output data\minilm-export-proxy-balanced-100-2-hn2 --device cuda --batch-size 16
```

| Candidate | Proxy Precision@10 | Proxy Recall@10 | Proxy MRR@10 |
| --- | ---: | ---: | ---: |
| **Balanced baseline** | **0.0782** | **0.4612** | **0.4416** |
| HN-1: top-100, LR `5e-6` | 0.0739 | 0.4188 | 0.3858 |
| HN-2: top-20, LR `2e-6` | 0.0755 | 0.4391 | 0.4166 |

HN-2 giảm mức suy giảm so với HN-1, nhưng vẫn không vượt baseline ở bất kỳ metric nào. Do đó không chạy thêm HN-only tuning; candidate deploy vẫn là Balanced baseline. Một HN phase tiếp theo, nếu làm, phải trộn hard-negative triples với original TMDB/Title pairs trong cùng objective để chống forgetting, không chỉ giảm rank/LR.

HN-3 đã được triển khai theo hướng mixed objective: cùng một epoch, trainer dùng `MultipleNegativesRankingLoss` riêng cho original balanced pairs và hard-negative triples, lấy batch theo tỷ lệ size dataset. Vì vậy Title/TMDB vẫn xuất hiện trong training thay vì để Genome triples ghi đè representation. Trial giữ top-20, LR `2e-6` và dùng đúng Balanced 100/2 source mix:

```cmd
.venv-train\Scripts\python.exe scripts\train_search_local.py --train-with-source-csv minilm_training_export_new\train_with_source.csv --balanced-query-sampling --min-pairs-per-query 2 --max-pairs-per-query 100 --hard-negatives --hard-negative-relevance data\genome-train-relevance.npz --hard-negative-base-model data\minilm-export-proxy-balanced-100-2 --hard-negative-max-rank 20 --hard-negative-learning-rate 2e-6 --hard-negative-mix-original --proxy-validation-share 0.1 --output data\minilm-export-proxy-balanced-100-2-hn3-mixed --device cuda --batch-size 16
```

| Candidate | Proxy Precision@10 | Proxy Recall@10 | Proxy MRR@10 |
| --- | ---: | ---: | ---: |
| **Balanced baseline** | 0.0782 | **0.4612** | **0.4416** |
| HN-1: top-100, HN only | 0.0739 | 0.4188 | 0.3858 |
| HN-2: top-20, HN only | 0.0755 | 0.4391 | 0.4166 |
| HN-3: top-20, mixed original + HN | **0.0788** | 0.4605 | 0.4398 |

HN-3 xác nhận mixed objective đã tránh phần lớn forgetting: metric gần baseline, Precision tăng `+0.0006`. Tuy nhiên Recall giảm `-0.0007` và MRR giảm `-0.0018`; chênh lệch quá nhỏ để coi là thắng rõ ràng. Theo tiêu chí chọn đặt trước (không giảm metric, ưu tiên MRR khi trade-off), giữ **Balanced baseline** làm candidate deploy. Không tune HN thêm dựa trên chính proxy split này.

### Phase 2 — Balanced Query Sampling

**Mục tiêu:** Không để query quá phổ biến thống trị gradient; tăng trọng số query hiếm để phù hợp với macro P@10.

**Cấu hình đã chốt trong proxy validation:** `min=2`, `max=100`. Đây là bản conservative, giữ nhiều unique pairs hơn cấu hình `min=3`, `max=50`.

**Quyết định sau train:** Proxy Balanced 100/2 thắng unseen-anchor proxy validation và giữ vị trí candidate tốt nhất sau khi Query Augmentation không thắng. Không dùng điểm test để lặp lại tuning nhiều lần.

### Phase 3 — Query Augmentation

**Mục tiêu:** Tăng khả năng hiểu query phrasing mới, đặc biệt phần unseen tags.

**Trạng thái:** Đã chạy trial một biến thể/anchor, công bằng trên đúng proxy split (`seed=42`, `--proxy-validation-share 0.1`).

**Dữ liệu có thể augment:** chỉ `anchor` thuộc `train.csv`, và chỉ sau khi đã tách proxy validation. Một biến thể có normalized key thuộc proxy holdout bị loại, nên augmentation không thể đưa quan hệ của query validation vào train.

Biến thể được kiểm soát, không gọi LLM và không lấy dữ liệu bên ngoài:

```text
sci-fi → sci fi; science fiction
Spider-Man → Spider Man
time travel → time-travel; travelling through time
zombies → zombie movie; zombie films
```

Mỗi generated anchor được ghi vào `query_augmentation.csv` cùng `original_anchor` và `strategy` (`punctuation_normalization` hoặc `curated_alias`). Cấu hình trial đầu chỉ dùng **tối đa 1 biến thể/anchor** để tránh tăng noise. Lệnh chạy:

```cmd
.venv-train\Scripts\python.exe scripts\train_search_local.py --train-csv minilm_training_export\train.csv --proxy-validation-share 0.1 --query-augmentation --max-augmented-variants-per-query 1 --balanced-query-sampling --min-pairs-per-query 2 --max-pairs-per-query 100 --output data\minilm-export-proxy-balanced-100-2-aug1 --device cuda --batch-size 16
```

Kết quả trial `data/minilm-export-proxy-balanced-100-2-aug1/`: 164.529 training rows sau balance, runtime 22 phút 35 giây, train loss `1.258`; Proxy Precision@10 `0.0780`, Recall@10 `0.4637`, MRR@10 `0.4383`.

Tiêu chí chọn yêu cầu Precision@10, Recall@10 và MRR@10 không giảm, ưu tiên MRR@10. Trial chỉ tăng Recall nhưng giảm Precision và MRR, nên **không được chọn**. Giữ `data/minilm-export-proxy-balanced-100-2/`.

Quy tắc an toàn:

- lưu audit CSV `generated_anchor -> original_anchor`;
- không dùng test query/ground truth;
- giới hạn 1–2 biến thể chất lượng cho mỗi anchor;
- so sánh bằng proxy trước; evaluator chính thức chỉ chạy sau khi chốt candidate.

### Phase 4 — Ablation và chốt deployment

**Mục tiêu:** Biết chính xác improvement đến từ đâu và chọn model dùng thật.

**Trạng thái:** `minilm_training_export_new/train_with_source.csv` đã được nhận và kiểm tra: 197.333 pair khớp hoàn toàn với `train.csv`, không có pair trống/trùng. Nguồn có sẵn là `genome`, `tmdb_keywords` và `title`; chưa có `user_tag`.

Source ablation chạy hoàn toàn từ CSV mới, không cần mở `mini_data.db`. Pair có nguồn gộp `genome;tmdb_keywords` được giữ trong trial `without_genome`, vì nhãn TMDB vẫn độc lập hỗ trợ pair đó. Tất cả trial dùng cùng proxy holdout được sinh từ full 197.333 pair (`seed=42`, 10% normalized anchor keys); chỉ data train thay đổi.

| Trial | Pairs sau source filter | Pairs train sau proxy holdout | Rows sau Balanced 100/2 |
| --- | ---: | ---: | ---: |
| Full (baseline đã có) | 197.333 | 178.465 | 159.769 |
| Genome only | 67.696 | 61.235 | 39.150 |
| TMDB only | 123.611 | 111.815 | 107.859 |
| Title only | 13.665 | 12.353 | 23.809 |
| Without Genome | 137.276 | 124.168 | 131.668 |
| Without TMDB | 81.361 | 73.588 | 62.959 |
| Without Title | 183.668 | 166.112 | 135.960 |

Lệnh mẫu cho Genome only:

```cmd
.venv-train\Scripts\python.exe scripts\train_search_local.py --train-with-source-csv minilm_training_export_new\train_with_source.csv --include-sources genome --proxy-validation-share 0.1 --balanced-query-sampling --min-pairs-per-query 2 --max-pairs-per-query 100 --output data\ablation-proxy-genome --device cuda --batch-size 16
```

Thay `--include-sources genome` bằng `tmdb_keywords` hoặc `title` cho source-only. Dùng `--exclude-sources genome` (hoặc `tmdb_keywords`, `title`) cho leave-one-source-out. Chạy ba source-only trước; nếu cần giải thích chi tiết hơn mới chạy ba leave-one-out.

Kết quả proxy đã có (cùng held-out keys và document pool):

| Trial | Proxy Precision@10 | Proxy Recall@10 | Proxy MRR@10 |
| --- | ---: | ---: | ---: |
| Full Balanced 100/2 | **0.0782** | **0.4612** | **0.4416** |
| Genome only | 0.0537 | 0.2766 | 0.2441 |
| TMDB only | 0.0609 | 0.3284 | 0.2754 |
| Title only | 0.0534 | 0.4085 | 0.4337 |
| Without Title | 0.0620 | 0.3158 | 0.2745 |
| Without TMDB | 0.0709 | 0.4375 | 0.4010 |
| Without Genome | 0.0768 | **0.4652** | **0.4428** |

**Kết luận ablation:**

- Title là nguồn đóng góp lớn nhất cho Recall/MRR: bỏ Title khiến Recall `0.4612 → 0.3158` và MRR `0.4416 → 0.2745`.
- TMDB keyword đứng thứ hai: bỏ TMDB làm P@10 `0.0782 → 0.0709`, Recall `0.4612 → 0.4375`, MRR `0.4416 → 0.4010`.
- Genome có ảnh hưởng biên rất nhỏ ở cấu hình hiện tại: bỏ Genome giảm Precision `0.0782 → 0.0768`, nhưng Recall/MRR tăng nhẹ (`0.4652`/`0.4428`). Sai khác nhỏ này chưa đủ để khẳng định Genome gây hại; chọn **Full** làm default vì P@10 là metric chính của keyword search và score P@10 cao nhất.

Vì source ablation đã hoàn tất, không chạy thêm trial sampling/augmentation theo proxy. Artifact Genome all-score train-only đã sẵn sàng; hướng tối ưu tiếp theo là HN-1 theo lệnh ở Phase 1.

Các ablation đã khả dụng:

1. baseline raw `train.csv`;
2. balanced query sampling;
3. baseline + query augmentation;
4. balanced + query augmentation (đã chạy, không thắng).

Để ablate `user_tag`, bên Data cần export thêm source đó vào CSV. Sau HN-1, chỉ dùng proxy metrics để chốt/loại candidate; candidate cuối mới được chấm bằng `Movie_Search_Test_GroundTruth_Evaluation_GiaiThich.ipynb` đúng một lần. Nếu application cần BM25 hybrid, chạy evaluation với cấu hình RRF đã khóa và báo cáo như một hệ hai tầng riêng.

---

## E. Quy tắc chốt kết quả

1. Không dùng `mini_data.db` hoặc evaluator ground truth để lựa chọn hyperparameter.
2. Không sửa model sau khi xem final test metrics rồi báo đó là test độc lập.
3. Lưu command, seed, batch size, số epochs, số training pairs và thư mục model cho mỗi candidate.
4. Báo cáo cả Seen và Unseen P@10; không chỉ chọn metric đẹp hơn.
5. Với pipeline export, model cuối chỉ được chấm sau khi đã chốt phase tối ưu.

---

## F. Final evaluation trên split khớp `train.csv`

Sau khi đã chốt candidate qua proxy validation và ablation, model `data/minilm-export-balanced-100-2/` được chấm **một lần** trên `minilm_training_export_new/mini_data.db`: split test khớp đúng với `train_with_source.csv`/`train.csv`. Có 1.798 content candidate movies, 741 content queries (598 seen, 143 unseen) và 3.425 title queries. JSON tổng hợp không chứa answer/query test nằm ở `data/final-evaluation-new-split-balanced-100-2.json`.

| Semantic/BM25 | Seen P@10 | Seen R@100 | Unseen P@10 | Unseen R@100 | Title hit@1 | Title hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100/0 | 0.3786 | 0.6559 | 0.3217 | 0.5581 | 0.7653 | 0.9577 |
| **80/20** | **0.3793** | **0.6585** | **0.3350** | **0.5581** | 0.8286 | 0.9766 |
| 70/30 | 0.3727 | 0.6523 | 0.3315 | 0.5558 | 0.8371 | 0.9834 |
| 60/40 | 0.3599 | 0.6417 | 0.3168 | 0.5451 | 0.8350 | **0.9857** |
| 50/50 | 0.3258 | 0.6305 | 0.2965 | 0.5367 | **0.8385** | 0.9839 |

**Kết luận deploy:** dùng `data/minilm-export-balanced-100-2/` với hybrid RRF MiniLM/BM25 `80/20`: tốt nhất cho cả Seen và Unseen P@10. Tỉ lệ 50/50--60/40 chỉ hợp khi ưu tiên tìm đúng tên phim, vì title-hit cao hơn nhưng content search giảm rõ rệt. Bảng sweep nằm tại `data/final-hybrid-weight-sweep-balanced-100-2.json` và `.csv`; đây là final metric đã khóa, không dùng để thay đổi source weight, Hard Negative hay hyperparameter.

---

## G. Retrospective evaluation với database cũ

Model deploy là `data/minilm-export-balanced-100-2/`, đã train trên **100% `train.csv`** sau khi cấu hình Balanced 100/2 được chọn bằng proxy validation. Model được chạy trên `moviebot-db-old/`: `moviebot.db`, `links.csv`, `tags.csv`, seed `42`, test share `20%`. Benchmark cũ tạo ra 1.798 test movies, 589 seen-tag queries và 152 unseen-tag queries. JSON đầy đủ nằm ở `data/old-db-evaluation-final-balanced-100-2.json`.

`data/minilm-export-proxy-balanced-100-2/` chỉ train trên 90% anchors cho mục đích model selection; không phải model deploy và không dùng bảng điểm đó làm kết quả cuối.

| Setup | Seen P@10 | Unseen P@10 | Title hit@1 | Title hit@10 |
| --- | ---: | ---: | ---: | ---: |
| Final Balanced 100/2 (semantic only) | 0.370 | **0.362** | 0.711 | 0.944 |
| **Final Balanced 100/2 + BM25 80/20** | **0.373** | **0.362** | 0.785 | 0.970 |
| Final Balanced 100/2 + BM25 70/30 | 0.368 | 0.353 | 0.799 | 0.980 |
| Final Balanced 100/2 + BM25 50/50 | 0.325 | 0.301 | **0.806** | **0.981** |
| Untuned MiniLM | 0.197 | 0.175 | 0.887 | 0.990 |

**Cách đọc và cấu hình deploy:** Hybrid **BM25 80/20** là lựa chọn cân bằng: Seen P@10 cao nhất (`0.373`), giữ Unseen P@10 bằng semantic-only (`0.362`), đồng thời tăng mạnh title hit@10 (`0.970` so với `0.944`). Nếu ứng dụng chỉ có tìm title chính xác thì 50/50 cao hơn một chút, nhưng P@10 theo nội dung giảm nhiều nên không chọn làm default.

Đây là **retrospective/compatibility evaluation**, không phải điểm test độc lập cuối cùng: `train.csv` được xuất theo split khác và có thể chứa một phần phim/tag mà benchmark cũ xem là test. Vì vậy chỉ dùng bảng này để kiểm tra model chạy tương thích trên DB cũ và báo cáo minh bạch; không dùng để tune hay claim là kết quả độc lập của pipeline export.
