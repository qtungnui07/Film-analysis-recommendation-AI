# Nhật ký thí nghiệm Movie Search

Tài liệu này ghi lại toàn bộ pipeline train, các model đã tạo, kết quả đo và cấu hình được chốt cho **Phase 1 — Hard Negative Mining**.

## 1. Dữ liệu, metric và split

### Dữ liệu

| Thành phần | Vai trò |
| --- | --- |
| `moviebot.db` | Movie metadata, overviews, TMDB keywords và MovieLens Genome vectors |
| `tags.csv`, `links.csv` | MovieLens community tags và ánh xạ MovieLens ID sang IMDb ID |
| MovieLens Genome | Điểm relevance từ `0` đến `1` cho từng cặp `(movie, tag)` |

Genome relevance `>= 0.80` là positive, còn `<= 0.10` là candidate hard negative. Các phim không có Genome vector vẫn dùng được khi train từ keywords/overview, nhưng không dùng được để mine hoặc chấm hard negatives.

### Metric

| Metric | Ý nghĩa |
| --- | --- |
| Seen tags P@10 | Tỷ lệ phim liên quan trong top 10 với tag đã xuất hiện khi train |
| Unseen tags P@10 | Tỷ lệ phim liên quan trong top 10 với tag chưa xuất hiện khi train |
| Title hit@1 / hit@10 | Khả năng trả đúng phim khi query là tên phim |

Split outer cố định theo seed `42`: 20% movies và 20% tag keys là frozen test. Khi tune Hard Negatives, inner validation tiếp tục tách 10% từ outer-train; frozen test không được dùng để chọn hyperparameter.

## 2. Chuẩn bị môi trường local GPU

Máy train: NVIDIA GeForce RTX 2060 6 GB VRAM. Cache, temp và virtual environment đều đặt trên ổ E.

```cmd
set "UV_CACHE_DIR=E:\uv-cache"
set "TEMP=E:\temp"
set "TMP=E:\temp"

uv venv .venv-train
uv pip install --python .venv-train\Scripts\python.exe --torch-backend=auto torch "sentence-transformers==6.2.0" "transformers==5.22.2" "datasets==5.2.0" "accelerate==1.15.0" numpy
.venv-train\Scripts\python.exe scripts\train_search_local.py --check-device --device cuda
```

Khi GPU có 6 GB VRAM, dùng `--batch-size 16`.

## 3. Các model ban đầu

### First model — tag và keyword search

First model được fine-tune từ `sentence-transformers/all-MiniLM-L6-v2` với ba nguồn positive pairs:

1. MovieLens Genome tags có relevance `>= 0.80`;
2. MovieLens user tags, chỉ giữ tag có ít nhất 2 users;
3. TMDB keywords.

```cmd
.venv-train\Scripts\python.exe scripts\train_search_local.py --no-title-pairs --output data\minilm-movie-search --device cuda --batch-size 16
```

Artifact: `data/minilm-movie-search/`.

### Second model — bổ sung title pairs

Second model thêm pair `(movie title, movie text)` cho mỗi phim train. Nó cải thiện title lookup, đổi lại P@10 cho tag giảm nhẹ.

```cmd
.venv-train\Scripts\python.exe scripts\train_search_local.py --previous-model data\minilm-movie-search --output data\minilm-movie-search-titles --device cuda --batch-size 16
```

Artifact: `data/minilm-movie-search-titles/`.

### Kết quả trước Hard Negatives

| Setup | Seen P@10 | Unseen P@10 | Title hit@1 | Title hit@10 |
| --- | ---: | ---: | ---: | ---: |
| First model alone | 0.390 | 0.327 | 0.278 | 0.520 |
| First + BM25 80/20 | **0.395** | **0.330** | 0.407 | 0.619 |
| Second model alone | 0.387 | 0.317 | 0.575 | 0.870 |
| Second + BM25 80/20 | 0.393 | 0.322 | 0.698 | 0.926 |
| Untuned MiniLM | 0.197 | 0.175 | **0.887** | **0.990** |

Kết luận: First model + BM25 80/20 là baseline tốt nhất cho tag search. Second model phù hợp khi title lookup là ưu tiên.

## 4. Phase 1 — Hard Negative Mining

### Cách mine

1. Lấy First model làm baseline retriever.
2. Với mỗi tag thuộc train, retrieve top 100 movie documents chỉ trong training split.
3. Positive là relevance `>= 0.80`.
4. Hard negative là phim nằm trong top 100 nhưng relevance `<= 0.10`.
5. Tạo triplet `(tag, positive movie, hard-negative movie)`.
6. Fine-tune thêm 1 epoch, learning rate `5e-6`, 1 negative/positive.

Artifact `hard_negatives.csv` lưu tag, positive/negative movie ID, relevance score và retrieval rank để audit.

### Validation experiments

| Variant | Mining top-K | Negatives / positive | Validation Seen P@10 | Validation Unseen P@10 |
| --- | ---: | ---: | ---: | ---: |
| Baseline First | — | 0 | 0.380 | 0.303 |
| HN-1 | 100 | 1 | **0.393** | **0.315** |
| HN-2 | 100 | 2 | 0.394 | 0.312 |
| HN-3 | 50 | 1 | 0.389 | 0.300 |

HN-1 được chọn vì tăng tốt cả Seen và Unseen. HN-2 chỉ hơn `0.001` Seen nhưng giảm Unseen; HN-3 giảm cả hai metric.

### Train final và frozen-test result

Sau khi chốt HN-1 trên validation, model final được fine-tune lại trên full outer-train:

```cmd
.venv-train\Scripts\python.exe scripts\train_search_local.py --hard-negatives --hard-negative-base-model data\minilm-movie-search --previous-model data\minilm-movie-search --no-title-pairs --output data\minilm-movie-search-hard-negatives-final --device cuda --batch-size 16
```

Artifact final: `data/minilm-movie-search-hard-negatives-final/`.

| Setup | Seen P@10 | Unseen P@10 | Title hit@1 | Title hit@10 |
| --- | ---: | ---: | ---: | ---: |
| First + BM25 80/20 | 0.395 | 0.330 | 0.407 | 0.619 |
| First + Hard Negatives | 0.412 | **0.355** | 0.325 | 0.570 |
| **First + Hard Negatives + BM25 80/20** | **0.417** | 0.354 | 0.467 | 0.671 |

### Chốt Phase 1

- **Tag-search deployment khuyến nghị:** First model + Hard Negatives + BM25 80/20 (`0.417` Seen P@10, `0.354` Unseen P@10).
- **Nếu Unseen P@10 là metric ưu tiên duy nhất:** First model + Hard Negatives không BM25 (`0.412 / 0.355`).
- So với baseline tag-search tốt nhất, cấu hình deploy tăng `+0.022` Seen P@10 và `+0.024` Unseen P@10.

## 5. Phase 2 — Balanced Query Sampling

Mục tiêu tiếp theo là giảm mất cân bằng query. Trong inner-training hiện có 209.355 pairs nhưng 31.334 query phrases:

| Thống kê | Giá trị |
| --- | ---: |
| Median pairs/query | 1 |
| Query có `<= 2` pairs | 21.045 |
| Query có `> 50` pairs | 677 |
| Lớn nhất | 1.628 pairs/query |

Kế hoạch:

1. Cap query phổ biến ở 50 pairs/query.
2. Oversample query hiếm lên tối thiểu 3 pairs/query.
3. Chạy trên validation trước, so sánh balance-only và balance + HN-1.
4. Chỉ cấu hình thắng validation mới được chạy một lần trên frozen test.

Các triển khai và kết quả của Phase 2 sẽ được thêm vào tài liệu này.
