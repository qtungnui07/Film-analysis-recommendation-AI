# Kế hoạch tối ưu Movie Search

> Bản tiếng Việt của [SEARCH_OPTIMIZATION_PLAN.md](SEARCH_OPTIMIZATION_PLAN.md).

## Mục tiêu và baseline hiện tại

Cải thiện khả năng tìm phim theo tag/keyword, đồng thời giữ cách đánh giá trên tập test độc lập. Cấu hình tốt nhất hiện tại là **First model + BM25 80/20**:

| Chỉ số | Kết quả hiện tại |
| --- | ---: |
| Seen tags P@10 | 0.395 |
| Unseen tags P@10 | 0.330 |

`Seen`/`unseen` tag và 20% số phim được giữ lại bằng seed-42 để làm test. Tập test này phải được đóng băng: không dùng phim/tag test để train, tạo câu augment hoặc mine hard negative.

## Giai đoạn 0 — Tạo validation split

Trước khi tune, cần giữ thêm một validation set *bên trong dữ liệu train hiện tại*.

- Giữ nguyên test movies và test tag keys hiện có để báo cáo kết quả cuối.
- Từ dữ liệu train còn lại, tách 10% movies và 10% tag keys bằng một seed cố định thứ hai.
- Chỉ mine negatives và train trên phần inner-training còn lại.
- Chọn threshold, số lượng negatives, epochs và learning rate theo validation P@10.
- Chỉ chạy frozen test một lần cho cấu hình đã được chọn.

Việc này ngăn tình trạng thử quá nhiều cấu hình rồi vô tình chọn cấu hình may mắn trên test set.

## Giai đoạn 1 — Hard-negative mining (ưu tiên triển khai đầu tiên)

### Mục tiêu

Dạy embedding model rằng một phim có thể trông gần về ngữ nghĩa với một tag, nhưng thực tế vẫn không liên quan. Random negative thường quá dễ nên ít giúp cải thiện thứ hạng.

### Dữ liệu và quy tắc

Với mỗi tag query chỉ thuộc tập train:

| Thành phần | Quy tắc |
| --- | --- |
| Positive movie | Genome relevance `>= 0.80` |
| Candidate negative | Genome relevance `<= 0.10` |
| Hard negative | Candidate được First model trả về trong top 100 cho tag đó |
| Phạm vi | Chỉ dùng inner-training movies và các tag key không bị hold-out |
| Số lượng | Bắt đầu với 1 negative/positive; chỉ thử 2 hoặc 3 sau validation |

Điều kiện top-100 làm negative trở nên “khó”: đó là phim baseline model tưởng là gần với tag, nhưng MovieLens relevance đánh giá là không liên quan.

### Pipeline triển khai

1. Train hoặc load **First model** hiện có.
2. Chỉ embed movie documents và tag queries của inner-training set.
3. Dùng First model retrieve top 100 movies cho mỗi tag.
4. Loại positives và mọi phim có relevance lớn hơn `0.10`.
5. Chọn candidate còn lại xếp hạng cao nhất làm hard negative; nếu một query có nhiều positives thì ưu tiên negatives khác nhau.
6. Tạo triplet: `(tag query, positive movie text, hard-negative movie text)`.
7. Fine-tune First model thêm một epoch với learning rate thấp hơn (`5e-6`) và contrastive loss nhận explicit negatives.
8. Đánh giá trên validation; chỉ giữ checkpoint nếu cả hai metric tag đều tăng, hoặc trade-off được giải thích rõ.
9. Đánh giá checkpoint được chọn đúng một lần trên frozen test và ghi vào report.

### Ma trận thử nghiệm ban đầu

Chạy theo thứ tự dưới đây và chỉ chọn cấu hình bằng validation results:

| Run | Top-K mining | Negative cutoff | Negatives / positive | Epochs | Learning rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| HN-0 | — | — | 0 | — | — |
| HN-1 | 100 | `<= 0.10` | 1 | 1 | `5e-6` |
| HN-2 | 100 | `<= 0.10` | 2 | 1 | `5e-6` |
| HN-3 | 50 | `<= 0.10` | 1 | 1 | `5e-6` |

`HN-0` là First-model baseline hiện tại. Nếu validation giảm liên tục thì dừng; nguyên nhân thường là noisy negatives, không phải thiếu epochs.

### Deliverables khi code

- Mở rộng `scripts/train_search_local.py` với phase `--hard-negatives`.
- Thêm tham số cho mining model/checkpoint, top-K, relevance cutoff, negatives per positive và fine-tuning epochs.
- Lưu mined triples cùng source relevance score ra CSV/JSON để kiểm tra.
- Lưu model sau hard-negative mining riêng, ví dụ `data/minilm-movie-search-hard-negatives/`.
- Thêm deterministic tests cho split isolation và negative selection.

### Tiêu chí thành công

- Không có validation hoặc frozen-test item nào xuất hiện trong mined triples.
- Mọi hard negative được chọn đều có Genome relevance `<= 0.10`.
- Cấu hình được chọn tăng validation `Seen P@10` mà không làm `Unseen P@10` giảm đáng kể.
- Báo cáo frozen-test metrics cạnh baseline hiện tại `0.395 / 0.330`.

## Giai đoạn 2 — Cân bằng query sampling

Nếu hard negatives giúp hoặc trung tính, cân bằng số training pair cho từng tag query.

- Giới hạn các tag quá phổ biến ở mức ban đầu 30–50 positives/tag.
- Oversample các tag train-only hiếm đến mức tối thiểu phù hợp.
- Báo cáo kết quả theo nhóm tần suất tag, không chỉ lấy trung bình tổng.

Điều này quan trọng vì P@10 tính trung bình mỗi tag bằng nhau, trong khi pair pool hiện tại bị chi phối bởi tag phổ biến.

## Giai đoạn 3 — Query augmentation

Thêm biến thể câu query chỉ cho train-only tags: số ít/số nhiều, cách viết có gạch nối, biến thể chính tả và paraphrase đã review.

- Chỉ tạo augment sau khi split.
- Không augment held-out tag keys và không dùng test queries.
- Lưu audit file ánh xạ từng phrase mới với source tag.

Giai đoạn này được kỳ vọng giúp `Unseen tags P@10` nhiều hơn chỉ cân bằng pairs.

## Giai đoạn 4 — Ablation nguồn dữ liệu và quyết định deploy

So sánh độ hữu ích và noise của từng nguồn positive pairs:

1. Chỉ Genome relevance
2. Genome + MovieLens user tags
3. Genome + TMDB keywords
4. Tất cả nguồn

Với tag search, giữ cấu hình có frozen-test P@10 cao nhất. Với ứng dụng cần cả title lookup, đánh giá riêng Second model title-aware và nêu rõ trade-off giữa tag search và title search.
