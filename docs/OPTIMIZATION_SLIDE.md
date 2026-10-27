# Agent Optimization — Tóm tắt phần thực hiện

**So với commit trước:** cập nhật **11 file**, thêm **449 dòng**.

## Agent ban đầu

- Chatbot gọi LLM theo vòng lặp **tool-calling** để tìm phim, tra content warning và map chủ đề người dùng sợ.
- Có profile với hai mức preference: `never` (loại khỏi gợi ý) và `warn` (hiển thị cảnh báo).
- Backend mock lọc phim có chủ đề `never`; CLI hỗ trợ xem tool trace bằng `--verbose`.

## Dữ liệu & profile hiện tại

- **Movie dataset:** `data/mock/movies.json` — tập mẫu phát triển gồm **18 phim**, dùng tên phim và IMDb ID thật; rating/overview là dữ liệu mock xấp xỉ.
- **Content warnings:** `data/mock/topics.json` và `triggers.json` — topic, synonym và vote được tạo thủ công theo format DTDD-style; **không phải dữ liệu DoesTheDogDie (DTDD) thật**.
- **User profile:** `data/mock/users.json` — 3 profile demo (`arachno`, `dog_lover`, `fresh`) do nhóm tạo; preference lưu theo `topic_id` với `never`/`warn`. Thay đổi chỉ tồn tại trong bộ nhớ khi chạy app.
- **Hướng production:** ingest/cached DTDD API, join phim bằng IMDb ID và lưu profile vào database thay cho JSON mock.

## Phần đã bổ sung / tối ưu

| Hạng mục | Cải tiến | Giá trị |
| --- | --- | --- |
| An toàn | Thêm *safety gate* kiểm tra lại kết quả; nhận diện câu tránh nội dung ngay đầu lượt chat. | Không để phim chứa phobia lọt qua khi model/backend xử lý thiếu. |
| Cá nhân hoá | Thêm session memory: phim vừa gợi ý, truy vấn gần nhất và loại trừ tạm thời. | Follow-up có ngữ cảnh hơn. |
| Profile | Luồng đề xuất → xác nhận cho `never`, `warn`, `fine`; từ chối/mơ hồ không ghi dữ liệu. | Model không thể tự ý đổi preference. |
| Retrieval | Hybrid ranking: exact title → keyword match → rating; thêm `match_reasons`. | Kết quả liên quan và giải thích được. |
| Đánh giá | Thêm metrics: safety recall, warning accuracy, phrase-mapping accuracy; thêm cấu hình token/memory/retrieval. | Dễ đo lường và tinh chỉnh. |

**Kiểm thử:** `42 passed, 1 deselected`.
