# Kế hoạch Kiểm chứng và Phân tích Chuyên sâu (Verification & In-depth Analysis Plan)

**Ngày:** 25/06/2024
**Trạng thái:** `Đang thực thi`

## 1. Bối cảnh và Mục tiêu mới

- **Bối cảnh:** Thí nghiệm ban đầu cho thấy một "Baseline được tinh chỉnh" (CA-TCC với siêu tham số của CoFT) đạt `85.07%` accuracy, gần như bằng kết quả `85.31%` của cấu hình "Temporal Only". Điều này đặt ra nghi vấn lớn về nguồn gốc thực sự của sự cải thiện hiệu suất.
- **Mục tiêu mới:** Thiết kế và thực hiện một loạt thí nghiệm kiểm chứng để phân tách và định lượng chính xác sự đóng góp hiệu suất từ ba nguồn chính:
    1.  **Tinh chỉnh Siêu tham số (Hyperparameter Tuning):** Lợi ích có được chỉ bằng cách dùng bộ tham số tốt hơn.
    2.  **Kiến trúc Huấn luyện Song song (Parallel Architecture):** Lợi ích có được từ hiệu ứng "regularizer" của nhánh tần số, ngay cả khi chưa có co-training.
    3.  **Cơ chế Co-training (Co-training Mechanism):** Lợi ích có được từ việc trao đổi kiến thức qua pseudo-label.

## 2. Định nghĩa các Cấu hình Thí nghiệm

Chúng ta sẽ so sánh 4 cấu hình chính:

- **C1: Baseline Gốc (Original Baseline):** Mô hình CA-TCC gốc với các siêu tham số ban đầu.
    - *Trạng thái:* Đã có kết quả (`77.3%`).
- **C2: Baseline Tinh chỉnh (Tuned Baseline / CA-TCC++):** Mô hình CA-TCC gốc nhưng được chạy với bộ siêu tham số đã tối ưu của CoFT.
    - *Trạng thái:* Có kết quả sơ bộ (`85.07%`). **Cần chạy lại với nhiều seed để xác thực.**
- **C3: CoFT - Huấn luyện Song song (Temporal Only):** Mô hình CoFT được bật (`--enable_coft`), nhưng `lambda_ct=0` và dự đoán cuối cùng chỉ lấy từ nhánh temporal. Thí nghiệm này đo lường hiệu ứng regularizer của kiến trúc.
    - *Trạng thái:* Có kết quả sơ bộ (`85.31%`). **Cần chạy lại với nhiều seed để xác thực.**
- **C4: CoFT - Đầy đủ (Full Model):** Mô hình CoFT hoàn chỉnh với co-training được bật (`lambda_ct=0.0001`).
    - *Trạng thái:* Có kết quả sơ bộ (`85.47%`). **Cần chạy lại với nhiều seed để xác thực.**

## 3. Kế hoạch Thực thi Chi tiết

### Giai đoạn 1: Xác thực độ bền của kết quả (Robustness Validation)

Mục tiêu là chạy lại các cấu hình C2, C3, C4, mỗi cấu hình 3 lần với 3 `seed` khác nhau (0, 1, 2) để có kết quả trung bình (mean) và độ lệch chuẩn (std). Điều này loại bỏ yếu tố may mắn.

**Yêu cầu:** Cần có một cách để truyền `seed` vào `main.py`. Tôi sẽ tìm cách thêm cờ `--seed` nếu nó chưa tồn tại.

**Các nhiệm vụ (Tasks):**

- **Task 1.1: Chạy C2 (Tuned Baseline) x 3 seeds:**
    - `conda run -n CoFT python main.py --training_mode ft_1p --selected_dataset HAR --seed 0`
    - `conda run -n CoFT python main.py --training_mode ft_1p --selected_dataset HAR --seed 1`
    - `conda run -n CoFT python main.py --training_mode ft_1p --selected_dataset HAR --seed 2`
- **Task 1.2: Chạy C3 (Temporal Only) x 3 seeds:**
    - Thực thi thông qua `search.sh` hoặc script tương tự, thay đổi seed.
- **Task 1.3: Chạy C4 (Full Model) x 3 seeds:**
    - Thực thi thông qua `search.sh` hoặc script tương tự, thay đổi seed.

### Giai đoạn 2: Phân tích và Cập nhật Luận văn

Sau khi có kết quả trung bình đáng tin cậy từ Giai đoạn 1:

- **Task 2.1: Cập nhật Bảng Ablation Study:** Xây dựng lại bảng kết quả theo cấu trúc mới, thể hiện rõ sự cải thiện tuần tự.
- **Task 2.2: Viết lại phần Phân tích:** Diễn giải lại câu chuyện trong luận văn. Thay vì nói "CoFT cải thiện 8% so với baseline", câu chuyện sẽ là:
    1.  "Đầu tiên, chúng tôi thiết lập một baseline mạnh mẽ hơn bằng cách tinh chỉnh siêu tham số, đạt `X%` accuracy."
    2.  "Trên nền tảng đó, chỉ riêng việc áp dụng kiến trúc huấn luyện song song của CoFT đã mang lại thêm `Y%` cải thiện, chứng tỏ hiệu quả của cơ chế regularizer."
    3.  "Cuối cùng, việc kích hoạt cơ chế co-training đã đóng góp thêm `Z%` hiệu suất, đưa kết quả cuối cùng lên `T%`."

## 4. Bảng Kết quả Phân tích (Cấu trúc mới)

Bảng trong luận văn sẽ được thay thế bằng cấu trúc sau để thể hiện rõ quá trình:

| # | Cấu hình (Configuration)           | Accuracy (Mean ± Std) | Cải thiện (+/-)     | Nguồn gốc Cải thiện (Source of Gain)       |
| :-: | :--------------------------------- | :-------------------: | :-----------------: | :--------------------------------------- |
| 1 | **Baseline Gốc** (CA-TCC)          | 77.3%                 | -                   | -                                        |
| 2 | **Baseline Tinh chỉnh** (CA-TCC++)  | `Kết quả Task 1.1`    | `(C2) - (C1)`       | **Tinh chỉnh Siêu tham số**               |
| 3 | **CoFT - Huấn luyện Song song**      | `Kết quả Task 1.2`    | `(C3) - (2)`       | **Kiến trúc (Hiệu ứng Regularizer)**     |
| 4 | **CoFT - Đầy đủ** (Co-training)    | `Kết quả Task 1.3`    | `(C4) - (C3)`       | **Cơ chế Co-training**                   | 