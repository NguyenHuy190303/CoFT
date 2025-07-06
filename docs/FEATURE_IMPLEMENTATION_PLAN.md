# Kế hoạch Triển khai các Tính năng CoFT Nâng cao

Tài liệu này trình bày kế hoạch chi tiết để triển khai các thành phần nâng cao cho framework CoFT, như đã được vạch ra trong luận án. Kế hoạch được chia thành ba giai đoạn chính, đảm bảo mỗi bước xây dựng trên một nền tảng vững chắc.

---

## Giai đoạn 1: Triển khai các Phép Tăng cường Dữ liệu Miền Tần số (Semantically-Aware Frequency Augmentations)

**Mục tiêu:** Thay thế phép `jitter` cơ bản bằng các phép tăng cường dữ liệu chuyên biệt cho miền tần số để cung cấp tín hiệu huấn luyện mạnh mẽ và thực tế hơn cho nhánh tần số.

### 1.1. Các bước thực hiện

1.  **Tạo hàm tăng cường trong `dataloader/augmentations.py`:**
    *   **`add_spectral_noise(x_fft, noise_level)`:**
        *   **Đầu vào:** `x_fft` (tensor phức sau khi FFT), `noise_level` (float, độ lớn của nhiễu).
        *   **Logic:** Tạo một tensor nhiễu Gauss phức có cùng kích thước với `x_fft` và cộng nó vào.
        *   **Đầu ra:** Tensor phức đã được thêm nhiễu.
    *   **`mask_frequency_bands(x_fft, mask_ratio, num_masks)`:**
        *   **Đầu vào:** `x_fft` (tensor phức), `mask_ratio` (tỷ lệ phần trăm của dải tần bị che), `num_masks` (số lượng dải tần riêng biệt bị che).
        *   **Logic:**
            1.  Tính toán độ rộng của mỗi dải che (`mask_width = int(freq_axis_length * mask_ratio)`).
            2.  Lặp `num_masks` lần: Chọn một điểm bắt đầu ngẫu nhiên trên trục tần số và đặt các giá trị trong khoảng `[start, start + mask_width]` thành `(0+0j)`.
        *   **Đầu ra:** Tensor phức đã được che một hoặc nhiều dải tần.

2.  **Tích hợp vào Pipeline Dữ liệu:**
    *   **Sửa đổi hàm `DataTransform_FD` trong `dataloader/augmentations.py`:**
        *   Hàm này sẽ nhận tín hiệu miền thời gian (`sample`) làm đầu vào.
        *   **Bước 1:** Chuyển đổi `sample` sang miền tần số: `x_fft = torch.fft.rfft(sample, norm='ortho')`.
        *   **Bước 2:** Tạo hai "view" cho học tương phản:
            *   `view1_fft = add_spectral_noise(x_fft, noise_level=...)` (weak augmentation).
            *   `view2_fft = mask_frequency_bands(x_fft, mask_ratio=...)` (strong augmentation).
        *   **Bước 3:** Chuyển đổi cả hai view trở lại miền thời gian: `view1 = torch.fft.irfft(view1_fft, ...)` và `view2 = torch.fft.irfft(view2_fft, ...)`.
        *   **Đầu ra:** Trả về `(view1, view2)`.
    *   **Lưu ý:** Việc chuyển đổi qua lại giữa hai miền là cần thiết vì mô hình CNN đầu vào vẫn mong đợi dữ liệu dạng chuỗi thời gian.

3.  **Cập nhật Files Cấu hình (ví dụ: `config_files/HAR_Configs.py`):**
    *   Trong `config.augmentation`, thêm các tham số mới: `spectral_noise_level`, `freq_mask_ratio`, `num_freq_masks`. Điều này cho phép tinh chỉnh dễ dàng.

### 1.2. Tiêu chí hoàn thành (Definition of Done)

-   [ ] Các hàm `add_spectral_noise` và `mask_frequency_bands` được triển khai và có unit test riêng biệt để kiểm tra tính đúng đắn.
-   [ ] Hàm `DataTransform_FD` được cập nhật và có thể tạo ra hai view dữ liệu khác nhau.
-   [ ] Chạy thành công một pipeline huấn luyện hoàn chỉnh trên bộ dữ liệu HAR với các phép tăng cường mới.
-   [ ] Hoàn thành một ablation study nhỏ so sánh hiệu suất giữa: (1) Chỉ `Spectral Noise`, (2) `Masking`.

---

## Giai đoạn 2: Nâng cấp Backbone Nhánh Tần số với Spectral CNN

**Mục tiêu:** Thay thế kiến trúc Transformer đa dụng hiện tại của nhánh tần số bằng một kiến trúc CNN được thiết kế riêng để khai thác các đặc trưng của phổ tín hiệu, giải quyết "nút thắt cổ chai" về kiến trúc.

### 2.1. Các bước thực hiện

1.  **Tạo file mới `models/spectral_cnn.py`:**
    *   Định nghĩa lớp `SpectralCNN(nn.Module)`.
    *   **Kiến trúc đề xuất:**
        *   Sử dụng các lớp `nn.Conv1d` với kernel size và stride được thiết kế để nắm bắt các mẫu trên trục tần số (ví dụ: kernel lớn ở các lớp đầu để nắm bắt các dải tần rộng, nhỏ dần ở các lớp sau để tinh chỉnh).
        *   Xen kẽ các lớp `nn.BatchNorm1d` và `nn.ReLU` (hoặc `nn.GELU`).
        *   Sử dụng `nn.MaxPool1d` để giảm chiều và tăng tính bất biến.
        *   Lớp cuối cùng là `nn.AdaptiveAvgPool1d(1)` để tạo ra một vector đặc trưng có kích thước cố định, giúp tương thích với bất kỳ độ dài chuỗi đầu vào nào.

2.  **Tích hợp `SpectralCNN` vào Mô hình chính:**
    *   **Sửa đổi `models/model.py` (hoặc nơi `base_Model` được định nghĩa):**
        *   Trong `__init__`, thêm logic để khởi tạo `SpectralCNN` cho `self.freq_encoder` khi một cờ mới (ví dụ: `args.use_spectral_cnn`) được kích hoạt.
        ```python
        from models.spectral_cnn import SpectralCNN
        # ...
        if args.enable_coft:
            if args.use_spectral_cnn:
                self.freq_encoder = SpectralCNN(input_channels, ...)
            else:
                # Giữ lại kiến trúc cũ làm fallback
                self.freq_encoder = self.temporal_encoder 
        ```
    *   **Trong `forward` pass:** Đảm bảo dữ liệu sau khi qua FFT và các phép biến đổi phù hợp được đưa vào `self.freq_encoder`.

3.  **Cập nhật `main.py` và Files Cấu hình:**
    *   Thêm cờ `--use_spectral_cnn` vào `argparse` trong `main.py`.
    *   Thêm các tham số cấu hình cho `SpectralCNN` (số kênh, kích thước kernel, v.v.) vào các file config.

### 2.2. Tiêu chí hoàn thành

-   [ ] Lớp `SpectralCNN` được triển khai và hoạt động độc lập.
-   [ ] Chạy thành công pipeline huấn luyện hoàn chỉnh với `SpectralCNN` làm backbone cho nhánh tần số.
-   [ ] Hiệu suất (accuracy) của mô hình với `SpectralCNN` được so sánh với kiến trúc Transformer cũ trên bộ dữ liệu HAR.

---

## Giai đoạn 3: Tích hợp Cơ chế Tương tác Miền Nâng cao (Frequency Attention)

**Mục tiêu:** Vượt qua cơ chế co-training gián tiếp bằng cách cho phép hai nhánh "tương tác" trực tiếp với nhau ở cấp độ đặc trưng, tạo ra một luồng thông tin phong phú hơn.

### 3.1. Các bước thực hiện

1.  **Tạo/Sửa đổi `models/attention.py`:**
    *   Tạo một lớp `CrossAttention(nn.Module)`.
    *   **Logic:** Lớp này sẽ nhận 3 đầu vào: `query` (từ đặc trưng nhánh thời gian), `key` và `value` (cùng từ đặc trưng nhánh tần số).
    *   Thực hiện phép tính attention chuẩn: `output = softmax(Q @ K.T / sqrt(d_k)) @ V`.

2.  **Tích hợp `CrossAttention` vào Mô hình chính:**
    *   **Sửa đổi `models/model.py` hoặc `models/coft_cotraining.py`:**
    *   Thêm một cờ điều khiển, ví dụ `args.use_cross_attention`.
    *   Trong `forward` pass, sau khi có được các đặc trưng từ hai bộ mã hóa (`temporal_features`, `spectral_features`):
        1.  Khởi tạo (hoặc gọi) module `CrossAttention`.
        2.  Sử dụng `temporal_features` làm `query` và `spectral_features` làm `key` và `value` để tính toán `context_vector`.
        3.  Kết hợp `context_vector` này với `temporal_features` ban đầu (ví dụ, qua phép cộng hoặc concatenate) trước khi đưa vào lớp phân loại cuối cùng. `final_features = temporal_features + self.dropout(context_vector)`.

3.  **Xem xét lại Hàm Loss:**
    *   Khi sử dụng Cross-Attention, có thể `L_consistency` (loss tương phản giữa hai nhánh) không còn cần thiết hoặc cần giảm trọng số, vì sự tương đồng đã được khuyến khích trực tiếp qua attention. Cần thử nghiệm để xác định điều này.

### 3.2. Tiêu chí hoàn thành

-   [ ] Lớp `CrossAttention` được triển khai và hoạt động.
-   [ ] Pipeline huấn luyện chạy thành công với cơ chế cross-attention.
-   [ ] So sánh hiệu suất của mô hình với và không có cross-attention. Phân tích xem có thể loại bỏ `L_consistency` hay không. 