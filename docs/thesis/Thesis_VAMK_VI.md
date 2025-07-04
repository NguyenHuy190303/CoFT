---
title: "CoFT: A Dual-Branch, Semi-Supervised Learning Framework for Time Series Analysis via Cross-Domain Co-Training"
author: "Nguyen Quoc Huy (e21010198)"
supervisor: "Tommi Rintala"
date: "July 2024"
degree: "Bachelor of Engineering"
programme: "Information Technology"
school: "Vaasa University of Applied Sciences VAMK"
university: "VAASAN AMMATTIKORKEAKOULU"
---

# **CoFT: A Dual-Branch, Semi-Supervised Learning Framework for Time Series Analysis via Cross-Domain Co-Training**

**Nguyen Quoc Huy**
**e21010198**

**Thesis**
**July 2024**
**Degree Programme in Information Technology**

**VAASAN AMMATTIKORKEAKOULU**
**UNIVERSITY OF APPLIED SCIENCES**

---
# TÓM TẮT

Tác giả: Nguyễn Quốc Huy
Tiêu đề luận văn: CoFT: Một Framework Học Bán Giám Sát Hai Nhánh cho Phân Tích Chuỗi Thời Gian thông qua Co-Training Đa Miền
Năm: 2024
Ngôn ngữ: Tiếng Anh
Số trang: [Sẽ được cập nhật]
Người hướng dẫn: [Sẽ điền sau]

Sự bùng nổ của dữ liệu chuỗi thời gian trên nhiều lĩnh vực đã gặp phải một trở ngại lớn: sự khan hiếm dữ liệu có nhãn cần thiết để huấn luyện các mô hình học sâu mạnh mẽ. Tình trạng "dữ liệu nhiều, nhãn ít" này đặc biệt nghiêm trọng trong các lĩnh vực có yêu cầu cao như y tế và giám sát công nghiệp, nơi việc chú thích dữ liệu rất tốn kém, mất thời gian và đòi hỏi chuyên môn sâu. Luận văn này giải quyết thách thức đó bằng cách đề xuất **CoFT (Co-training với miền Tần số và Thời gian)**, một framework học bán giám sát hai nhánh mới lạ. CoFT khai thác một cách độc đáo tính chất bổ sung của miền thời gian và tần số, vận hành chúng không phải như các đặc trưng để hợp nhất, mà là hai "khung nhìn" (views) độc lập có điều kiện cho một phương pháp co-training thực thụ.

Nghiên cứu chủ yếu sử dụng một quy trình huấn luyện sáu giai đoạn phức tạp được xây dựng dựa trên framework tiên tiến CA-TCC, mở rộng nó với một nhánh miền tần số song song. Sự đổi mới cốt lõi là một mô-đun co-training đa miền được điều phối bởi một hàm mất mát hỗn hợp (hybrid loss). Các thí nghiệm được thực hiện trên ba bộ dữ liệu benchmark công khai—HAR, Sleep-EDF, và Epilepsy—sau một quá trình tái tạo phương pháp tiền xử lý và phân chia dữ liệu của bài báo gốc một cách nghiêm ngặt và đầy thách thức để đảm bảo tính công bằng học thuật.

Kết quả thực nghiệm cho thấy framework CoFT vượt trội đáng kể để với mô hình cơ sở mạnh, đạt được cải thiện độ chính xác lên tới **+8.17%**. Một phát hiện quan trọng là hiện tượng "Càng ít càng tốt" (Less is More), một khám phá phản trực giác rằng trọng số co-training siêu nhỏ (λ_ct = 0.0001) là tối ưu để ngăn chặn "sự nhầm lẫn nhãn" (label confusion) và tối đa hóa hiệu suất. Công trình này không chỉ cung cấp một mô hình thực tiễn, hiệu suất cao mà còn mang lại những hiểu biết cơ bản về động lực của co-training đa miền và một phương pháp luận có nguyên tắc để chuyển giao các tham số đã học sang các bộ dữ liệu mới.

Từ khóa: Phân tích chuỗi thời gian, Học bán giám sát, Học tự giám sát, Co-Training, Học sâu, Miền tần số, Học tương phản, Khan hiếm nhãn.
---
# MỤC LỤC

TÓM TẮT ................................................................................ 2
DANH MỤC HÌNH ẢNH .................................................................... 5
DANH MỤC BẢNG ..................................................................... 6
CÁC TỪ VIẾT TẮT .................................................................... 7
BẢNG KÝ HIỆU ................................................................... 8

# 1 GIỚI THIỆU .................................................................... 9
1.1 Thách thức về sự khan hiếm nhãn trong phân tích chuỗi thời gian hiện đại .. 9
1.2 Mục tiêu và Câu hỏi nghiên cứu ............................. 11
1.2.1 Mục tiêu của luận văn .................................................... 11
1.2.2 Câu hỏi nghiên cứu ................................................. 11
1.3 Cấu trúc luận văn ................................................................. 12
1.4 Việc sử dụng AI trong luận văn này ...................................................... 12

# 2 CƠ SỞ LÝ THUYẾT VÀ LÝ LUẬN ........................................... 14
2.1 Các mô hình học từ dữ liệu có nhãn giới hạn ................. 14
2.2 Học tự giám sát: Nghệ thuật học từ chính dữ liệu 15
2.2.1 Các phương pháp dự đoán và tương phản ...................... 16
2.2.2 Tại sao lại là Học tương phản cho luận văn này? ................... 17
2.3 Giải phẫu của Học tương phản cho chuỗi thời gian .............. 18
2.3.1 Nền tảng: Tăng cường dữ liệu ........................ 18
2.3.2 Động cơ: Hàm mất mát NT-Xent .......................... 19
2.3.3 Mô hình cơ sở: Quy trình đa giai đoạn CA-TCC ................ 21
2.4 Hợp nhất đa miền: Từ hợp nhất đơn giản đến Co-Training ........ 28
2.4.1 Phổ các chiến lược hợp nhất .............................. 28
2.4.2 Tại sao lại là Co-Training? Một phương pháp có nguyên tắc hơn ............ 29
2.5 Kết luận: Xây dựng trên một nền tảng vững chắc ...................... 31

# 3 TRIỂN KHAI VÀ PHƯƠNG PHÁP LUẬN .................................... 32
3.1 Hành trình phương pháp luận: Lý giải và Thách thức ........ 32
3.1.1 Lựa chọn chiến trường: Lựa chọn mô hình cơ sở và benchmark 32
3.1.2 Thử thách về khả năng tái lập: Các rào cản kỹ thuật và dữ liệu ............................................................................ 33
3.2 Các nguyên tắc chỉ đạo cho khả năng tái lập .................................. 34
3.3 Công nghệ và Triển khai .......................................... 35
3.4 Các bộ dữ liệu benchmark .......................................................... 36
3.4.1 Nhận dạng hoạt động của con người (HAR) ............................. 37
3.4.2 Sleep-EDF (Phân loại giai đoạn giấc ngủ) ........................ 37
3.4.3 Epilepsy (Phát hiện co giật) .................................... 38
3.4.4 Phương pháp phân chia dữ liệu bán giám sát .............. 38
3.5 Framework CoFT: Kiến trúc và Quy trình ................ 39
3.5.1 Kiến trúc hai nhánh: Thiết kế và Triển khai .. 39
3.5.2 Hàm mất mát hỗn hợp: Một giải phẫu chi tiết ............. 43
3.5.3 Quy trình huấn luyện sáu giai đoạn: Công thức từng bước ...... 44
3.5.4 Tăng cường dữ liệu ................................................. 46

# 4 KẾT QUẢ VÀ PHÂN TÍCH ...................................................... 47
4.1 Trả lời Câu hỏi nghiên cứu 1: CoFT có thể vượt trội hơn một mô hình cơ sở tiên tiến không? .................................................................... 47
4.2 Trả lời Câu hỏi nghiên cứu 2: Nguồn gốc thực sự của việc tăng hiệu suất là gì? .................................................................... 49
4.3 Trả lời Câu hỏi nghiên cứu 3: Hành trình nghiên cứu để tối ưu hóa việc chuyển giao kiến thức .................................................................... 51
4.3.1 Giả thuyết ban đầu và những thất bại đầu tiên: Nguy cơ của sự liên kết mạnh ............................................................................ 51
4.3.2 Khám phá "Càng ít càng tốt": Một cuộc điều tra có hệ thống 52
4.3.3 Động lực của phương pháp Ensemble: Hiện tượng đảo ngược ....... 53
4.4 Trả lời Câu hỏi nghiên cứu 4: Các nguyên tắc có thể được chuyển giao sang các bộ dữ liệu mới không? ............................................................................ 54

# 5 KẾT LUẬN ..................................................................... 56
5.1 Tóm tắt hành trình nghiên cứu ...................................... 56
5.2 Hạn chế của nghiên cứu .................................................... 57
5.3 Hướng nghiên cứu trong tương lai ................................................. 58
5.4 Suy ngẫm cuối cùng ............................................................... 59

TÀI LIỆU THAM KHẢO ........................................................................... 60
PHỤ LỤC ........................................................................... 63

---
# DANH MỤC HÌNH ẢNH

Hình 1. Học tương phản hai mục tiêu trong CA-TCC. Mô hình học cách bất biến với các phép tăng cường thông qua Tương phản Thời gian và nhạy cảm với cấu trúc thời gian thông qua Tương phản Ngữ cảnh.
Hình 2. Tổng quan cấp cao về framework CoFT.
Hình 3. Kiến trúc của khối Transformer được sử dụng trong cả hai bộ mã hóa.
Hình 4. Cơ chế Tương phản Tần số.
Hình 5. Chiến lược huấn luyện đa giai đoạn, được điều chỉnh từ CA-TCC.

---
# DANH MỤC BẢNG

Bảng 1. Nghiên cứu cắt bỏ (Ablation Study) các thành phần trong TS-TCC và CA-TCC.
Bảng 2. Hiệu suất cuối cùng của CoFT so với mô hình cơ sở CA-TCC.
Bảng 3. Phân tích thống kê về mức tăng hiệu suất.
Bảng 4. Nghiên cứu cắt bỏ - Phân tích các yếu tố tăng hiệu suất của CoFT.
Bảng 5. Ảnh hưởng của trọng số Co-training (λ_ct) đến độ chính xác HAR 1%.
Bảng 6. Tương tác giữa phương pháp Ensemble và trọng số Co-training.
Bảng 7. Các tham số được chuyển giao và lý do.
Bảng 8. Các tham số huấn luyện và mô hình chung.
Bảng 9. Các siêu tham số cụ thể của CoFT.
Bảng 10. Các tham số học tương phản và tăng cường.

---
# CÁC TỪ VIẾT TẮT
| Viết tắt | Tên đầy đủ                                                |
| :----------- | :------------------------------------------------------- |
| **CoFT**     | Co-training với miền Tần số và Thời gian          |
| **SSL**      | Học Tự Giám Sát                                 |
| **CA-TCC**   | Tăng cường Tương phản - Phân cụm Tương phản Thời gian |
| **TS-TCC**   | Tương phản Thời gian và Ngữ cảnh (cho Chuỗi Thời gian)    |
| **FFT**      | Biến đổi Fourier Nhanh                                   |
| **HAR**      | Nhận dạng Hoạt động của Con người                               |
| **EEG**      | Điện não đồ                                     |
| **ECG**      | Điện tâm đồ                                        |
| **PSG**      | Đa ký giấc ngủ                                          |
| **REM**      | Chuyển động mắt nhanh                                       |
| **InfoTS**   | Tăng cường Chuỗi Thời gian dựa trên Lý thuyết Thông tin           |
| **NT-Xent**  | Cross-Entropy chuẩn hóa theo nhiệt độ              |
| **SupCon**   | Học Tương phản có Giám sát                          |

---

# BẢNG KÝ HIỆU

| Ký hiệu          | Mô tả                                                                 |
| :---------------- | :-------------------------------------------------------------------------- |
| \( \lambda_{ct} \)    | **Trọng số co-training:** Siêu tham số kiểm soát ảnh hưởng của mất mát co-training. Khám phá "Càng ít càng tốt" cho thấy giá trị siêu nhỏ (0.0001) là tối ưu. |
| \( \lambda_{cs} \)    | **Trọng số nhất quán:** Siêu tham số kiểm soát mất mát nhất quán đặc trưng giữa nhánh thời gian và tần số. |
| \( L_{total} \)      | Hàm mất mát hỗn hợp tổng thể được sử dụng để huấn luyện mô hình CoFT.                |
| \( L_{cotraining} \) | Thành phần mất mát bắt nguồn từ các nhãn giả do nhánh đối diện tạo ra. |
| \( L_{supervised} \)| Mất mát phân loại có giám sát tiêu chuẩn (ví dụ: Cross-Entropy).          |
| \( \tau \)          | **Nhiệt độ:** Một tham số tỷ lệ được sử dụng trong hàm mất mát tương phản (NT-Xent) và softmax để kiểm soát độ sắc nét của phân phối xác suất. |
| \( y_{true} \)       | Các nhãn thực tế (ground-truth) được cung cấp trong bộ dữ liệu.                            |
| \( y_{pseudo} \)    | Các nhãn được tạo ra bởi một nhánh của mô hình để huấn luyện nhánh còn lại.      |
| \( \theta \)         | Đại diện cho các tham số có thể học được của mô hình mạng nơ-ron.            |

</rewritten_file> 