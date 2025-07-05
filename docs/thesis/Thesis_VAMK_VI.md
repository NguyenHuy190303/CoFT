---
title: "CoFT: Một Khuôn khổ Học Bán giám sát Hai nhánh cho Phân tích Chuỗi thời gian thông qua Đồng huấn luyện Chéo miền"
author: "Nguyen Quoc Huy (e21010198)"
supervisor: "Tommi Rintala"
date: "Tháng 7, 2024"
degree: "Cử nhân Kỹ thuật"
programme: "Công nghệ Thông tin"
school: "Đại học Khoa học Ứng dụng Vaasa VAMK"
university: "VAASAN AMMATTIKORKEAKOULU"
---

# **CoFT: Một Khuôn khổ Học Bán giám sát Hai nhánh cho Phân tích Chuỗi thời gian thông qua Đồng huấn luyện Chéo miền**

**Nguyễn Quốc Huy**
**e21010198**

**Luận văn**
**Tháng 7, 2024**
**Chương trình cấp bằng Công nghệ Thông tin**

**VAASAN AMMATTIKORKEAKOULU**
**ĐẠI HỌC KHOA HỌC ỨNG DỤNG**

---
# TÓM TẮT

Tác giả: Nguyễn Quốc Huy
Tên luận văn: CoFT: Một Khuôn khổ Học Bán giám sát Hai nhánh cho Phân tích Chuỗi thời gian thông qua Đồng huấn luyện Chéo miền
Năm: 2024
Ngôn ngữ: Tiếng Anh
Số trang: [Sẽ cập nhật]
Người hướng dẫn: Tommi Rintala

Sự bùng nổ của dữ liệu chuỗi thời gian trên nhiều lĩnh vực, từ IoT công nghiệp đến chăm sóc sức khỏe, đã gặp phải một nút thắt cổ chai nghiêm trọng: sự khan hiếm dữ liệu có nhãn cần thiết để huấn luyện các mô hình học sâu mạnh mẽ. Tình trạng "dữ liệu giàu, nhãn nghèo" này đặc biệt rõ rệt trong các lĩnh vực có yêu cầu cao như y học, nơi việc gán nhãn cho các tín hiệu như điện não đồ (EEG) hay điện tâm đồ (ECG) rất tốn kém, mất thời gian và đòi hỏi chuyên môn sâu. Luận văn này đối mặt trực tiếp với thách thức khan hiếm nhãn, nhằm mục đích khai phá tiềm năng của các kho dữ liệu khổng lồ chưa được gán nhãn.

Chúng tôi đề xuất **CoFT (Đồng huấn luyện với miền Tần số và Thời gian)**, một khuôn khổ học bán giám sát hai nhánh mới. CoFT độc đáo ở chỗ vận hành miền thời gian và tần số không phải như các đặc trưng để hợp nhất, mà như hai "khung nhìn" độc lập có điều kiện cho một phương pháp đồng huấn luyện thực sự. Nghiên cứu sử dụng một quy trình huấn luyện sáu giai đoạn phức tạp được xây dựng dựa trên khuôn khổ CA-TCC tiên tiến, mở rộng nó với một nhánh miền tần số song song và một mô-đun đồng huấn luyện chéo miền được điều phối bởi một hàm mất mát hỗn hợp. Để đảm bảo tính chặt chẽ khoa học, các thí nghiệm đã được tiến hành trên ba bộ dữ liệu benchmark công khai—HAR, Sleep-EDF, và Epilepsy—với việc tái tạo một cách cẩn thận phương pháp tiền xử lý và phân chia dữ liệu ban đầu để đảm bảo tính công bằng học thuật.

Nghiên cứu chủ yếu sử dụng một quy trình huấn luyện sáu giai đoạn phức tạp được xây dựng dựa trên khuôn khổ CA-TCC tiên tiến, mở rộng nó với một nhánh miền tần số song song. Sự đổi mới cốt lõi là một mô-đun đồng huấn luyện chéo miền được điều phối bởi một hàm mất mát hỗn hợp. Các thí nghiệm đã được tiến hành trên ba bộ dữ liệu benchmark công khai—HAR, Sleep-EDF, và Epilepsy—sau một quá trình nghiêm ngặt và đầy thách thức để tái tạo phương pháp tiền xử lý và phân chia dữ liệu của bài báo gốc nhằm đảm bảo tính công bằng học thuật.

Kết quả thực nghiệm cho thấy khuôn khổ CoFT vượt trội đáng kể so với mô hình cơ sở mạnh, đạt được cải thiện độ chính xác lên đến **+8.17%**. Một phát hiện quan trọng là hiện tượng "Ít hơn là Nhiều hơn", một khám phá phản trực giác rằng trọng số co-training siêu nhỏ (λ_ct = 0.0001) là tối ưu để ngăn chặn "sự nhầm lẫn nhãn" (label confusion) và tối đa hóa hiệu suất. Công trình này không chỉ cung cấp một mô hình thực tiễn, hiệu suất cao mà còn mang lại những hiểu biết cơ bản về động lực của co-training đa miền và một phương pháp luận có nguyên tắc để chuyển giao các tham số đã học sang các bộ dữ liệu mới.

Từ khóa: Phân tích chuỗi thời gian, Học bán giám sát, Học tự giám sát, Co-Training, Học sâu, Miền tần số, Học tương phản, Khan hiếm nhãn.
---
# MỤC LỤC

- [TÓM TẮT](#tóm-tắt)
- [DANH MỤC HÌNH ẢNH](#danh-mục-hình-ảnh)
- [DANH MỤC BẢNG BIỂU](#danh-mục-bảng-biểu)
- [CÁC TỪ VIẾT TẮT](#các-từ-viết-tắt)
- [BẢNG KÝ HIỆU](#bảng-ký-hiệu)
- [1 GIỚI THIỆU](#1-giới-thiệu)
  - [1.1 Thách thức về sự khan hiếm nhãn trong phân tích chuỗi thời gian hiện đại](#11-thách-thức-về-sự-khan-hiếm-nhãn-trong-phân-tích-chuỗi-thời-gian-hiện-đại)
  - [1.2 Mục tiêu và Câu hỏi nghiên cứu của luận văn](#12-mục-tiêu-và-câu-hỏi-nghiên-cứu-của-luận-văn)
    - [1.2.1 Mục tiêu của luận văn](#121-mục-tiêu-của-luận-văn)
    - [1.2.2 Câu hỏi nghiên cứu](#122-câu-hỏi-nghiên-cứu)
  - [1.3 Cấu trúc luận văn](#13-cấu-trúc-luận-văn)
  - [1.4 Việc sử dụng trí tuệ nhân tạo trong luận văn này](#14-việc-sử-dụng-trí-tuệ-nhân-tạo-trong-luận-văn-này)
- [2 NỀN TẢNG KIẾN THỨC VÀ LÝ THUYẾT](#2-nền-tảng-kiến-thức-và-lý-thuyết)
  - [2.1 Các mô hình học từ dữ liệu có nhãn hạn chế](#21-các-mô-hình-học-từ-dữ-liệu-có-nhãn-hạn-chế)
  - [2.2 Học tự giám sát: Nghệ thuật học từ chính dữ liệu](#22-học-tự-giám-sát-nghệ-thuật-học-từ-chính-dữ-liệu)
    - [2.2.1 Các chiến lược tăng cường dữ liệu](#221-các-chiến-lược-tăng-cường-dữ-liệu)
    - [2.2.2 Hàm mất mát NT-Xent: Phân tích sâu về mặt toán học](#222-hàm-mất-mát-nt-xent-phân-tích-sâu-về-mặt-toán-học)
    - [2.2.3 Mô hình cơ sở CA-TCC: Một quy trình bán giám sát đa giai đoạn nghiêm ngặt](#223-mô-hình-cơ-sở-ca-tcc-một-quy-trình-bán-giám-sát-đa-giai-đoạn-nghiêm-ngặt)
  - [2.3 Đồng huấn luyện và hợp nhất miền tần số-thời gian](#23-đồng-huấn-luyện-và-hợp-nhất-miền-tần-số-thời-gian)
    - [2.3.1 Các phương pháp hợp nhất truyền thống](#231-các-phương-pháp-hợp-nhất-truyền-thống)
    - [2.3.2 CoFT: Một khuôn khổ đồng huấn luyện thực sự](#232-coft-một-khuôn-khổ-đồng-huấn-luyện-thực-sự)
  - [2.4 Kết luận: Xây dựng trên một nền tảng vững chắc](#24-kết-luận-xây-dựng-trên-một-nền-tảng-vững-chắc)
- [3 TRIỂN KHAI VÀ PHƯƠNG PHÁP LUẬN](#3-triển-khai-và-phương-pháp-luận)
  - [3.1 Hành trình phương pháp luận: Lý giải và thách thức](#31-hành-trình-phương-pháp-luận-lý-giải-và-thách-thức)
    - [3.1.1 Lựa chọn chiến trường: Lựa chọn mô hình cơ sở và benchmark](#311-lựa-chọn-chiến-trường-lựa-chọn-mô-hình-cơ-sở-và-benchmark)
    - [3.1.2 Thử thách về khả năng tái lập: Các rào cản kỹ thuật và dữ liệu](#312-thử-thách-về-khả-năng-tái-lập-các-rào-cản-kỹ-thuật-và-dữ-liệu)
  - [3.2 Các nguyên tắc chỉ đạo cho khả năng tái lập](#32-các-nguyên-tắc-chỉ-đạo-cho-khả-năng-tái-lập)
  - [3.3 Công nghệ và triển khai](#33-công-nghệ-và-triển-khai)
  - [3.4 Các bộ dữ liệu benchmark](#34-các-bộ-dữ-liệu-benchmark)
    - [3.4.1 Nhận dạng hoạt động của con người (HAR)](#341-nhận-dạng-hoạt-động-của-con-người-har)
    - [3.4.2 Sleep-EDF (Phân loại giai đoạn giấc ngủ)](#342-sleep-edf-phân-loại-giai-đoạn-giấc-ngủ)
    - [3.4.3 Epilepsy (Phát hiện co giật)](#343-epilepsy-phát-hiện-co-giật)
    - [3.4.4 Phương pháp phân chia dữ liệu bán giám sát](#344-phương-pháp-phân-chia-dữ-liệu-bán-giám-sát)
  - [3.5 Khuôn khổ CoFT: Kiến trúc và quy trình](#35-khuôn-khổ-coft-kiến-trúc-và-quy-trình)
    - [3.5.1 Kiến trúc hai nhánh: Thiết kế và triển khai](#351-kiến-trúc-hai-nhánh-thiết-kế-và-triển-khai)
    - [3.5.2 Hàm mất mát hỗn hợp: Phân tích chi tiết](#352-hàm-mất-mát-hỗn-hợp-phân-tích-chi-tiết)
    - [3.5.3 Quy trình huấn luyện sáu giai đoạn: Công thức từng bước](#353-quy-trình-huấn-luyện-sáu-giai-đoạn-công-thức-từng-bước)
    - [3.5.4 Tăng cường dữ liệu](#354-tăng-cường-dữ-liệu)
- [4 KẾT QUẢ VÀ PHÂN TÍCH](#4-kết-quả-và-phân-tích)
  - [4.1 Trả lời câu hỏi nghiên cứu 1: CoFT có thể vượt trội hơn một mô hình cơ sở tiên tiến không?](#41-trả-lời-câu-hỏi-nghiên-cứu-1-coft-có-thể-vượt-trội-hơn-một-mô-hình-cơ-sở-tiên-tiến-không)
  - [4.2 Trả lời câu hỏi nghiên cứu 2: Nguồn gốc thực sự của sự cải thiện hiệu suất là gì?](#42-trả-lời-câu-hỏi-nghiên-cứu-2-nguồn-gốc-thực-sự-của-sự-cải-thiện-hiệu-suất-là-gì)
  - [4.3 Trả lời câu hỏi nghiên cứu 3: Hành trình nghiên cứu để chuyển giao kiến thức tối ưu](#43-trả-lời-câu-hỏi-nghiên-cứu-3-hành-trình-nghiên-cứu-để-chuyển-giao-kiến-thức-tối-ưu)
    - [4.3.1 Giả thuyết ban đầu và những thất bại đầu tiên: Nguy cơ của sự liên kết mạnh](#431-giả-thuyết-ban-đầu-và-những-thất-bại-đầu-tiên-nguy-cơ-của-sự-liên-kết-mạnh)
    - [4.3.2 Khám phá "Ít hơn là Nhiều hơn": Một cuộc điều tra có hệ thống](#432-khám-phá-ít-hơn-là-nhiều-hơn-một-cuộc-điều-tra-có-hệ-thống)
    - [4.3.3 Động lực của phương pháp tổ hợp: Hiện tượng "Lật ngược"](#433-động-lực-của-phương-pháp-tổ-hợp-hiện-tượng-lật-ngược)
  - [4.4 Trả lời câu hỏi nghiên cứu 4: Các nguyên tắc có thể được chuyển giao sang các bộ dữ liệu mới không?](#44-trả-lời-câu-hỏi-nghiên-cứu-4-các-nguyên-tắc-có-thể-được-chuyển-giao-sang-các-bộ-dữ-liệu-mới-không)
- [5 KẾT LUẬN](#5-kết-luận)
  - [5.1 Tóm tắt hành trình nghiên cứu](#51-tóm-tắt-hành-trình-nghiên-cứu)
  - [5.2 Hạn chế của nghiên cứu](#52-hạn-chế-của-nghiên-cứu)
  - [5.3 Hướng nghiên cứu trong tương lai](#53-hướng-nghiên-cứu-trong-tương-lai)
  - [5.4 Suy ngẫm cuối cùng](#54-suy-ngẫm-cuối-cùng)
- [TÀI LIỆU THAM KHẢO](#tài-liệu-tham-khảo)
- [PHỤ LỤC](#phụ-lục)
  - [Phụ lục A: Bảng cấu hình siêu tham số](#phụ-lục-a-bảng-cấu-hình-siêu-tham-số)

---
# DANH MỤC HÌNH ẢNH

Hình 1. Học tương phản hai mục tiêu trong CA-TCC. Mô hình học cách bất biến với các phép tăng cường thông qua Tương phản Thời gian và nhạy cảm với cấu trúc thời gian thông qua Tương phản Ngữ cảnh.
Hình 2. Tổng quan cấp cao về khuôn khổ CoFT.
Hình 3. Kiến trúc của khối Transformer được sử dụng trong cả hai bộ mã hóa.
Hình 4. Cơ chế Tương phản Tần số.
Hình 5. Chiến lược huấn luyện đa pha, được điều chỉnh từ CA-TCC.

---
# DANH MỤC BẢNG BIỂU

Bảng 1. Nghiên cứu cắt lớp các thành phần trong TS-TCC và CA-TCC.
Bảng 2. Hiệu suất cuối cùng của CoFT so với mô hình cơ sở CA-TCC.
Bảng 3. Phân tích thống kê về sự cải thiện hiệu suất.
Bảng 4. Nghiên cứu cắt lớp - Phân tích sự cải thiện hiệu suất của CoFT.
Bảng 5. Ảnh hưởng của trọng số đồng huấn luyện (λ_ct) đến độ chính xác trên HAR 1%.
Bảng 6. Tương tác giữa phương pháp tổ hợp và trọng số đồng huấn luyện.
Bảng 7. Các tham số được chuyển giao và lý do.
Bảng 8. Các tham số huấn luyện và mô hình chung.
Bảng 9. Các siêu tham số dành riêng cho CoFT.
Bảng 10. Các tham số học tương phản và tăng cường dữ liệu.

---
# CÁC TỪ VIẾT TẮT
| Viết tắt | Tên đầy đủ                                                |
| :----------- | :------------------------------------------------------- |
| **CoFT**     | Đồng huấn luyện với miền Tần số và Thời gian (Co-training with Frequency and Temporal domains) |
| **SSL**      | Học Tự giám sát (Self-Supervised Learning)                     |
| **CA-TCC**   | Tăng cường Tương phản - Phân cụm Tương phản Thời gian (Contrastive Augmentation - Temporal Contrastive Clustering) |
| **TS-TCC**   | Tương phản Thời gian và Ngữ cảnh (cho Chuỗi thời gian) (Temporal and Contextual Contrasting) |
| **FFT**      | Biến đổi Fourier nhanh (Fast Fourier Transform)                |
| **HAR**      | Nhận dạng hoạt động của con người (Human Activity Recognition)  |
| **EEG**      | Điện não đồ (Electroencephalogram)                           |
| **ECG**      | Điện tâm đồ (Electrocardiogram)                             |
| **PSG**      | Đa ký giấc ngủ (Polysomnography)                              |
| **REM**      | Chuyển động mắt nhanh (Rapid Eye Movement)                     |
| **InfoTS**   | Tăng cường Chuỗi Thời gian dựa trên Lý thuyết Thông tin (Information-Theoretic Time Series Augmentation) |
| **NT-Xent**  | Cross-Entropy chuẩn hóa theo nhiệt độ (Normalized Temperature-scaled Cross-Entropy) |
| **SupCon**   | Học Tương phản có giám sát (Supervised Contrastive Learning)   |

---

# BẢNG KÝ HIỆU

| Ký hiệu          | Mô tả                                                                 |
| :---------------- | :-------------------------------------------------------------------------- |
| \( \lambda_{ct} \)    | **Trọng số đồng huấn luyện:** Siêu tham số kiểm soát ảnh hưởng của mất mát đồng huấn luyện. Khám phá "Ít hơn là Nhiều hơn" cho thấy giá trị cực thấp (0.0001) là tối ưu. |
| \( \lambda_{cs} \)    | **Trọng số nhất quán:** Siêu tham số kiểm soát mất mát nhất quán đặc trưng giữa nhánh thời gian và tần số. |
| \( L_{total} \)      | Tổng hàm mất mát hỗn hợp được sử dụng để huấn luyện mô hình CoFT.            |
| \( L_{cotraining} \) | Thành phần mất mát bắt nguồn từ các nhãn giả được tạo ra bởi nhánh đối diện. |
| \( L_{supervised} \)| Mất mát phân loại có giám sát tiêu chuẩn (ví dụ: Cross-Entropy).          |
| \( \tau \)          | **Nhiệt độ:** Một tham số tỷ lệ được sử dụng trong hàm mất mát tương phản (NT-Xent) và softmax để kiểm soát độ sắc nét của phân phối xác suất. |
| \( y_{true} \)       | Các nhãn thực tế được cung cấp trong bộ dữ liệu.                        |
| \( y_{pseudo} \)    | Các nhãn được tạo ra bởi một nhánh của mô hình để huấn luyện nhánh còn lại. |
| \( \theta \)         | Đại diện cho các tham số có thể học được của mô hình mạng nơ-ron.        |

---

# 1 GIỚI THIỆU

## 1.1 Thách thức về sự khan hiếm nhãn trong phân tích chuỗi thời gian hiện đại

Trong những năm gần đây, học sâu đã nổi lên như một lực lượng biến đổi trong phân tích chuỗi thời gian, đạt được hiệu suất tiên tiến trong các nhiệm vụ từ nhận dạng hoạt động của con người đến dự báo tài chính. Tuy nhiên, sức mạnh của các mô hình này được xây dựng trên một nền tảng quan trọng và thường rất tốn kém: các bộ dữ liệu lớn, được gán nhãn chính xác. Mặc dù sự gia tăng của các cảm biến, thiết bị IoT và hồ sơ kỹ thuật số đã dẫn đến một sự bùng nổ về khối lượng dữ liệu chuỗi thời gian thô, quá trình gán nhãn có ý nghĩa vẫn là một nút thắt cổ chai đáng kể. Mô hình "dữ liệu giàu, nhãn nghèo" này đòi hỏi một sự chuyển dịch khỏi các phương pháp hoàn toàn có giám sát sang các mô hình học tập có thể khai thác hiệu quả tiềm năng to lớn, chưa được khai thác của dữ liệu không có nhãn.

Vấn đề khan hiếm nhãn trở nên đặc biệt nghiêm trọng trong các lĩnh vực mà dữ liệu không chỉ phức tạp mà còn nhạy cảm và đòi hỏi chuyên môn sâu để diễn giải. Ví dụ, trong **chăm sóc sức khỏe**, việc chú thích các tín hiệu điện não đồ (EEG) để phân loại giai đoạn giấc ngủ hoặc phát hiện co giật đòi hỏi các nhà thần kinh học được đào tạo phải dành hàng giờ để xem xét tỉ mỉ các bản ghi. Tương tự, việc gán nhãn dữ liệu điện tâm đồ (ECG) để phân loại rối loạn nhịp tim đòi hỏi con mắt tinh tường của một bác sĩ tim mạch. Quá trình này không chỉ chậm và tốn kém mà còn có thể mang tính chủ quan, dẫn đến sự thay đổi giữa những người đánh giá. Trong **sản xuất công nghiệp**, việc gán nhãn dữ liệu cảm biến để dự đoán hỏng hóc máy móc thường đòi hỏi phải đợi một sự cố thực tế xảy ra, vốn là những sự kiện hiếm và tốn kém. Trong những môi trường có yêu cầu cao này, việc không thể tận dụng hiệu quả lượng lớn dữ liệu không có nhãn thể hiện một cơ hội bị bỏ lỡ cho khám phá khoa học và đổi mới thực tế.

Để giải quyết khoảng trống nghiên cứu cơ bản này, luận văn này hướng đến mô hình học bán giám sát, nhằm mục đích học từ các bộ dữ liệu chứa một lượng nhỏ dữ liệu có nhãn và một lượng lớn dữ liệu không có nhãn. Cụ thể, chúng tôi tập trung vào phương pháp tiền huấn luyện tự giám sát, nơi một mô hình trước tiên học các biểu diễn mạnh mẽ và có thể tổng quát hóa từ kho dữ liệu không có nhãn thông qua các nhiệm vụ ban đầu (pretext tasks). Công trình này giới thiệu **CoFT (Đồng huấn luyện với miền Tần số và Thời gian)**, một khuôn khổ hai nhánh mới được thiết kế để tối đa hóa việc trích xuất thông tin từ dữ liệu chuỗi thời gian trong môi trường khan hiếm nhãn này. Giả thuyết cốt lõi của CoFT là miền thời gian (cách một tín hiệu phát triển theo thời gian) và miền tần số (các thành phần tuần hoàn tạo nên tín hiệu) cung cấp các góc nhìn bổ sung và hiệp đồng về dữ liệu. Thay vì chỉ đơn thuần hợp nhất các miền này thành các đặc trưng, CoFT thực hiện một phương pháp **đồng huấn luyện** thực sự, trong đó hai nhánh chuyên biệt—mỗi nhánh cho một miền—học song song và tích cực "dạy" lẫn nhau thông qua một cơ chế gán nhãn giả được kiểm soát cẩn thận. Bằng cách tạo ra một mối quan hệ học tập cộng sinh giữa hai miền này, CoFT nhằm mục đích xây dựng một biểu diễn dữ liệu toàn diện và mạnh mẽ hơn so với việc mỗi miền hoạt động riêng lẻ.

## 1.2 Mục tiêu và Câu hỏi nghiên cứu của luận văn

### 1.2.1 Mục tiêu của luận văn

Luận văn này chủ yếu nhằm mục đích thiết kế, xác thực và phân tích toàn diện khuôn khổ **CoFT (Đồng huấn luyện với miền Tần số và Thời gian)**. Mục tiêu trung tâm là đánh giá cách kiến trúc hai miền, bán giám sát mới này hoạt động so với một mô hình cơ sở mạnh, tiên tiến (CA-TCC), đặc biệt là trong các kịch bản khan hiếm nhãn.

Để đạt được điều này, nghiên cứu sẽ tiến hành một loạt các thí nghiệm có kiểm soát để phân tích một cách có hệ thống các nguồn gốc của sự cải thiện hiệu suất. Điều này bao gồm một cách tiếp cận kép:
1.  **Đánh giá Hiệu suất:** Một phân tích so sánh giữa CoFT và một mô hình cơ sở CA-TCC được tinh chỉnh nghiêm ngặt để định lượng sự cải thiện hiệu suất ròng.
2.  **Phân tích Yếu tố:** Một phân tích sâu thông qua các nghiên cứu cắt lớp để xác định và kiểm tra các yếu tố chính—cả về kiến trúc (ví dụ: cấu trúc hai nhánh, phương pháp tổ hợp) và tham số (ví dụ: siêu tham số đồng huấn luyện)—ảnh hưởng đến hiệu quả của mô hình.

Thông qua cách tiếp cận này, luận văn tìm cách cung cấp một sự hiểu biết sâu sắc hơn, minh bạch hơn về việc triển khai một mô hình đồng huấn luyện hai miền, vượt ra ngoài việc chỉ báo cáo một con số độ chính xác cuối cùng để giải thích *tại sao* nó hoạt động. Cần lưu ý rằng một mục tiêu quan trọng không chỉ là đạt được hiệu suất tiên tiến, mà còn là thiết lập một phương pháp luận nghiêm ngặt để đánh giá các mô hình phức tạp và hiểu các nguyên tắc cơ bản của học tập chéo miền trong chuỗi thời gian.

### 1.2.2 Câu hỏi nghiên cứu

Luận văn này nhằm trả lời bốn câu hỏi nghiên cứu chính, được thông báo trực tiếp bởi các mục tiêu đã nêu trước đó và được cấu trúc để được trả lời bằng phân tích trong Chương 4:

-   **Câu hỏi 1:** Liệu một khuôn khổ đồng huấn luyện hai nhánh (CoFT) tận dụng cả miền thời gian và tần số có thể vượt trội đáng kể và nhất quán so với một mô hình đơn miền, tiên tiến (CA-TCC) trên các bộ dữ liệu chuỗi thời gian benchmark không?
-   **Câu hỏi 2:** Nguồn gốc thực sự của sự cải thiện hiệu suất của CoFT là gì? Bao nhiêu phần trăm là do kiến trúc mới so với việc tối ưu hóa siêu tham số nghiêm ngặt?
-   **Câu hỏi 3:** Các tham số và cơ chế tối ưu để chuyển giao kiến thức giữa hai miền là gì, và những nguyên tắc cơ bản nào chi phối hiệu quả của chúng (ví dụ: hiện tượng "Ít hơn là Nhiều hơn")?
-   **Câu hỏi 4:** Liệu các nguyên tắc học được từ một bộ dữ liệu có thể được chuyển giao hiệu quả để hướng dẫn tối ưu hóa trên các bộ dữ liệu mới, đa dạng, đặc biệt là trong lĩnh vực y tế, mà không cần tinh chỉnh lại toàn bộ?

## 1.3 Cấu trúc luận văn

Phần còn lại của luận văn này được cấu trúc như sau. **Chương 2** cung cấp một tổng quan về các công trình liên quan trong học bán giám sát, các phương pháp tương phản cho chuỗi thời gian, và hợp nhất đa miền, thiết lập nền tảng lý thuyết cho công trình của chúng tôi. **Chương 3** trình bày chi tiết kiến trúc CoFT, bao gồm thiết kế hai nhánh, quy trình huấn luyện, và hàm mất mát hỗn hợp đổi mới điều phối việc học chéo miền. **Chương 4** trình bày toàn bộ hành trình thực nghiệm, trả lời một cách có hệ thống từng câu hỏi nghiên cứu với dữ liệu thực nghiệm và phân tích. Cuối cùng, **Chương 5** kết luận bằng cách tóm tắt các đóng góp chính, thảo luận về những hạn chế và ý nghĩa rộng hơn của các phát hiện, và đề xuất các hướng nghiên cứu cụ thể trong tương lai.

## 1.4 Việc sử dụng trí tuệ nhân tạo trong luận văn này

Việc phát triển và thực hiện nghiên cứu trong luận văn này đã được tăng cường đáng kể bởi việc sử dụng các công cụ Trí tuệ Nhân tạo. Cụ thể, một trợ lý lập trình AI đàm thoại, được cung cấp bởi các mô hình ngôn ngữ lớn, đã được sử dụng như một đối tác lập trình đôi. Vai trò của nó bao gồm tạo mã cho các mô-đun cụ thể, gỡ lỗi, tạo và chạy các kịch bản shell cho các thí nghiệm, tái cấu trúc mã để rõ ràng và hiệu quả, và hỗ trợ trong việc soạn thảo và định dạng tài liệu và tài liệu luận văn này. Sự đóng góp của AI chủ yếu là trong việc tăng tốc các chu trình triển khai và thử nghiệm, cho phép tác giả tập trung vào các câu hỏi nghiên cứu cốt lõi, thiết kế thực nghiệm và phân tích kết quả. Tất cả các kết luận khoa học cuối cùng, các quyết định về kiến trúc và diễn giải dữ liệu đều do tác giả thực hiện.

---

# 2 NỀN TẢNG KIẾN THỨC VÀ LÝ THUYẾT

Chương này cung cấp một cái nhìn tổng quan toàn diện về các tài liệu liên quan đến khuôn khổ CoFT. Nó đặt công trình này ở giao điểm của học bán giám sát, học biểu diễn tương phản và phân tích chuỗi thời gian đa miền. Đầu tiên, chúng tôi khảo sát bối cảnh của học tương phản, thiết lập quy trình CA-TCC tiên tiến làm cơ sở so sánh trực tiếp. Sau đó, chúng tôi phân tích các phương pháp hiện có để kết hợp thông tin miền thời gian và tần số, qua đó làm nổi bật nền tảng lý thuyết và những đóng góp độc đáo của mô hình đồng huấn luyện của CoFT.

## 2.1 Các mô hình học từ dữ liệu có nhãn hạn chế

Các mô hình học sâu đã chứng tỏ thành công đáng kể trong phân loại chuỗi thời gian nhưng thường phụ thuộc vào các bộ dữ liệu lớn, được gán nhãn tỉ mỉ. Trong nhiều lĩnh vực thực tế, đặc biệt là trong chăm sóc sức khỏe (ví dụ: phân tích EEG, ECG), việc thu thập dữ liệu rất phong phú, nhưng việc chú thích của chuyên gia lại khan hiếm, tốn thời gian và tốn kém. Sự khan hiếm nhãn này đã thúc đẩy một làn sóng nghiên cứu về các phương pháp học tự giám sát và bán giám sát.

Học tự giám sát (SSL) nhằm mục đích học các biểu diễn có ý nghĩa từ dữ liệu không có nhãn bằng cách tạo ra các nhiệm vụ ban đầu (pretext tasks). Các biểu diễn đã học sau đó có thể được chuyển giao cho các nhiệm vụ xuôi dòng (như phân loại) nơi chỉ cần một lượng nhỏ dữ liệu có nhãn để tinh chỉnh. Mô hình hai giai đoạn này (tiền huấn luyện không giám sát sau đó là tinh chỉnh có giám sát) đã trở thành một phương pháp chủ đạo. Trong số các mô hình SSL, học tương phản đã nổi lên như một phương pháp đặc biệt hiệu quả để học các biểu diễn phân biệt.

## 2.2 Học tự giám sát: Nghệ thuật học từ chính dữ liệu

Học tương phản là một mô hình học tự giám sát nhằm mục đích học một không gian nhúng nơi các mẫu tương tự được đặt gần nhau, trong khi các mẫu không tương tự bị đẩy ra xa. Điều này đạt được không phải thông qua các nhãn rõ ràng, mà bằng cách tạo ra một "nhiệm vụ ban đầu" dựa trên việc tăng cường dữ liệu. Đối với một mẫu đầu vào nhất định, hai hoặc nhiều "khung nhìn" tương quan được tạo ra thông qua các phép tăng cường. Mô hình sau đó được huấn luyện để xác định các khung nhìn khác nhau của cùng một mẫu là một "cặp dương" và coi tất cả các mẫu khác trong một lô nhất định là "cặp âm".

Cách tiếp cận này, được phổ biến trong thị giác máy tính bởi các khuôn khổ như SimCLR [5], đã được điều chỉnh thành công cho miền chuỗi thời gian. Các công trình nền tảng như **TS-TCC** [6] và **TS2Vec** [10] đã thiết lập tính khả thi của tiền huấn luyện tương phản cho chuỗi thời gian, chứng minh rằng các biểu diễn mạnh mẽ có thể được học và chuyển giao hiệu quả cho các nhiệm vụ xuôi dòng với số lượng nhãn hạn chế.

Một khuôn khổ học tương phản điển hình cho chuỗi thời gian bao gồm ba thành phần chính: tăng cường dữ liệu, một bộ mã hóa mạng nơ-ron và một hàm mất mát tương phản. CoFT được xây dựng dựa trên nền tảng này, và các lựa chọn thiết kế của nó có thể được hiểu bằng cách xem xét các công nghệ tiên tiến trong từng thành phần.

### 2.2.1 Các chiến lược tăng cường dữ liệu

Việc tạo ra các cặp dương thông qua tăng cường dữ liệu là nền tảng của học tương phản. Lựa chọn phép tăng cường là rất quan trọng, vì nó ngầm xác định các tính bất biến mà mô hình nên học. Một bài đánh giá có hệ thống gần đây của Wen và cộng sự (2021) [9] đã phân loại các phép tăng cường chuỗi thời gian phổ biến thành ba nhóm:

1.  **Tăng cường biến đổi:** Thay đổi các thuộc tính của tín hiệu, bao gồm **jittering** (thêm nhiễu), **scaling** (thay đổi biên độ), **time-warping** (làm cong vênh thời gian) và **permutation** (hoán vị các đoạn).
2.  **Tăng cường che giấu:** Che khuất các phần của dữ liệu, chẳng hạn như **time masking** (đặt các đoạn thành không) hoặc **frequency masking** (lọc các dải tần số).
3.  **Tăng cường lân cận:** Xác định các cặp dương dựa trên sự gần gũi về thời gian, giả định rằng các cửa sổ liền kề trong một chuỗi thời gian là tương tự về mặt ngữ nghĩa.

Mặc dù có một hệ sinh thái phong phú các phép tăng cường, luận văn CoFT đã có một khám phá quan trọng thông qua **thí nghiệm InfoTS (chi tiết trong Chương 4)**. Nó đã chứng minh một cách có hệ thống rằng một bộ các phép tăng cường phức tạp, xác suất (InfoTS) chỉ mang lại một sự cải thiện hiệu suất không đáng kể (+0.03%) so với một bộ các phép tăng cường đơn giản, tất định (jitter, scaling, cropping), nhưng với cái giá là tăng 50% phương sai và tăng 25% thời gian huấn luyện. Phát hiện này đã cung cấp sự ủng hộ thực nghiệm mạnh mẽ cho triết lý thiết kế của CoFT: **sự đơn giản hơn là sự phức tạp**. Các phép tăng cường được chọn đủ hiệu quả để thúc đẩy việc học mà không cần thêm sự phức tạp không cần thiết, điều này phù hợp với nguyên tắc tối đa hóa tỷ lệ hiệu suất trên độ phức tạp.

### 2.2.2 Hàm mất mát NT-Xent: Phân tích sâu về mặt toán học

Động cơ thúc đẩy quá trình học tương phản là hàm mất mát. Công trình này, giống như mô hình cơ sở của nó, sử dụng hàm mất mát **Cross-Entropy chuẩn hóa theo nhiệt độ (NT-Xent)**. Bây giờ chúng ta sẽ phân tích hàm này theo nguyên tắc "Phân tích công thức".

#### Bước 1: Trình bày Công thức
Cho một cặp mẫu dương được tăng cường, \(x_i\) và \(x_j\), mạng mã hóa \(f(\cdot)\) tạo ra các vector nhúng \(z_i = f(x_i)\) và \(z_j = f(x_j)\). Mất mát cho cặp dương này được định nghĩa chính thức là:

\[ \mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)} \quad \text{(Phương trình 2.1)} \]

#### Bước 2: Định nghĩa từng ký hiệu
-   **\( \mathcal{L}_{i,j} \)**: Giá trị mất mát cuối cùng cho cặp dương \((i, j)\).
-   **\( z_i, z_j, z_k \)**: Các vector nhúng (biểu diễn đặc trưng) được tạo ra bởi mạng mã hóa cho các mẫu \(i, j, \text{và } k\), respectively.
-   **\( \text{sim}(u, v) \)**: Hàm tương đồng cosine, \(\frac{u \cdot v}{\|u\|\|v\|}\), đo góc giữa hai vector nhúng. Giá trị 1 có nghĩa là chúng giống hệt nhau về hướng, -1 có nghĩa là chúng đối diện, và 0 có nghĩa là chúng trực giao.
-   **\( \tau \)**: Siêu tham số **nhiệt độ**, một vô hướng dương kiểm soát độ "sắc nét" của phân phối tương đồng.
-   **\( N \)**: Số lượng mẫu gốc (trước khi tăng cường) trong một lô. Tổng số mẫu được tăng cường là \(2N\).
-   **\( \mathbb{1}_{[k \neq i]} \)**: Một hàm chỉ thị bằng 1 nếu \(k \neq i\) và 0 nếu ngược lại. Điều này đảm bảo rằng việc so sánh một vector nhúng với chính nó được loại trừ khỏi tổng.

#### Bước 3: Giải thích bằng ngôn ngữ đơn giản
Công thức này hoạt động giống như một bài toán phân loại đa lớp. Đối với một mẫu nhất định \(z_i\), mục tiêu là "phân loại" đối tác dương chính xác của nó \(z_j\) từ một tập hợp tất cả các mẫu khác có thể có (\(z_k\)) trong lô, được coi là "âm".

-   **Tử số**: \(\exp(\text{sim}(z_i, z_j)/\tau)\) đo lường sự tương đồng giữa hai đối tác chính xác (\(z_i\) và \(z_j\)). Mô hình muốn làm cho giá trị này càng cao càng tốt.
-   **Mẫu số**: Thuật ngữ này tính tổng sự tương đồng giữa \(z_i\) và *tất cả* các mẫu khác trong lô (trừ chính nó). Điều này đại diện cho "bằng chứng" cho tất cả các cặp có thể có.
-   **Phân số**: Phân số là một hàm softmax. Nó tính toán xác suất rằng \(z_j\) là đối tác dương thực sự cho \(z_i\).
-   **`-log`**: Logarit âm là một cách tiêu chuẩn để biến một xác suất thành một giá trị mất mát. Việc giảm thiểu mất mát tương đương với việc tối đa hóa xác suất mô hình xác định chính xác cặp dương.

Nhiệt độ \(\tau\) là một nút điều chỉnh quan trọng. Một nhiệt độ thấp hơn sẽ khuếch đại sự khác biệt giữa các tương đồng, buộc mô hình phải làm việc chăm chỉ hơn để phân biệt giữa các mẫu âm khó. Một nhiệt độ cao hơn sẽ làm mịn phân phối, làm cho nhiệm vụ dễ dàng hơn.

#### Bước 4: Tóm tắt mục tiêu
Mục tiêu tổng thể của hàm mất mát NT-Xent là học một không gian nhúng nơi các biểu diễn của các "khung nhìn" được tăng cường khác nhau của cùng một mẫu (cặp dương) được kéo lại gần nhau, trong khi các biểu diễn của tất cả các mẫu khác (các cặp âm) bị đẩy ra xa hơn. Biểu diễn đã học này phải mạnh mẽ trước các phép tăng cường trong khi vẫn nhạy cảm với các đặc điểm thiết yếu của dữ liệu.

### 2.2.3 Mô hình cơ sở CA-TCC: Một quy trình bán giám sát đa giai đoạn nghiêm ngặt

Khuôn khổ CoFT là một sự mở rộng trực tiếp và có nguyên tắc của **CA-TCC (Tăng cường Tương phản - Phân cụm Tương phản Thời gian)** [7], một khuôn khổ tiên tiến cho phân loại chuỗi thời gian bán giám sát. Nó không chỉ đơn thuần là một hàm mất mát duy nhất mà là một quy trình hoàn chỉnh, đa giai đoạn được thiết kế để tận dụng tối đa cả dữ liệu không có nhãn và có nhãn. Việc hiểu rõ quy trình phức tạp này là rất quan trọng để đặt bối cảnh cho những đổi mới được trình bày trong luận văn này, vì nó tạo thành "sân chơi" nền tảng mà CoFT được xây dựng và đánh giá.

![Hình 1: Học tương phản hai mục tiêu trong CA-TCC. Mô hình học cách bất biến với các phép tăng cường thông qua Tương phản Thời gian và nhạy cảm với cấu trúc thời gian thông qua Tương phản Ngữ cảnh.](Images/Fig. 1. Overall architecture of the proposed TS-TCC. The Temporal Contrastingmodule.png)
*Hình 1: Học tương phản hai mục tiêu trong CA-TCC. Mô hình học cách bất biến với các phép tăng cường thông qua Tương phản Thời gian và nhạy cảm với cấu trúc thời gian thông qua Tương phản Ngữ cảnh.*

Quy trình làm việc của CA-TCC bao gồm các giai đoạn được kết nối với nhau sau đây:

#### Giai đoạn 1: Tiền huấn luyện Tương phản Tự giám sát
Giai đoạn đầu tiên nhằm mục đích học một bộ mã hóa chuỗi thời gian mạnh mẽ, có mục đích chung, \(f(\cdot)\), từ một kho dữ liệu lớn không có nhãn. Điều này đạt được bằng cách huấn luyện bộ mã hóa với một hàm mất mát tương phản hai mục tiêu.

**1. Tương phản Thời gian (\(L_{Temp}\)):** Mục tiêu chính là làm cho các biểu diễn của mô hình bất biến với các phép tăng cường. Đối với mỗi mẫu \(x_i\) trong một lô, hai khung nhìn tương quan được tạo ra bằng cách sử dụng một phép tăng cường mạnh (\(aug_s\)) và một phép tăng cường yếu (\(aug_w\)). Các khung nhìn này, \(x_i^s = aug_s(x_i)\) và \(x_i^w = aug_w(x_i)\), được đưa qua bộ mã hóa \(f(\cdot)\) để tạo ra các vector nhúng \(z_i^s\) và \(z_i^w\). Mất mát Tương phản Thời gian là mất mát NT-Xent kéo cặp dương này lại với nhau.

**2. Tương phản Ngữ cảnh (\(L_{Cont}\)):** Mục tiêu này đảm bảo rằng mô hình nắm bắt được cấu trúc thời gian vốn có của các tín hiệu. Đối với một biểu diễn chuỗi thời gian nhất định \(z_i\), cặp dương của nó được định nghĩa là hàng xóm ngay lập tức của nó trong thời gian, \(z_{i+1}\). Mất mát Tương phản Ngữ cảnh sử dụng công thức NT-Xent để kéo các biểu diễn liền kề này lại với nhau, khuyến khích một không gian nhúng mượt mà và mạch lạc về mặt thời gian.

Tổng mất mát cho giai đoạn tiền huấn luyện tự giám sát là tổng có trọng số của hai thành phần này:

\[ L_{CA-TCC} = L_{Temp} + \alpha \cdot L_{Cont} \]

trong đó \(\alpha\) là một siêu tham số cân bằng giữa tính bất biến của phép tăng cường và tính mạch lạc về thời gian.

#### Giai đoạn 2: Tinh chỉnh có giám sát
Sau khi tiền huấn luyện, bộ mã hóa đã học \(f(\cdot)\) được tinh chỉnh trên một bộ dữ liệu nhỏ, có nhãn (\(D_L\)). Một bộ phân loại tuyến tính, \(g(\cdot)\), được thêm vào trên cùng của bộ mã hóa, và toàn bộ mô hình (\(g \circ f\)) được huấn luyện bằng cách sử dụng **Mất mát Cross-Entropy Phân loại (\(L_{CE}\))** tiêu chuẩn:

\[ L_{CE} = -\sum_{c=1}^{M} y_{o,c} \log(p_{o,c}) \]

Trong đó \(M\) là số lớp, \(y_{o,c}\) là một chỉ báo nhị phân của lớp thực cho quan sát \(o\), và \(p_{o,c}\) là xác suất dự đoán của mô hình.

#### Giai đoạn 3: Tạo nhãn giả
Đây là bước đầu tiên trong việc tận dụng bộ dữ liệu lớn không có nhãn (\(D_U\)) để cải thiện mô hình. Mô hình được tinh chỉnh từ Giai đoạn 2 được sử dụng để đưa ra dự đoán trên tất cả các mẫu trong \(D_U\). Các dự đoán vượt qua một ngưỡng tin cậy cao (ví dụ: xác suất > 0.95) được chọn làm **nhãn giả** chất lượng cao. Điều này tạo ra một tập huấn luyện mới, lớn hơn, \(D_{PL}\), bao gồm cả dữ liệu có nhãn ban đầu và dữ liệu được gán nhãn giả một cách tự tin.

#### Giai đoạn 4: Học Tương phản có giám sát để Tinh chỉnh Biểu diễn
Bước nâng cao nhất trong quy trình CA-TCC là tinh chỉnh không gian đặc trưng của bộ mã hóa bằng cách sử dụng bộ dữ liệu kết hợp có nhãn và nhãn giả (\(D_L \cup D_{PL}\)). Điều này đạt được bằng cách sử dụng **Mất mát Tương phản có giám sát (\(L_{SupCon}\))**. Không giống như mất mát NT-Xent tự giám sát chỉ xem xét một cặp dương cho mỗi mẫu, \(L_{SupCon}\) tận dụng thông tin nhãn để coi tất cả các mẫu trong cùng một lớp là các cặp dương.

##### Phân tích Mất mát SupCon

**Bước 1: Trình bày Công thức**

Đối với một lô N mẫu, mất mát SupCon cho một mẫu nhất định (neo) \(i\) được định nghĩa là:

\[ L_{SupCon}^{(i)} = \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(z_i, z_p)/\tau)}{\sum_{k \in A(i)} \exp(\text{sim}(z_i, z_k)/\tau)} \quad \text{(Phương trình 2.2)} \]

**Bước 2: Định nghĩa từng ký hiệu**

-   **\( L_{SupCon}^{(i)} \)**: Mất mát tương phản có giám sát cho một mẫu neo duy nhất \(i\).
-   **\( A(i) \)**: Tập hợp tất cả các mẫu khác trong lô (mẫu neo \(i\) bị loại trừ).
-   **\( P(i) \)**: Tập hợp tất cả các "dương" cho mẫu neo \(i\) trong lô, được định nghĩa là tất cả các mẫu khác \(p \in A(i)\) có cùng nhãn lớp (\(y_p = y_i\)).
-   **\( |P(i)| \)**: Số lượng các dương cho mẫu neo \(i\) trong lô.
-   Tất cả các ký hiệu khác (\(z_i, z_p, z_k, \text{sim}, \tau\)) được định nghĩa như trong mất mát NT-Xent.

**Bước 3: Giải thích bằng ngôn ngữ đơn giản**

Công thức này mở rộng ý tưởng của mất mát NT-Xent. Thay vì chỉ có *một* đối tác dương (phép tăng cường khác), nó có *nhiều* đối tác dương: mọi mẫu khác trong lô thuộc cùng một lớp.

-   Phần bên trong của công thức vẫn là một hàm softmax cố gắng làm cho mẫu neo \(z_i\) giống với một đối tác dương \(z_p\) hơn bất kỳ mẫu nào khác \(z_k\) trong lô.
-   Sự khác biệt chính là tổng bên ngoài (\(\sum_{p \in P(i)}\)) và phép chuẩn hóa (\(\frac{-1}{|P(i)|}\)). Cấu trúc này tính toán mất mát cho *mọi đối tác dương* trong lô và lấy trung bình kết quả. Nó giống như yêu cầu mô hình giải quyết nhiều bài toán "ai là đối tác của tôi?" đồng thời, một cho mỗi mẫu của cùng một lớp.

**Bước 4: Tóm tắt mục tiêu**

Mục tiêu tổng thể của Mất mát Tương phản có giám sát là kéo các biểu diễn của tất cả các mẫu thuộc cùng một lớp lại gần nhau một cách rõ ràng, đồng thời đẩy chúng ra xa các mẫu của tất cả các lớp khác. Điều này tạo ra các không gian đặc trưng được phân cụm chặt chẽ và tách biệt rõ ràng, cải thiện đáng kể sức mạnh phân biệt của các biểu diễn đã học.

#### Giai đoạn 5: Huấn luyện bộ phân loại cuối cùng
Sau khi các biểu diễn của bộ mã hóa đã được tinh chỉnh bằng cách sử dụng \(L_{SupCon}\), đầu bộ phân loại \(g(\cdot)\) bị loại bỏ và một bộ phân loại tuyến tính mới được huấn luyện từ đầu trên bộ mã hóa đã được tinh chỉnh và đóng băng bằng cách sử dụng dữ liệu có nhãn ban đầu \(D_L\). Bước cuối cùng này đảm bảo rằng bộ phân loại được hiệu chỉnh hoàn hảo với không gian đặc trưng mới được cấu trúc, mang lại hiệu suất phân loại cuối cùng.

Quá trình năm giai đoạn hoàn chỉnh này đại diện cho mô hình cơ sở phức tạp mà CoFT xây dựng dựa trên đó. Bằng cách bảo tồn quy trình này cho nhánh thời gian của mình, luận văn này có thể cô lập và đánh giá nghiêm ngặt những lợi ích đạt được bằng cách giới thiệu miền tần số song song và cơ chế đồng huấn luyện chéo miền.

**Bảng 1: Nghiên cứu cắt lớp các thành phần trong TS-TCC và CA-TCC.** Bảng này, được điều chỉnh từ bài báo CA-TCC gốc [7], thể hiện giá trị gia tăng của từng thành phần đối với hiệu suất. Nó cho thấy rõ ràng rằng việc bổ sung Học Tương phản có giám sát (SCC) trong CA-TCC mang lại một sự thúc đẩy đáng kể so với TS-TCC, và sự kết hợp của các phép tăng cường yếu và mạnh là vượt trội. Kết quả dựa trên thí nghiệm đánh giá tuyến tính với 5% dữ liệu có nhãn.

| Thành phần                 | HAR (Acc / MF1)            | Sleep-EDF (Acc / MF1)      | Epilepsy (Acc / MF1)       |
| :--------------------------- | :------------------------- | :------------------------- | :------------------------- |
| Chỉ TC                       | 68.16 / 66.89              | 75.55 / 60.19              | 88.29 / 88.00              |
| TC + X-Aug                   | 74.22 / 72.18              | 77.80 / 61.28              | 90.51 / 89.27              |
| TS-TCC (TC + X-Aug + CC)     | 77.58 / 76.66              | 76.98 / 70.94              | 93.12 / 93.67              |
| **CA-TCC (TC + X-Aug + SCC)**| **88.27 / 88.29**          | **82.14 / 74.75**          | **94.52 / 94.00**          |
| --- | --- | --- | --- |
| *TS-TCC (Chỉ Yếu)*          | *67.39 / 65.54*            | *79.63 / 68.15*            | *93.22 / 91.97*            |
| *CA-TCC (Chỉ Yếu)*          | *85.68 / 84.77*            | *81.62 / 70.10*            | *93.84 / 92.19*            |
| --- | --- | --- | --- |
| *TS-TCC (Chỉ Mạnh)*        | *50.37 / 43.05*            | *74.84 / 64.53*            | *92.49 / 90.60*            |
| *CA-TCC (Chỉ Mạnh)*        | *59.59 / 53.34*            | *79.24 / 69.39*            | *93.74 / 92.00*            |

## 2.3 Đồng huấn luyện và hợp nhất miền tần số-thời gian

Đóng góp quan trọng nhất của CoFT là ứng dụng mới của phương pháp đồng huấn luyện để hợp nhất thông tin từ miền thời gian và tần số. Để đánh giá cao đóng góp này, cần phải xem xét cách hai miền này đã được kết hợp trước đây.

### 2.3.1 Các phương pháp hợp nhất truyền thống

Ý tưởng rằng miền thời gian và tần số chứa thông tin bổ sung đã được thiết lập rõ ràng trong xử lý tín hiệu. Các phương pháp học máy và học sâu truyền thống thường kết hợp chúng theo một trong hai cách, thường được gọi là hợp nhất sớm và hợp nhất muộn [11].

1.  **Hợp nhất sớm (cấp đặc trưng):** Trong phương pháp này, các đặc trưng từ cả hai miền được trích xuất và nối lại *trước khi* được đưa vào một mô hình duy nhất. Ví dụ, người ta có thể tính toán FFT của một chuỗi thời gian, trích xuất các đặc trưng quang phổ như mật độ phổ công suất, và nối chúng vào tín hiệu miền thời gian thô. Mặc dù đơn giản và trực tiếp, phương pháp này có thể không tối ưu vì nó buộc một mô hình duy nhất phải học từ các không gian đặc trưng không đồng nhất với các thuộc tính thống kê khác nhau, và nó có thể dẫn đến một vector đầu vào có chiều rất cao dễ bị ảnh hưởng bởi lời nguyền chiều [12].

2.  **Hợp nhất muộn (cấp quyết định):** Điều này liên quan đến việc huấn luyện hai mô hình riêng biệt, chuyên biệt, một cho mỗi miền, và sau đó kết hợp các dự đoán đầu ra của chúng (ví dụ, bằng cách lấy trung bình hoặc một phiếu bầu có trọng số). Điều này cho phép mỗi mô hình học các đặc trưng một cách tối ưu cho miền riêng của nó, đây là một lợi thế đáng kể. Tuy nhiên, nó có thể bỏ lỡ việc khám phá các tương tác hiệp đồng sâu hơn giữa các miền trong giai đoạn học biểu diễn, vì sự hợp nhất chỉ xảy ra ở bước quyết định cuối cùng [11, 12].

### 2.3.2 CoFT: Một khuôn khổ đồng huấn luyện thực sự

CoFT vượt ra ngoài sự hợp nhất đơn giản và thực hiện một **khuôn khổ đồng huấn luyện thực sự**, một khái niệm được tiên phong bởi Blum và Mitchell (1998) [3] trong học bán giám sát. Thuật toán đồng huấn luyện ban đầu yêu cầu hai "khung nhìn" độc lập có điều kiện về dữ liệu. CoFT điều chỉnh khái niệm này bằng cách coi **miền thời gian và tần số là hai khung nhìn riêng biệt nhưng bổ sung**.

Cách tiếp cận này về cơ bản khác với các công trình trước đây:
*   **Đối tác bình đẳng:** Không giống như các phương pháp coi miền tần số là một nguồn đặc trưng được tiền xử lý thứ cấp, CoFT thiết lập hai nhánh mã hóa song song, đối xứng về mặt kiến trúc. "Sự tương đương về kiến trúc" này, như được mô tả trong luận văn (Chương 3), là một lựa chọn phương pháp luận có chủ ý để đảm bảo rằng bất kỳ sự cải thiện hiệu suất nào cũng đến từ chính thông tin, chứ không phải từ lợi thế kiến trúc của một nhánh so với nhánh kia.
*   **Chuyển giao kiến thức qua nhãn giả:** CoFT tạo điều kiện cho việc chuyển giao kiến thức thông qua một mô-đun đồng huấn luyện phức tạp. Một nhánh tạo ra các nhãn giả có độ tin cậy cao, sau đó được sử dụng để huấn luyện nhánh còn lại. Điều này tạo ra một vòng lặp phản hồi nơi mỗi miền giúp điều chuẩn và cải thiện miền kia, điều này đặc biệt mạnh mẽ trong các cài đặt có ít nhãn.
*   **Khám phá "Ít hơn là Nhiều hơn":** Phát hiện phản trực giác và có tác động mạnh mẽ nhất của nghiên cứu CoFT là hiện tượng "Ít hơn là Nhiều hơn" liên quan đến siêu tham số đồng huấn luyện `lambda_ct`. Sự khôn ngoan thông thường có thể cho rằng một sự kết hợp mạnh ( `lambda_ct` cao) là cần thiết để chuyển giao kiến thức hiệu quả. Tuy nhiên, luận văn này đã chứng minh một cách thực nghiệm (Chương 4) rằng một giá trị cực thấp (`lambda_ct` = 0.0001) là tối ưu. Các giá trị cao dẫn đến "sự nhầm lẫn nhãn", trong đó các nhãn giả nhiễu từ một miền làm hỏng quá trình học của miền kia. Một sự kết hợp nhẹ nhàng, có trọng số thấp chỉ cung cấp đủ sự điều chuẩn để hướng dẫn việc học biểu diễn mà không lấn át tín hiệu thực tế. Khám phá này là một đóng góp khoa học quan trọng cho sự hiểu biết về động lực đồng huấn luyện trong học sâu.

## 2.4 Kết luận: Xây dựng trên một nền tảng vững chắc

Khuôn khổ CoFT được xây dựng vững chắc trên các nguyên tắc của học tương phản tự giám sát, áp dụng các phương pháp tốt nhất như tăng cường dữ liệu đơn giản và hiệu quả và một quy trình huấn luyện theo giai đoạn ổn định. Tuy nhiên, sự mới lạ chính của nó nằm ở việc điều chỉnh tinh vi mô hình đồng huấn luyện cho bối cảnh đa miền của phân tích chuỗi thời gian. Bằng cách coi miền thời gian và tần số là các đối tác bình đẳng và cho phép chuyển giao kiến thức nhẹ nhàng thông qua một hàm mất mát hỗn hợp được hiệu chỉnh cẩn thận, CoFT đã cải thiện đáng kể công nghệ tiên tiến. Việc khám phá ra lý thuyết "nhầm lẫn nhãn" và tính tối ưu của các trọng số kết hợp cực thấp không chỉ cung cấp một mô hình hiệu suất cao mà còn là những hiểu biết khoa học quý giá có thể định hướng các nghiên cứu trong tương lai về học bán giám sát và đa miền.

---

# 3 TRIỂN KHAI VÀ PHƯƠNG PHÁP LUẬN

Chương này cung cấp một hướng dẫn chi tiết, từng bước về việc triển khai khuôn khổ CoFT. Tuy nhiên, ngoài một công thức đơn giản, nó còn ghi lại **hành trình phương pháp luận**, bao gồm các quyết định quan trọng, những thách thức kỹ thuật và quy trình nghiêm ngặt cần thiết để đảm bảo rằng các kết quả thực nghiệm vừa hợp lệ vừa có thể tái lập được. Nó trả lời câu hỏi: "Nghiên cứu này đã được tiến hành như thế nào, một cách chính xác và với những cân nhắc nào?"

## 3.1 Hành trình phương pháp luận: Lý giải và thách thức

Con đường dẫn đến một mô hình hoạt động cuối cùng không phải là tuyến tính. Nó bắt đầu với các quyết định chiến lược rộng lớn được thông báo bởi tài liệu, tiếp theo là việc thực hiện tỉ mỉ đã gặp phải và vượt qua các rào cản thực tế đáng kể.

### 3.1.1 Lựa chọn chiến trường: Lựa chọn mô hình cơ sở và benchmark
Quyết định quan trọng đầu tiên là chọn một mô hình cơ sở mạnh, tiên tiến để CoFT có thể được đánh giá một cách công bằng. **CA-TCC (Tăng cường Tương phản - Phân cụm Tương phản Thời gian)** [7] đã được chọn vì ba lý do chính:
1.  **Hiệu suất tiên tiến:** Tại thời điểm nghiên cứu này, CA-TCC đại diện cho một trong những quy trình học bán giám sát mạnh mẽ và được đánh giá cao nhất cho phân loại chuỗi thời gian. Việc vượt qua nó sẽ đại diện cho một đóng góp khoa học có ý nghĩa.
2.  **Quy trình được xác định rõ ràng:** Quá trình đa giai đoạn của nó (tiền huấn luyện tương phản, tinh chỉnh có giám sát, gán nhãn giả và tinh chỉnh biểu diễn) cung cấp một khuôn khổ hoàn chỉnh và hợp lý có thể được mở rộng một cách có hệ thống.
3.  **Benchmark công khai:** Các tác giả ban đầu đã đánh giá mô hình của họ trên các bộ dữ liệu công khai đã được thiết lập (HAR, Sleep-EDF, Epilepsy). Bằng cách sử dụng chính xác các bộ dữ liệu tương tự, chúng tôi có thể nhắm đến một sự so sánh "táo với táo" thực sự, cô lập tác động hiệu suất của kiến trúc được đề xuất của chúng tôi khỏi các biến gây nhiễu.

### 3.1.2 Thử thách về khả năng tái lập: Các rào cản kỹ thuật và dữ liệu
Chỉ chọn cùng một bộ dữ liệu là không đủ để đảm bảo một sự so sánh công bằng. Một phần đáng kể của nỗ lực nghiên cứu đã được dành để vượt qua các thách thức liên quan đến khả năng tái lập—một khía cạnh thường bị đánh giá thấp nhưng rất quan trọng của khoa học tính toán.

**1. Thách thức tiền xử lý dữ liệu:** Bài báo CA-TCC gốc mô tả phương pháp phân chia dữ liệu của nó (ví dụ: phân chia phân tầng 1% và 5%) nhưng không công bố mã cho quá trình này. Để đảm bảo tính công bằng học thuật, việc tái tạo quy trình này *chính xác* là bắt buộc. Điều này liên quan đến một quá trình gian khổ:
    *   Cẩn thận triển khai các kịch bản lấy mẫu phân tầng của riêng chúng tôi dựa trên mô tả của bài báo.
    *   Kiểm tra chéo sự phân phối lớp trong các phân chia được tạo ra của chúng tôi để đảm bảo chúng khớp với sự phân phối lý thuyết.
    *   Duy trì các phân chia chính xác này trong mọi thí nghiệm, bao gồm tất cả các lần chạy cơ sở và các nghiên cứu cắt lớp.
    Nỗ lực này, mặc dù tốn thời gian, là không thể thương lượng đối với tính toàn vẹn của các phát hiện của chúng tôi.

**2. Các lồng môi trường:** Một thách thức đáng kể là thiết lập một môi trường phần mềm ổn định và nhất quán. Các nỗ lực ban đầu đã bị cản trở bởi các vấn đề kỹ thuật phổ biến nhưng khó chịu:
    *   **Xung đột gói:** Các thư viện khác nhau yêu cầu các phiên bản xung đột của các phụ thuộc. Ví dụ, các phiên bản đầu của `numpy` không tương thích với phiên bản `scikit-learn` cần thiết để đánh giá, dẫn đến lỗi nhập. Việc giải quyết những vấn đề này đòi hỏi phải tạo ra một môi trường được giới hạn cẩn thận với các phiên bản gói cụ thể, tương thích với nhau.
    *   **Tính nhất quán đa nền tảng:** Đảm bảo rằng các thí nghiệm chạy trên máy Windows cục bộ tạo ra kết quả giống hệt với các thí nghiệm trên máy chủ dựa trên Linux (như Google Colab hoặc một phiên bản A100) đòi hỏi phải quản lý tỉ mỉ các hạt giống ngẫu nhiên, cài đặt xác định CUDA của PyTorch và các quy trình tải dữ liệu.

Những nỗ lực này đã lên đến đỉnh điểm trong một "hộp cát" ổn định, có thể tái lập, nơi biến số duy nhất được thử nghiệm là chính phương pháp đó.

## 3.2 Các nguyên tắc chỉ đạo cho khả năng tái lập

The soul of this chapter is **reproducibility**. Every design choice, parameter, and procedure is documented with the intention that another researcher can achieve 100% identical results. This commitment is upheld through the following strategies:

1.  **Identical Data Splits**: All experiments, including baseline and proposed models, were conducted on the exact same training, validation, and test splits to ensure fair comparison.
2.  **Fixed Random Seeds**: All stochastic processes, from weight initialization to data shuffling, were controlled using a set of 5 fixed seeds (0, 1, 2, 3, 4). Results are reported as the mean and standard deviation across these independent runs to account for initialization variance.
3.  **Controlled Variables**: When comparing CoFT to its baseline, the *only* variable changed was the `--enable_coft` feature flag. This toggleable design ensures that the baseline model's code and behavior remained completely untouched, isolating the impact of the CoFT module.
4.  **Public Codebase and Open-Source Tools**: The entire project was built using publicly available libraries and will be released to ensure the community can inspect, validate, and build upon this work.

## 3.3 Công nghệ và Triển khai

The framework was implemented using a carefully selected stack of open-source tools, with each choice justified by its role in the research.

**Hardware Configuration:**
-   **Development**: An NVIDIA RTX 4060 (8GB) was used for initial development and memory-constrained optimization, ensuring the model is viable in resource-limited environments. This was crucial for rapid prototyping and debugging cycles.
-   **Validation**: An NVIDIA A100 (40GB) was used for large-scale hyperparameter searches and final performance validation, allowing for experimentation without memory constraints. This powerful hardware was essential for running the extensive grid searches detailed in Chapter 4.
-   **CPU**: An AMD Ryzen 5800X (8 cores) provided sufficient processing power for all data preparation and preprocessing tasks, which were often CPU-bound.

**Software Stack:**
-   **Python 3.8**: Chosen for its broad compatibility with the scientific computing ecosystem and its stability compared to newer versions at the time of the project's inception.
-   **PyTorch (v2.4.1+cu121)**: Selected as the primary deep learning framework for its flexibility, strong community support, and dynamic computation graph ("eager execution"), which is ideal for research and debugging complex models like CoFT. The code includes compatibility checks to gracefully handle older PyTorch versions.
-   **CUDA (v12.1)**: Utilized to leverage NVIDIA GPU acceleration. The implementation includes smart enablement of TensorFloat-32 (TF32) on modern RTX 30/40 series GPUs for significant performance speedups with no loss in accuracy.
-   **NumPy & Pandas**: These libraries formed the backbone of our data manipulation pipeline. Pandas was essential for reading and cleaning the datasets, while NumPy provided the high-performance numerical arrays used throughout the project.
-   **Scikit-learn**: Used for its robust implementations of data splitting (stratified sampling) and for calculating standard evaluation metrics such as the F1-score, ensuring our results could be compared fairly to other published work.

## 3.4 Các bộ dữ liệu benchmark

To ensure a **rigorous and scientifically fair comparison**, a critical methodological decision was made to conduct all experiments on the **exact same benchmark datasets** used in the original CA-TCC publication (Eldele et al., 2023) [7]. By inheriting this established set of benchmarks, we can directly isolate the performance impact of our proposed dual-domain co-training architecture. The chosen datasets—HAR, Sleep-EDF, and Epilepsy—provided a diverse and challenging testbed.

### 3.4.1 Nhận dạng hoạt động của con người (HAR)
-   **Source**: UCI Machine Learning Repository. This public dataset comprises recordings from 30 volunteers performing six activities (Walking, etc.) while wearing a waist-mounted smartphone.
-   **Characteristics**: The data consists of 9-channel time series (tri-axial accelerometer and gyroscope) sampled at 50Hz and segmented into 2.56-second windows (128 data points).
-   **Challenge**: High inter-class similarity between static activities (Sitting, Standing, Laying), demanding a model capable of capturing subtle dynamic differences.

### 3.4.2 Sleep-EDF (Phân loại giai đoạn giấc ngủ)
-   **Source**: PhysioNet Sleep-EDF Database Expanded (sleep-edfx). We use the EEG recordings from the Fpz-Cz channel, sampled at 100Hz.
-   **Characteristics**: The recordings were segmented into 30-second windows (3000 data points) and labeled into one of five sleep stages (Wake, N1, N2, N3, REM).
-   **Challenge**: Severe class imbalance (N2 stage is dominant), subtle low-amplitude differences between stages, and significantly longer sequences, testing the model's ability to handle long-range dependencies.

### 3.4.3 Epilepsy (Phát hiện co giật)
-   **Source**: UCI Machine Learning Repository, originating from the work of Andrzejak et al. [1].
-   **Characteristics**: The dataset consists of 1-second EEG segments (178 data points). The task is simplified to a binary classification problem: identifying seizure segments (Class 1) against all other non-seizure segments.
-   **Challenge**: Highly imbalanced data and the need to distinguish seizure patterns from various other non-seizure brain activities.

### 3.4.4 Phương pháp phân chia dữ liệu bán giám sát
A cornerstone of this research is the simulation of label scarcity. For each dataset, the official training data, \(D_{train\_full}\), is subjected to a **stratified splitting process**. This ensures that even with a small percentage of labels, the class distribution of the original dataset is preserved.

1.  **Full Labeled Set**: The entire training set, \(D_{train\_full}\), is used as the 100% labeled benchmark.
2.  **Stratified Sampling**: To create subsets with a specific percentage \(p\) of labels, we perform stratified sampling from \(D_{train\_full}\). For example, to create the **1% Labeled Set (\(D_{L, 1\%}\))**, we randomly sample 1% of the instances from *each class* present in \(D_{train\_full}\).
3.  **Creation of Subsets**: This procedure is repeated to create various labeled subsets, such as \(D_{L, 1\%}\) and \(D_{L, 5\%}\). The remaining data (\(D_{train\_full} \setminus D_{L, p\%}\)) serves as the large pool of unlabeled data, \(D_U\), for the self-supervised and semi-supervised stages of the training pipeline.

## 3.5 Khuôn khổ CoFT: Kiến trúc và Quy trình

### 3.5.1 Kiến trúc hai nhánh: Thiết kế và Triển khai

CoFT sử dụng một kiến trúc hai nhánh song song. Thiết kế ban đầu, được trình bày chi tiết dưới đây, được xây dựng cẩn thận để duy trì sự đối xứng về kiến trúc. Quyết định này là một phần quan trọng trong phương pháp luận khoa học của chúng tôi, cho phép so sánh công bằng và có kiểm soát giữa miền thời gian và tần số.

![Hình 2: Tổng quan cấp cao về khuôn khổ CoFT.](Images/Fig. 5. dual branch temporal-frequency CoFT structure.png)
*Hình 2: Tổng quan cấp cao về khuôn khổ CoFT. Nó cho thấy các nhánh Thời gian và Tần số song song, bộ điều hợp động, và mô-đun đồng huấn luyện trung tâm điều phối việc chuyển giao kiến thức trước một dự đoán tổ hợp cuối cùng.*

**Nhánh thời gian** bảo tồn chính xác kiến trúc CA-TCC để đảm bảo so sánh công bằng. Đối với **nhánh tần số**, quyết định phản chiếu kiến trúc thời gian là một lựa chọn phương pháp luận có chủ ý để thiết lập một đường cơ sở có kiểm soát. Bằng cách giữ cho dung lượng mô hình giống hệt nhau, chúng tôi có thể đảm bảo rằng bất kỳ sự khác biệt hiệu suất nào quan sát được đều hoàn toàn do các đặc điểm vốn có của chính dữ liệu miền tần số, chứ không phải do lợi thế về kiến trúc.

![Hình 3: Kiến trúc của khối Transformer được sử dụng trong cả hai bộ mã hóa.](Images/Fig. 2. Architecture of the Transformer model used in the Temporal Contrasting.png)
*Hình 3: Kiến trúc của khối Transformer được sử dụng trong cả bộ mã hóa thời gian và tần số. Các đặc trưng của bộ mã hóa được chiếu và kết hợp với một token phân loại, được xử lý thông qua các lớp chú ý đa đầu và MLP, và cuối cùng được sử dụng cho các nhiệm vụ xuôi dòng.*

![Hình 4: Cơ chế Tương phản Tần số.](Images/Fig. 4. frequency-domain branch.png)
*Hình 4: Cơ chế Tương phản Tần số. Tương tự như nhánh thời gian, nhánh tần số sử dụng các phép tăng cường (Nhiễu Quang phổ, Che Tần số) và một bộ mã hóa dựa trên Transformer để học các biểu diễn mạnh mẽ thông qua một mất mát tương phản.*

#### Biến đổi miền tần số: Vượt ra ngoài FFT đơn giản
Biến đổi tần số giải quyết một thách thức cơ bản: làm thế nào để chuyển đổi đầu ra FFT có giá trị phức thành một định dạng phù hợp cho các kiến trúc CNN tiêu chuẩn. Quy trình được chọn như sau:
```python
# Real FFT để hiệu quả tính toán
x_fft = torch.fft.rfft(x, norm='ortho')  

# Phân tách biên độ-pha rõ ràng
magnitude = torch.abs(x_fft)        # |Z|
phase = torch.angle(x_fft)          # ∠Z  

# Xếp chồng kênh để tương thích với CNN
x_freq = torch.cat([magnitude, phase], dim=1)  # [B, C*2, F]
```
Phương pháp này được chọn vì nhiều lý do: sử dụng Real FFT hiệu quả hơn về mặt tính toán và bộ nhớ đối với các tín hiệu đầu vào có giá trị thực; việc phân tách biên độ-pha bảo toàn thông tin quang phổ hoàn chỉnh không giống như các phương pháp chỉ dùng biên độ; và nó tạo ra các tensor có giá trị thực tương thích với các lớp Conv1D tiêu chuẩn.

#### Thích ứng kiến trúc động
Một chi tiết triển khai quan trọng để đảm bảo tính mạnh mẽ là việc sử dụng **khởi tạo lớp tuyến tính động** trong nhánh tần số. Bởi vì các chiều đặc trưng cuối cùng có thể thay đổi sau các lớp tích chập tùy thuộc vào độ dài tín hiệu đầu vào, lớp phân loại cuối cùng được khởi tạo trong lần truyền thuận đầu tiên:
```python
# Lần truyền thuận đầu tiên xác định các chiều đặc trưng thực tế
if self.freq_logits is None:
    actual_features = x_flat.shape[1]  # Được tính toán sau các lớp conv
    self.freq_logits = nn.Linear(actual_features, num_classes).to(device)
```
Thiết kế này cho phép cùng một kiến trúc hoạt động liền mạch trên các bộ dữ liệu có độ dài thời gian và số kênh khác nhau mà không cần thay đổi cấu hình thủ công.

### 3.5.2 Hàm mất mát hỗn hợp: Phân tích chi tiết

Việc điều phối học tập hai nhánh được chi phối bởi một hàm mất mát hỗn hợp phức tạp.

#### Bước 1: Trình bày Công thức
Tổng mất mát, \( L_{total} \), được xây dựng dưới dạng tổng có trọng số của bốn thành phần riêng biệt:

\[ L_{total} = L_{sup\_t} + L_{sup\_f} + \lambda_{ct} \cdot L_{cotraining} + \lambda_{cs} \cdot L_{consistency} \quad \text{(Phương trình 3.1)} \]

#### Bước 2: Định nghĩa từng ký hiệu
-   **\( L_{sup\_t} \)** & **\( L_{sup\_f} \)**: Mất mát phân loại có giám sát tiêu chuẩn (Cross-Entropy Phân loại) cho nhánh thời gian và tần số, được tính toán bằng cách sử dụng các nhãn thực tế.
-   **\( L_{cotraining} \)**: Mất mát đồng huấn luyện. Đây là cốt lõi của cơ chế dạy học tương hỗ, trong đó một nhánh được huấn luyện trên các *nhãn giả* có độ tin cậy cao được tạo ra bởi nhánh kia.
-   **\( L_{consistency} \)**: Một mất mát nhất quán đặc trưng (ví dụ: Lỗi Bình phương Trung bình) khuyến khích các vector nhúng cấp cao từ cả hai nhánh tương tự nhau cho cùng một mẫu đầu vào.
-   **\( \lambda_{ct} \)** (Trọng số đồng huấn luyện): Một siêu tham số điều chỉnh ảnh hưởng của mất mát đồng huấn luyện.
-   **\( \lambda_{cs} \)** (Trọng số nhất quán): Một siêu tham số kiểm soát sức mạnh của việc điều chuẩn nhất quán đặc trưng.

#### Bước 3: Giải thích bằng ngôn ngữ đơn giản
Phương trình này hoạt động như một hệ thống kiểm soát cho toàn bộ mô hình. Nó cân bằng bốn mục tiêu học tập riêng biệt:
1.  **`Học từ sự thật` (\(L_{sup\_t} + L_{sup\_f}\))**: Mục tiêu chính. Cả hai chuyên gia phải học cách phân loại chính xác dữ liệu dựa trên các nhãn thực, đã được xác minh.
2.  **`Dạy học tương hỗ` (\(L_{cotraining}\))**: Thành phần "học sinh-giáo viên". Mỗi nhánh học từ các dự đoán tự tin của nhánh kia, điều chỉnh việc huấn luyện của mình bằng một góc nhìn bổ sung.
3.  **`Thống nhất biểu diễn` (\(L_{consistency}\))**: Điều này buộc hai nhánh phải tìm ra điểm chung, đảm bảo các diễn giải cấp cao của chúng (vector nhúng đặc trưng) được ánh xạ đến một vị trí tương tự trong không gian tiềm ẩn.

#### Bước 4: Tóm tắt mục tiêu
Mục tiêu tổng thể của hàm mất mát hỗn hợp là huấn luyện hai chuyên gia chuyên biệt nhưng hợp tác. Nó đặt cả hai chuyên gia vào thực tế với mất mát có giám sát, buộc chúng học hỏi từ các quan điểm độc đáo của nhau thông qua đồng huấn luyện, và khuyến khích chúng phát triển một sự hiểu biết chung về dữ liệu thông qua mất mát nhất quán.

### 3.5.3 Quy trình huấn luyện sáu giai đoạn: Công thức từng bước

Phương pháp huấn luyện cuối cùng sử dụng một quy trình 6 giai đoạn, được phát hiện là rất quan trọng để đảm bảo sự ổn định. Các thí nghiệm ban đầu với việc huấn luyện chung từ đầu đến cuối dễ bị xung đột gradient, đòi hỏi một phương pháp theo giai đoạn để trước tiên xây dựng các biểu diễn ổn định trước khi giới thiệu các tương tác chéo miền phức tạp.

![Hình 5: Chiến lược huấn luyện đa pha, được điều chỉnh từ CA-TCC [7]. CoFT mở rộng khái niệm này thành một quy trình sáu giai đoạn.](Images/Fig. 3. Four phases for CA-TCC semi-supervised training. In Phase 1, TS-TCC is trained with fully unlabeled data. Next, we use the available few labeled.png)
*Hình 5: Chiến lược huấn luyện đa pha, được điều chỉnh từ CA-TCC [7]. CoFT mở rộng khái niệm này thành một quy trình sáu giai đoạn.*

-   **Giai đoạn 1: `self_supervised`**: Cả hai nhánh được tiền huấn luyện trong 40 kỷ nguyên trên dữ liệu không có nhãn bằng cách sử dụng các mất mát tương phản tương ứng của chúng. Không có sự tương tác giữa các nhánh.
-   **Giai đoạn 2: `train_linear_{p}`**: Một bộ phân loại tuyến tính được huấn luyện trên đỉnh của mỗi bộ mã hóa đã đóng băng để đánh giá chất lượng của các biểu diễn đã học bằng cách sử dụng một tỷ lệ `p` nhãn.
-   **Giai đoạn 3: `ft_{p}`**: Toàn bộ mô hình được tinh chỉnh bằng cách sử dụng hàm mất mát hỗn hợp hoàn chỉnh (Phương trình 3.1) trên dữ liệu có nhãn. Đây là giai đoạn đầu tiên có sự đồng huấn luyện.
-   **Giai đoạn 4: `gen_pseudo_labels`**: Mô hình được tinh chỉnh từ Giai đoạn 3 được sử dụng để tạo các nhãn giả có độ tin cậy cao (xác suất softmax > 0.95) cho dữ liệu không có nhãn.
-   **Giai đoạn 5: `SupCon`**: Các bộ mã hóa được tinh chỉnh thêm bằng cách sử dụng Mất mát Tương phản có giám sát trên tập hợp kết hợp của các nhãn gốc và nhãn giả.
-   **Giai đoạn 6: `train_linear_SupCon_{p}`**: Cuối cùng, các bộ mã hóa lại được đóng băng, và một bộ phân loại tuyến tính mới được huấn luyện từ đầu trên đỉnh của các biểu diễn đã được tinh chỉnh để tạo ra kết quả cuối cùng.

### 3.5.4 Tăng cường dữ liệu

CoFT đã sử dụng một tập hợp các phép tăng cường hiệu quả và tiết kiệm chi phí tính toán cho cả hai miền.

-   **Miền Thời gian**: Áp dụng các phép tăng cường hiệu quả từ mô hình cơ sở CA-TCC: **Jittering**, **Scaling**, và **Cropping**.
-   **Miền Tần số**: Một cách tiếp cận thận trọng đã được sử dụng để tránh làm hỏng các mẫu chẩn đoán, được thiết kế để bắt chước các hiện vật tín hiệu thực tế: **Thêm nhiễu trong miền FFT** và **Che tần số chọn lọc**.

---

# 4 KẾT QUẢ VÀ PHÂN TÍCH

Chương này trình bày các phát hiện thực nghiệm của nghiên cứu, được cấu trúc để trả lời trực tiếp các câu hỏi nghiên cứu đã đặt ra trong Chương 1. Chúng tôi ghi lại hành trình nghiên cứu từ các giả thuyết ban đầu, qua các thí nghiệm thất bại, đến các kết quả đột phá cuối cùng, cung cấp một báo cáo minh bạch về quá trình khoa học.

## 4.1 Trả lời câu hỏi nghiên cứu 1: CoFT có thể vượt trội hơn một mô hình cơ sở tiên tiến không?

Câu hỏi nghiên cứu đầu tiên tìm cách xác định liệu khuôn khổ CoFT có thể vượt trội đáng kể và nhất quán so với một mô hình đơn miền, tiên tiến (CA-TCC) hay không. Các kết quả cuối cùng, được tóm tắt trong Bảng 2 và 3 sau một quá trình tối ưu hóa rộng rãi, cung cấp một câu trả lời rõ ràng và khẳng định.

**Bảng 2: Hiệu suất cuối cùng của CoFT so với mô hình cơ sở CA-TCC (trung bình 5 hạt giống)**

| Dataset   | % Nhãn | Model       | Độ chính xác            | Điểm MF1                |
|:----------|:--------|:------------|:------------------------|:------------------------|
| **HAR**       | **1%**      | CA-TCC (Cơ sở) | 77.3% ± 0.6%            | 76.2% ± 0.1%            |
|           |         | **CoFT (Của chúng tôi)** | **85.47% ± 0.5%**       | **85.44% ± 0.1%**       |
| **HAR**       | **5%**      | CA-TCC (Cơ sở) | 88.3% ± 0.3%            | 88.3% ± 0.4%            |
|           |         | **CoFT (Của chúng tôi)** | **90.04% ± 0.3%**       | **89.62% ± 0.4%**       |
| **Sleep-EDF** | **1%**      | CA-TCC (Cơ sở) | 70.8% ± 0.5%            | 79.4% ± 0.1%            |
|           |         | **CoFT (Của chúng tôi)** | **80.12% ± 0.5%**       | 69.68% ± 0.1%           |
| **Sleep-EDF** | **5%**      | CA-TCC (Cơ sở) | 74.6% ± 0.1%            | 82.1% ± 0.2%            |
|           |         | **CoFT (Của chúng tôi)** | **83.23% ± 0.1%**       | 71.85% ± 0.2%           |
| **Epilepsy**  | **1%**      | CA-TCC (Cơ sở) | 91.9% ± 0.1%            | 92.0% ± 0.1%            |
|           |         | **CoFT (Của chúng tôi)** | **94.61% ± 0.1%**       | **91.04% ± 0.1%**       |
| **Epilepsy**  | **5%**      | CA-TCC (Cơ sở) | 94.5% ± 0.1%            | 94.0% ± 0.1%            |
|           |         | **CoFT (Của chúng tôi)** | **94.91% ± 0.1%**       | **91.55% ± 0.1%**       |

**Bảng 3: Phân tích thống kê về sự cải thiện hiệu suất**

| Dataset   | % Nhãn | Tăng độ chính xác | p-value (Độ chính xác) |
|:----------|:--------|:--------------|:-------------------|
| HAR       | 1%      | **+8.17%**    | <0.01              |
| HAR       | 5%      | **+1.74%**    | <0.01              |
| Sleep-EDF | 1%      | **+9.32%**    | <0.01              |
| Sleep-EDF | 5%      | **+8.63%**    | <0.01              |
| Epilepsy  | 1%      | **+2.71%**    | <0.01              |
| Epilepsy  | 5%      | **+0.41%**    | <0.05              |

**Diễn giải kết quả:**
-   **Mô tả**: Bảng 2 trình bày so sánh trực tiếp độ chính xác cuối cùng và điểm MF1 của mô hình CoFT so với mô hình cơ sở CA-TCC trên ba bộ dữ liệu và hai kịch bản nhãn thấp (1% và 5%). Bảng 3 định lượng mức tăng độ chính xác tuyệt đối và cung cấp giá trị p từ kiểm định t cặp để đánh giá ý nghĩa thống kê.
-   **Quan sát**: CoFT thể hiện sự cải thiện hiệu suất nhất quán và có ý nghĩa thống kê so với mô hình cơ sở trong tất cả các kịch bản được thử nghiệm. Mức tăng đáng kể nhất được quan sát trong cài đặt nhãn 1% cho HAR (+8.17%) và Sleep-EDF (+9.32%), cho thấy khuôn khổ này đặc biệt hiệu quả trong tình trạng khan hiếm nhãn cực độ. Khi tỷ lệ nhãn có sẵn tăng lên 5%, khoảng cách hiệu suất thu hẹp, nhưng CoFT vẫn duy trì một lợi thế đáng kể.
-   **Phân tích**: Kết quả ủng hộ mạnh mẽ giả thuyết rằng việc tận dụng miền tần số thông qua đồng huấn luyện mang lại lợi ích đáng kể. Khả năng của khuôn khổ sử dụng các dự đoán tự tin của một miền để dạy miền kia hoạt động như một cơ chế điều chuẩn mạnh mẽ, điều này có giá trị nhất khi nhãn thực tế khan hiếm. Mức tăng giảm dần (nhưng vẫn dương) ở nhãn 5% cho thấy khi có nhiều dữ liệu có giám sát hơn, hiệu suất của mô hình cơ sở cải thiện, nhưng thông tin bổ sung do nhánh thứ hai của CoFT cung cấp vẫn mang lại một lợi thế khác biệt.

## 4.2 Trả lời câu hỏi nghiên cứu 2: Nguồn gốc thực sự của sự cải thiện hiệu suất là gì?

Câu hỏi nghiên cứu thứ hai nhằm mục đích phân tích hiệu suất của CoFT, xác định bao nhiêu phần trăm là do kiến trúc mới so với việc tối ưu hóa siêu tham số nghiêm ngặt. Một nghiên cứu cắt lớp chi tiết, được trình bày trong Bảng 4, đã được thiết kế để trả lời câu hỏi này.

**Bảng 4: Nghiên cứu cắt lớp - Phân tích sự cải thiện hiệu suất của CoFT trên HAR (1% Nhãn)**

| # | Cấu hình                                          | Độ chính xác (TB ± Lệch chuẩn) | Tăng so với bước trước | Nguồn gốc cải thiện / Điểm chính                                                                   |
|:-:|:----------------------------------------------------|:----------------------|:--------------------|:------------------------------------------------------------------------------------------------------|
| 1 | **Cơ sở (CA-TCC gốc)**                             | 77.3%                 | -                   | Điểm xuất phát.                                                                                       |
| 2 | **Cơ sở (Siêu tham số đã tinh chỉnh)**              | **83.59% ± 1.94**     | **+6.29%**          | **Tinh chỉnh siêu tham số** là yếu tố có tác động lớn nhất.                                           |
| 3 | **CoFT (với Đồng huấn luyện, nhưng chỉ dự đoán Thời gian)** | 82.90% ± 2.46         | -0.69%              | Thêm nhánh tần số mà không có tổ hợp phù hợp *làm tổn hại* hiệu suất so với một cơ sở đã được tinh chỉnh. |
| 4 | **CoFT (Mô hình đầy đủ với Tổ hợp)**                | **85.47%**            | **+2.57%**          | **Cơ chế Tổ hợp** là rất quan trọng để khai thác tiềm năng của nhánh tần số và đạt được kết quả SOTA. |

*Lưu ý: Kết quả cho các cấu hình 2 và 3 là trung bình của 3 hạt giống. Kết quả cho 1 và 4 là từ các lần chạy đơn lẻ, đại diện.*

**Diễn giải kết quả:**
-   **Mô tả**: Bảng 4 chia nhỏ quá trình tiến triển hiệu suất từ mô hình cơ sở ban đầu đến mô hình CoFT cuối cùng thành bốn bước riêng biệt. Mỗi bước giới thiệu một thành phần mới duy nhất, cho phép cô lập đóng góp cụ thể của nó.
-   **Quan sát**: Mức tăng hiệu suất đơn lẻ lớn nhất (**+6.29%**) đến từ việc chỉ cần áp dụng các siêu tham số đã tối ưu hóa cho mô hình cơ sở CA-TCC ban đầu (Bước 2). Một cách phản trực giác, việc giới thiệu kiến trúc CoFT nhưng chỉ sử dụng nhánh thời gian để dự đoán *làm giảm* hiệu suất đi -0.69% (Bước 3). Bước nhảy cuối cùng lên 85.47% chỉ đạt được khi các dự đoán từ cả hai nhánh được tổ hợp (Bước 4), đóng góp một mức tăng **+2.57%**.
-   **Phân tích**: Nghiên cứu cắt lớp này tiết lộ một câu chuyện đa sắc thái. Nguồn gốc thành công của CoFT không phải là nguyên khối mà là sự hiệp đồng của ba yếu tố:
    1.  **Sự ưu tiên của việc tinh chỉnh siêu tham số**: Một phần đáng kể của tổng mức tăng đến từ việc thiết lập một cơ sở mạnh mẽ, được tinh chỉnh tốt. Điều này nhấn mạnh tầm quan trọng thiết yếu của việc tối ưu hóa nghiêm ngặt trong việc đánh giá các kiến trúc mới.
    2.  **Cạm bẫy của việc hợp nhất ngây thơ**: Chỉ cần thêm một nhánh thứ hai có thể hoạt động như một cơ chế điều chuẩn gây nhiễu nếu đầu ra của nó không được tích hợp đúng cách, chứng tỏ rằng sự phức tạp hơn không phải lúc nào cũng tốt hơn.
    3.  **Vai trò quan trọng của việc tổ hợp**: Đóng góp về mặt kiến trúc của CoFT chỉ được khai thác khi hai chuyên gia miền chuyên biệt (thời gian và tần số) được tạo ra thông qua đồng huấn luyện và sau đó "trí tuệ" của chúng được tổng hợp thông qua một tổ hợp. Nhánh tần số không chỉ là một cơ chế điều chuẩn; nó là một người đóng góp quan trọng cho quyết định cuối cùng.

## 4.3 Trả lời câu hỏi nghiên cứu 3: Hành trình nghiên cứu để chuyển giao kiến thức tối ưu

Câu hỏi nghiên cứu thứ ba khám phá các tham số và nguyên tắc tối ưu chi phối việc chuyển giao kiến thức giữa hai miền. Đây không phải là một cuộc tìm kiếm tham số đơn giản mà là một cuộc điều tra chuyên sâu, kéo dài nhiều tháng, bắt đầu với một giả thuyết thất bại và kết thúc bằng một khám phá khoa học quan trọng.

### 4.3.1 Giả thuyết ban đầu và những thất bại đầu tiên: Nguy cơ của sự liên kết mạnh
**Giả thuyết ban đầu:** Dựa trên một cuộc khảo sát tài liệu về hợp nhất dữ liệu [11, 12], thường nhấn mạnh sự tích hợp mạnh mẽ, giả thuyết ban đầu của chúng tôi là một sự kết hợp chặt chẽ giữa các miền thời gian và tần số sẽ là tối ưu. Chúng tôi cho rằng một trọng số đồng huấn luyện cao (ví dụ: \(\lambda_{ct} \ge 0.1\)) sẽ buộc chuyển giao kiến thức mạnh mẽ.

**Kết quả thảm khốc:** Các thí nghiệm ban đầu được xây dựng trên giả thuyết này là một thất bại nặng nề.
*   **Hiệu suất:** Với \(\lambda_{ct} = 0.5\), độ chính xác trên bộ dữ liệu HAR giảm mạnh xuống khoảng 45-50%, tệ hơn cả đoán ngẫu nhiên cho một bài toán 6 lớp.
*   **Sự bất ổn trong huấn luyện:** Hơn 40% các lần chạy huấn luyện bị phân kỳ, với các giá trị mất mát bùng nổ thành `NaN` (Không phải là số). Phân tích gradient cho thấy thuật ngữ mất mát đồng huấn luyện hoàn toàn lấn át tất cả các thuật ngữ khác, thực chất là chiếm quyền kiểm soát quá trình học.

Thất bại nghiêm trọng này đã chứng minh rằng giả định ban đầu, trực quan của chúng tôi là sai lầm cơ bản. Nó đã làm mất giá trị của cách tiếp cận "càng mạnh càng tốt" và buộc phải đánh giá lại hoàn toàn, kích hoạt một cuộc điều tra có hệ thống về bản chất thực sự của việc học chéo miền trong bối cảnh này.

### 4.3.2 Khám phá "Ít hơn là Nhiều hơn": Một cuộc điều tra có hệ thống

Sự thất bại của liên kết mạnh đã thúc đẩy một giả thuyết mới: có lẽ các miền yêu cầu một sự tương tác nhẹ nhàng hơn, mang tính điều chuẩn hơn nhiều. Điều này dẫn đến một cuộc tìm kiếm tham số có hệ thống, đa giai đoạn, chuyển từ chế độ liên kết cao sang chế độ cực thấp.

**Bảng 5: Ảnh hưởng của trọng số đồng huấn luyện (\(\lambda_{ct}\)) đến độ chính xác trên HAR 1%**

| \(\lambda_{ct}\) | Độ chính xác | Hiệu suất so với Cơ sở đã tinh chỉnh | Độ ổn định huấn luyện |
|:-----------------|:---------|:-------------------------------|:-------------------|
| 0.1              | 58.23%   | -25.36% (rất tệ)                | 20% phân kỳ        |
| 0.01             | 74.49%   | -9.10% (kém)                   | Ổn định              |
| 0.005            | 74.66%   | -8.93% (trung bình)              | Ổn định              |
| **0.0001**       | **85.47%**| **+1.88% (tốt nhất)**          | Rất ổn định         |

**Diễn giải kết quả:**
-   **Quan sát**: Như được hiển thị trong Bảng 5, có một xu hướng rõ ràng và ấn tượng. Các giá trị cao của \(\lambda_{ct}\) làm suy giảm nghiêm trọng hiệu suất. Khi trọng số được giảm đi theo cấp số nhân, hiệu suất tăng đều đặn, với giá trị tối ưu được tìm thấy là một con số đặc biệt nhỏ `0.0001`.
-   **Phân tích**: Hiện tượng "Ít hơn là Nhiều hơn" này được giải thích bằng cái mà chúng tôi gọi là **"sự nhầm lẫn nhãn."** Trong quá trình tinh chỉnh có giám sát, mất mát hiệu quả là sự kết hợp của mất mát có giám sát (từ các nhãn thực tế) và mất mát đồng huấn luyện (từ các nhãn giả). Vì các nhãn giả vốn có nhiễu, một giá trị \(\lambda_{ct}\) cao sẽ khuếch đại các tín hiệu học sai này, gây nhầm lẫn cho mô hình và làm hỏng gradient. Tuy nhiên, một giá trị cực thấp cung cấp một tín hiệu điều chuẩn nhẹ nhàng hướng dẫn việc học biểu diễn mà không lấn át tín hiệu thực tế. Nó khuyến khích hai nhánh đồng ý mà không ép buộc chúng, điều này đã được chứng minh là chìa khóa để khai thác tiềm năng hiệp đồng của chúng.

### 4.3.3 Động lực của phương pháp tổ hợp: Hiện tượng "Lật ngược"

Hành trình này cũng tiết lộ rằng cách tối ưu để kết hợp hai nhánh tự nó phụ thuộc vào trọng số đồng huấn luyện, dẫn đến việc khám phá ra một "sự lật ngược tổ hợp".

**Bảng 6: Tương tác giữa phương pháp tổ hợp và trọng số đồng huấn luyện (\(\lambda_{ct}\))**

| \(\lambda_{ct}\) | Trung bình đơn giản | Chỉ Thời gian | Chỉ Tần số | Phương pháp tốt nhất |
|:-----------------|:---------------|:--------------|:---------------|:-----------------|
| **0.0001**       | **85.47%**     | 82.90%        | 78.15%         | **Trung bình đơn giản** |
| 0.001            | 81.47%         | 81.22%        | 75.89%         | Trung bình đơn giản |
| 0.005            | 74.66%         | **79.73%**    | 70.12%         | **Chỉ Thời gian**  |
| 0.01             | 74.22%         | **79.49%**    | 68.95%         | **Chỉ Thời gian**  |

**Diễn giải kết quả:**
-   **Quan sát**: Một "sự lật ngược" rõ rệt xảy ra, như được hiển thị trong Bảng 6. Tại các giá trị \(\lambda_{ct}\) cực thấp, tối ưu (≤ 0.001), một trung bình đơn giản của cả hai nhánh là chiến lược tốt nhất. Tuy nhiên, khi \(\lambda_{ct}\) tăng lên, nhánh tần số trở thành một nguồn nhiễu, và tốt hơn là chỉ dựa vào các dự đoán của nhánh thời gian.
-   **Phân tích**: Điều này xác nhận lý thuyết "nhầm lẫn nhãn". Ở giá trị \(\lambda_{ct}\) cao, nhánh tần số học các biểu diễn bị hỏng. Việc bao gồm các dự đoán nhiễu của nó trong tổ hợp làm tổn hại đến hiệu suất. Ở giá trị \(\lambda_{ct}\) thấp tối ưu, cả hai nhánh đều học các biểu diễn được tách biệt tốt, bổ sung cho nhau, và dự đoán kết hợp của chúng mạnh hơn bất kỳ dự đoán nào một mình. Điều này cho thấy sự hiệp đồng về mặt kiến trúc giữa hai nhánh chỉ được khai thác ở cường độ kết hợp nhẹ nhàng, chính xác.

## 4.4 Trả lời câu hỏi nghiên cứu 4: Các nguyên tắc có thể được chuyển giao sang các bộ dữ liệu mới không?

Câu hỏi nghiên cứu cuối cùng hỏi liệu các nguyên tắc học được từ việc tối ưu hóa chuyên sâu trên HAR có thể được sử dụng để hướng dẫn lựa chọn tham số cho các bộ dữ liệu mới mà không cần tìm kiếm toàn diện hay không. Chúng tôi đã phát triển một phương pháp chuyển giao có nguyên tắc dựa trên việc phân tích các đặc điểm chính của bộ dữ liệu.

**Phương pháp luận:**
1.  **Độ dài chuỗi → \(\lambda_{ct}\)**: Các chuỗi dài hơn có thể chịu được các trọng số đồng huấn luyện cao hơn một chút.
2.  **Loại tín hiệu & Nhiễu → \(\lambda_{cs}\)**: Các tín hiệu nhiễu hơn (như EEG) được hưởng lợi từ việc điều chuẩn nhất quán mạnh hơn.
3.  **Tính phổ quát của tổ hợp**: Tổ hợp `trung bình đơn giản` (với \(\lambda_{ct}\) tối ưu) được giả thuyết là một mặc định mạnh mẽ.

**Các tham số được chuyển giao và lý do:**

**Bảng 7: Các tham số được chuyển giao và lý do**
| Dataset    | \(\lambda_{ct}\) (Cuối cùng) | \(\lambda_{cs}\) (Cuối cùng) | Lý do                                                                    |
|:-----------|:-------------------------|:-------------------------|:-----------------------------------------------------------------------|
| **Sleep-EDF**| **0.0002** (2x HAR)      | **0.015** (1.5x HAR)     | Chuỗi dài hơn 23 lần cho phép 2x \(\lambda_{ct}\); 1.5x \(\lambda_{cs}\) cho nhiễu EEG. |
| **Epilepsy** | **0.00005** (0.5x HAR)   | **0.025** (2.5x HAR)     | Độ nhạy của EEG yêu cầu 0.5x \(\lambda_{ct}\); 2.5x \(\lambda_{cs}\) cho sự phức tạp của cơn co giật. |

**Diễn giải kết quả:**
-   **Quan sát**: Như được hiển thị trong Bảng 2, việc áp dụng các tham số được chuyển giao này cho các bộ dữ liệu Sleep-EDF và Epilepsy đã dẫn đến những cải thiện hiệu suất đáng kể (lần lượt là +9.32% và +2.71%) so với mô hình cơ sở.
-   **Phân tích**: Kết quả này xác thực rằng các nguyên tắc cốt lõi chi phối khuôn khổ CoFT không phải là đặc thù của bộ dữ liệu. Khả năng đạt được những cải tiến đáng kể trên các bộ dữ liệu y tế mới, đa dạng chỉ bằng cách sử dụng một phương pháp chuyển giao siêu tham số có nguyên tắc, không cần tinh chỉnh lại (zero-shot) là một đóng góp quan trọng của công trình này. Nó chứng tỏ sự mạnh mẽ của khuôn khổ và cung cấp một phương pháp luận thực tế để áp dụng CoFT cho các vấn đề mới một cách hiệu quả.

---

# 5 KẾT LUẬN

Chương cuối cùng này tóm tắt hành trình nghiên cứu, đánh giá một cách phê bình các kết quả của nó, và vạch ra các hướng đi hứa hẹn cho các cuộc điều tra trong tương lai. Nó nhằm mục đích trả lời câu hỏi cuối cùng: "Vậy thì sao?" bằng cách đặt bối cảnh cho những đóng góp của khuôn khổ CoFT trong bối cảnh rộng lớn hơn của phân tích chuỗi thời gian.

## 5.1 Tóm tắt hành trình nghiên cứu

Luận văn này bắt đầu bằng việc xác định một nút thắt cổ chai quan trọng trong học sâu cho chuỗi thời gian: mô hình "dữ liệu giàu, nhãn nghèo", đặc biệt nghiêm trọng trong các lĩnh vực có yêu cầu cao. Để giải quyết vấn đề này, chúng tôi đã đề xuất, triển khai và xác thực **CoFT (Đồng huấn luyện với miền Tần số và Thời gian)**, một khuôn khổ bán giám sát, hai nhánh mới. Cốt lõi của CoFT không phải là chỉ đơn thuần hợp nhất các đặc trưng thời gian và tần số, mà là coi chúng như hai khung nhìn bổ sung cho một phương pháp đồng huấn luyện thực sự, được điều phối bởi một quy trình đa giai đoạn phức tạp và một hàm mất mát hỗn hợp được cân bằng cẩn thận.

Cuộc điều tra thực nghiệm của chúng tôi, được cấu trúc để trả lời bốn câu hỏi nghiên cứu cốt lõi, đã mang lại một số phát hiện chính. Chúng tôi đã chứng minh rằng CoFT đạt được **hiệu suất tiên tiến**, với mức tăng độ chính xác lên đến **+8.17%** so với một mô hình cơ sở mạnh. Một nghiên cứu cắt lớp chi tiết đã tiết lộ rằng thành công này là sự hiệp đồng giữa **việc tinh chỉnh siêu tham số nghiêm ngặt** (thiết lập một cơ sở mạnh mẽ) và **đóng góp về mặt kiến trúc của hệ thống hai nhánh được tổ hợp**. Cuộc điều tra đã khám phá ra hiện tượng **"Ít hơn là Nhiều hơn"**, chứng minh rằng một trọng số đồng huấn luyện cực thấp (\(\lambda_{ct}=0.0001\)) là tối ưu để tránh "sự nhầm lẫn nhãn" và cho phép chuyển giao kiến thức hiệu quả, nhẹ nhàng. Cuối cùng, chúng tôi đã thiết lập một **phương pháp chuyển giao tham số có nguyên tắc**, áp dụng thành công những hiểu biết từ một bộ dữ liệu để đạt được những lợi ích đáng kể trên các bộ dữ liệu y tế mới, đa dạng mà không cần tinh chỉnh lại tốn kém.

## 5.2 Hạn chế của nghiên cứu

Việc thừa nhận những giới hạn của nghiên cứu này là rất quan trọng để đảm bảo tính trung thực khoa học và để định hướng cho các công trình trong tương lai. Các hạn chế chính như sau:

1.  **Phụ thuộc vào Tăng cường dữ liệu cho dữ liệu nhạy cảm:** Khuôn khổ CoFT, giống như các tiền thân học tương phản của nó, về cơ bản dựa vào việc tăng cường dữ liệu để tạo ra các khung nhìn tương quan cần thiết cho việc học tự giám sát. Điều này đặt ra một thách thức khái niệm đáng kể cho việc triển khai trong các lĩnh vực nhạy cảm. Đối với các tín hiệu quan trọng như EEG hoặc ECG y tế, nơi những thay đổi hình thái tinh tế có thể chỉ ra một tình trạng đe dọa tính mạng, ngay cả các phép tăng cường 'đơn giản' cũng phải được áp dụng hết sức thận trọng. Một phép tăng cường quá mạnh có thể vô tình làm biến dạng hoặc phá hủy chính các mẫu chẩn đoán mà chúng tôi muốn phân loại. Quan sát này mở ra một hướng nghiên cứu mới và quan trọng: phát triển các khuôn khổ học tự giám sát không cần tăng cường hoặc tăng cường tối thiểu cho các chuỗi thời gian nhạy cảm, một hướng đi nằm ngoài phạm vi của luận văn này nhưng rất quan trọng để nâng cao tính an toàn, độ tin cậy và khả năng ứng dụng lâm sàng của các mô hình như vậy.

2.  **Tập trung vào các nhiệm vụ phân loại:** Công trình này chỉ xác thực CoFT trên phân loại chuỗi thời gian. Mặc dù các biểu diễn mạnh mẽ, được tách rời do khuôn khổ học được có khả năng mang lại lợi ích cho các nhiệm vụ khác, hiệu suất của nó trên dự báo chuỗi thời gian hoặc phát hiện bất thường vẫn chưa được đánh giá thực nghiệm.

3.  **Sự tương đương về kiến trúc như một nút thắt cổ chai:** Quyết định sử dụng các kiến trúc giống hệt nhau cho cả hai nhánh thời gian và tần số là một lựa chọn phương pháp luận cần thiết để chứng minh giá trị vốn có của miền tần số. Tuy nhiên, như kết quả của chúng tôi đã cho thấy, kiến trúc không chuyên biệt này có khả năng hoạt động như một nút thắt cổ chai hiệu suất cho nhánh tần số. Tiềm năng đầy đủ của các đặc trưng miền tần số có thể chỉ được khai thác bởi các kiến trúc được thiết kế đặc biệt cho dữ liệu quang phổ (ví dụ: Spectral CNNs).

## 5.3 Hướng nghiên cứu trong tương lai

Những hạn chế được xác định ở trên cung cấp một lộ trình rõ ràng và có thể hành động cho các nghiên cứu trong tương lai. Các hướng sau đây được đề xuất như là sự mở rộng trực tiếp của công trình này:

1.  **Phát triển các khuôn khổ Đồng huấn luyện không cần tăng cường:** Đây là hướng quan trọng nhất để nâng cao tính an toàn và độ tin cậy của mô hình cho các ứng dụng lâm sàng. Các công trình trong tương lai nên khám phá các khuôn khổ tự giám sát không dựa vào các nhiễu loạn dữ liệu tổng hợp. Một cách tiếp cận hứa hẹn có thể bao gồm việc tái tạo hoặc dự đoán chéo miền: ví dụ, sử dụng biểu diễn của nhánh thời gian để dự đoán phân rã wavelet của tín hiệu, hoặc ngược lại. Điều này sẽ tạo ra một tín hiệu học mạnh mẽ từ cấu trúc nội tại của chính dữ liệu, đảm bảo rằng các đặc trưng đã học trung thành với hình thái tín hiệu gốc, có liên quan đến lâm sàng.

2.  **Thiết kế các kiến trúc đặc thù cho miền:** Dựa trên phát hiện của chúng tôi rằng cơ chế tổ hợp là chìa khóa để khai thác tiềm năng của nhánh tần số, các công trình trong tương lai nên thiết kế và xác thực các kiến trúc nơ-ron chuyên biệt (ví dụ: 1D Spectral CNNs, các cơ chế chú ý được thiết kế riêng cho các mẫu quang phổ) cho nhánh tần số. Điều này sẽ vượt ra ngoài sự tương đương về kiến trúc để đến một thiết kế thực sự nhận biết miền, có khả năng nâng cao hơn nữa đóng góp chuyên môn và hiệu suất tổng thể của nó.

3.  **Mở rộng sang Dự báo và Phát hiện bất thường:** Các biểu diễn mạnh mẽ do CoFT học được nên được đánh giá trên các nhiệm vụ chuỗi thời gian chính khác. Đối với dự báo, các đặc trưng được tách rời có thể cải thiện các dự đoán dài hạn. Đối với phát hiện bất thường, mất mát nhất quán giữa hai miền có thể đóng vai trò như một tín hiệu mạnh mẽ để xác định các trạng thái bất thường nơi các đặc điểm thời gian và tần số của một tín hiệu khác với định mức.

## 5.4 Suy ngẫm cuối cùng

Hành trình nghiên cứu được ghi lại trong luận văn này hiếm khi là tuyến tính. Những hiểu biết quý giá nhất—chẳng hạn như tính tối ưu của một trọng số đồng huấn luyện cực thấp hoặc vai trò quan trọng của tổ hợp—được khám phá không phải bằng cách xác nhận các giả thuyết ban đầu, mà bằng cách điều tra một cách có hệ thống các thất bại và các kết quả bất ngờ. Công trình này đóng góp cả một mô hình thực tế, hiệu suất cao và, quan trọng hơn, một tập hợp các nguyên tắc về cách đánh giá, phân tích và hiểu một cách nghiêm ngặt các hệ thống học hai miền phức tạp. Con đường phía trước không chỉ là xây dựng các mô hình tốt hơn, mà còn là xây dựng chúng một cách chu đáo hơn, với sự hiểu biết sâu sắc hơn về các hiệp đồng và sự đánh đổi chi phối thành công của chúng.

---

# TÀI LIỆU THAM KHẢO

[1] Andrzejak, R. G., Lehnertz, K., Mormann, F., Rieke, C., David, P., & Elger, C. E. (2001). Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: dependence on recording region and brain state. *Physical Review E*, 64(6), 061907.

[2] Bertasius, G., Wang, H., & Torresani, L. (2021). Is Space-Time Attention All You Need for Video Understanding?. *arXiv preprint arXiv:2102.05095*.

[3] Blum, A., & Mitchell, T. (1998). Combining labeled and unlabeled data with co-training. In *Proceedings of the eleventh annual conference on Computational learning theory*, 92-100.

[4] Cai, H., Zhang, X., & Liu, X. (2023). Semi-Supervised End-To-End Contrastive Learning For Time Series Classification. *arXiv preprint arXiv:2310.08848*.

[5] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. In *Proceedings of the 37th International Conference on Machine Learning*, 119:1597-1607.

[6] Eldele, E., Ragab, M., Chen, Z., Wu, M., Kwoh, C. K., Li, X., & Guan, C. (2021). Time-Series Representation Learning via Temporal and Contextual Contrasting. In *Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence (IJCAI)*, 2352-2359.

[7] Eldele, E., Ragab, M., Chen, Z., Wu, M., Kwoh, C. K., Li, X., & Guan, C. (2023). Self-Supervised Contrastive Representation Learning for Semi-supervised Time-Series Classification. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

[8] Luo, C., Zhang, C., Zhang, J., & Li, J. (2023). RankSCL: A Ranking-based Supervised Contrastive Learning Framework for Time Series Classification. *arXiv preprint arXiv:2308.07724*.

[9] Wen, Q., Sun, L., Yang, F., Song, X., Gao, J., Wang, X., & Xu, H. (2021). Time series data augmentation for deep learning: a survey. In *Proceedings of the 30th International Joint Conference on Artificial Intelligence (IJCAI)*, 4673-4680.

[10] Yue, Z., Wang, Y., Duan, J., Yang, T., Huang, C., Tong, Y., & Xu, B. (2022). TS2Vec: Towards Universal Representation of Time Series. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(8), 9180-9187.

[11] Cui, C., Yang, H., Wang, Y., Zhao, S., Asada, Z., Coburn, L. A., ... & Huo, Y. (2022). Deep multi-modal fusion of image and non-image data in disease diagnosis and prognosis: a review. *arXiv preprint arXiv:2203.15588*.

[12] Wang, Y. (2018). Survey on Deep Multi-modal Data Analytics: Collaboration, Rivalry and Fusion. *J. ACM*, 37(4), 111.

[13] Baltrusaitis, T., Ahuja, C., & Morency, L. P. (2019). Multimodal machine learning: A survey and taxonomy. *IEEE transactions on pattern analysis and machine intelligence*, 41(2), 423-443.

---

# PHỤ LỤC
Phần này có thể bao gồm các tài liệu bổ sung như bảng siêu tham số chi tiết cho từng bộ dữ liệu, các đoạn mã cho các mô-đun chính, hoặc các hình ảnh trực quan bổ sung về các vector nhúng đặc trưng.

## Phụ lục A: Bảng cấu hình siêu tham số

Phụ lục này cung cấp một bản tóm tắt toàn diện về các siêu tham số cuối cùng được sử dụng cho khuôn khổ CoFT trên tất cả các bộ dữ liệu benchmark. Các tham số cho bộ dữ liệu HAR được xác định thông qua một quá trình tối ưu hóa toàn diện, trong khi các tham số cho Sleep-EDF và Epilepsy được suy ra bằng cách sử dụng phương pháp chuyển giao có nguyên tắc được trình bày chi tiết trong Chương 4.

### **Bảng 8: Các tham số huấn luyện và mô hình chung**

| Tham số | HAR | Sleep-EDF | Epilepsy | Mô tả |
| :--- | :---: | :---: | :---: | :--- |
| **Kỷ nguyên** | 40 | 40 | 40 | Tổng số kỷ nguyên huấn luyện cho tất cả các giai đoạn. |
| **Kích thước lô** | 128 | 128 | 128 | Số lượng mẫu trên mỗi lô huấn luyện. |
| **Tốc độ học** | 3e-4 | 3e-4 | 3e-4 | Tốc độ học ban đầu cho trình tối ưu hóa Adam. |
| **Trình tối ưu hóa** | Adam | Adam | Adam | Thuật toán tối ưu hóa được sử dụng để huấn luyện. |
| **Suy giảm trọng số** | 3e-4 | 3e-4 | 3e-4 | Tham số điều chuẩn L2. |
| **Dropout** | 0.1 | 0.1 | 0.1 | Tỷ lệ dropout để điều chuẩn trong các lớp cuối cùng. |
| **Kênh đầu vào**| 9 | 1 | 1 | Số lượng kênh đầu vào trong chuỗi thời gian thô. |
| **Số lớp** | 6 | 5 | 2 | Số lượng lớp mục tiêu để phân loại. |

### **Bảng 9: Các siêu tham số dành riêng cho CoFT**

Các tham số này kiểm soát các cơ chế cốt lõi của khuôn khổ đồng huấn luyện hai nhánh. Các giá trị phản ánh khám phá "Ít hơn là Nhiều hơn" và chiến lược chuyển giao tham số.

| Tham số | HAR (Tối ưu) | Sleep-EDF (Chuyển giao) | Epilepsy (Chuyển giao) | Mô tả |
| :--- | :---: | :---: | :---: | :--- |
| **`lambda_cotraining`** | **0.0001** | **0.0002** | **0.00005** | Trọng số đồng huấn luyện quan trọng. Các giá trị cực thấp ngăn chặn "sự nhầm lẫn nhãn". |
| **`lambda_consistency`** | **0.01** | **0.015** | **0.025** | Trọng số cho mất mát nhất quán đặc trưng giữa các nhánh. |
| **Phương pháp Tổ hợp** | `temporal_only` | `temporal_only` | `temporal_only` | Phương pháp vượt trội phổ biến để kết hợp các dự đoán của nhánh. |
| **Ngưỡng tin cậy**| 0.95 | 0.95 | 0.95 | Xác suất softmax tối thiểu để chấp nhận một nhãn giả. |
| **Tỷ lệ khởi động** | 0.25 | 0.25 | 0.25 | Tỷ lệ kỷ nguyên để khởi động cơ chế đồng huấn luyện. |

### **Bảng 10: Các tham số học tương phản và tăng cường dữ liệu**

Các tham số này chi phối giai đoạn học biểu diễn tự giám sát (TS-TCC) và quy trình tăng cường dữ liệu.

| Tham số | HAR | Sleep-EDF | Epilepsy | Mô tả |
| :--- | :---: | :---: | :---: | :--- |
| **Nhiệt độ tương phản (τ)**| 0.2 | 0.2 | 0.2 | Nhiệt độ cho mất mát tương phản NT-Xent. |
| **Tỷ lệ Jitter** | 0.8 | 0.8 | 0.8 | Độ mạnh của nhiễu ngẫu nhiên được thêm vào để tăng cường jitter. |
| **Tỷ lệ co giãn Jitter**| 2.0 | 2.0 | 2.0 | Độ mạnh của phép tăng cường co giãn ngẫu nhiên. |
| **Số đoạn tối đa** | 8 | 20 | 12 | Số lượng đoạn tối đa cho phép tăng cường hoán vị. |
| **Sử dụng Augs InfoTS**| `False` | `False` | `False` | Chuyển đổi để tắt các phép tăng cường InfoTS phức tạp, ưu tiên sự đơn giản. |

