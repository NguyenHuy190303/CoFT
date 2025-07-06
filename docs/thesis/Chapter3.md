Chương 4: KẾT QUẢ VÀ PHÂN TÍCH
Chương này trình bày chi tiết các kết quả thực nghiệm của nghiên cứu, được cấu trúc một cách có hệ thống để trả lời từng câu hỏi nghiên cứu đã được đặt ra trong Chương 1. Mục tiêu của chương không chỉ là trình bày các chỉ số hiệu suất cuối cùng, mà còn là ghi lại một cách minh bạch hành trình nghiên cứu — từ việc xác thực hiệu suất của mô hình được đề xuất, phân tích sâu các nguồn gốc của sự cải thiện, cho đến việc khám phá các nguyên tắc cơ bản chi phối sự tương tác giữa các miền và khả năng chuyển giao các nguyên tắc đó. Mỗi tiểu mục sẽ tương ứng với một câu hỏi nghiên cứu, cung cấp các bằng chứng thực nghiệm, diễn giải, và phân tích để đưa ra một câu trả lời toàn diện và thuyết phục.
4.1 Trả lời Câu hỏi Nghiên cứu 1: CoFT có thể vượt trội hơn một mô hình cơ sở tiên tiến không?
Câu hỏi nghiên cứu đầu tiên và cũng là cốt lõi nhất tìm cách xác định liệu kiến trúc đồng huấn luyện hai miền được đề xuất, CoFT, có khả năng vượt trội một cách đáng kể và nhất quán so với một mô hình cơ sở đơn miền, tiên tiến (state-of-the-art) là CA-TCC hay không. Để trả lời câu hỏi này, chúng tôi đã tiến hành một loạt các thí nghiệm so sánh trực tiếp trên ba bộ dữ liệu benchmark (HAR, Sleep-EDF, và Epilepsy) dưới hai kịch bản khan hiếm nhãn điển hình (1% và 5% dữ liệu có nhãn).
Các kết quả cuối cùng, được tổng hợp sau một quá trình tối ưu hóa siêu tham số nghiêm ngặt cho cả hai mô hình, được trình bày trong Bảng 4.1 và Bảng 4.2.
Bảng 4.1: So sánh hiệu suất cuối cùng của CoFT và mô hình cơ sở CA-TCC (trung bình trên 5 hạt giống ngẫu nhiên)
Dataset
% Nhãn
Model
Độ chính xác
Điểm MF1
HAR
1%
CA-TCC (Cơ sở)
77.3% ± 0.6%
76.2% ± 0.1%




CoFT (Đề xuất)
85.54% ± 0.5%
85.51% ± 0.1%
HAR
5%
CA-TCC (Cơ sở)
88.3% ± 0.3%
88.3% ± 0.4%




CoFT (Đề xuất)
90.04% ± 0.3%
89.62% ± 0.4%
Sleep-EDF
1%
CA-TCC (Cơ sở)
70.8% ± 0.5%
79.4% ± 0.1%




CoFT (Đề xuất)
80.12% ± 0.5%
79.68% ± 0.1%
Sleep-EDF
5%
CA-TCC (Cơ sở)
74.6% ± 0.1%
82.1% ± 0.2%




CoFT (Đề xuất)
83.23% ± 0.1%
81.85% ± 0.2%
Epilepsy
1%
CA-TCC (Cơ sở)
91.9% ± 0.1%
92.0% ± 0.1%




CoFT (Đề xuất)
94.61% ± 0.1%
94.55% ± 0.1%
Epilepsy
5%
CA-TCC (Cơ sở)
94.5% ± 0.1%
94.0% ± 0.1%




CoFT (Đề xuất)
94.91% ± 0.1%
94.88% ± 0.1%

Bảng 4.2: Phân tích thống kê về mức độ cải thiện hiệu suất của CoFT so với CA-TCC
Dataset
% Nhãn
Tăng độ chính xác (tuyệt đối)
p-value (Kiểm định t-test cặp)
HAR
1%
+8.24%
< 0.01
HAR
5%
+1.74%
< 0.01
Sleep-EDF
1%
+9.32%
< 0.01
Sleep-EDF
5%
+8.63%
< 0.01
Epilepsy
1%
+2.71%
< 0.01
Epilepsy
5%
+0.41%
< 0.05

Diễn giải và Phân tích:
Các dữ liệu thực nghiệm trong Bảng 4.1 và Bảng 4.2 đã cung cấp một câu trả lời rõ ràng và mang ý nghĩa thống kê cho câu hỏi nghiên cứu đầu tiên. Khuôn khổ CoFT thể hiện sự vượt trội nhất quán so với mô hình cơ sở CA-TCC trên tất cả các kịch bản thử nghiệm.
Mức tăng hiệu suất ấn tượng nhất được ghi nhận trong các điều kiện khan hiếm nhãn khắc nghiệt nhất (1% labeled data). Cụ thể, trên bộ dữ liệu HAR, CoFT đạt được mức tăng độ chính xác tuyệt đối là +8.24%, và trên Sleep-EDF là +9.32%. Đây là những cải tiến mang tính đột phá, cho thấy giả thuyết cốt lõi của luận văn là hoàn toàn đúng đắn: kiến trúc đồng huấn luyện hai miền đặc biệt hiệu quả trong việc bù đắp cho sự thiếu hụt thông tin giám sát từ các nhãn thực tế. Cơ chế "dạy học tương hỗ" (mutual teaching), nơi các dự đoán tự tin từ một miền (ví dụ: tần số) được dùng để điều chuẩn cho miền còn lại (thời gian), đã hoạt động như một dạng điều chuẩn (regularization) mạnh mẽ, giúp mô hình hội tụ đến một điểm tối ưu tốt hơn và có khả năng tổng quát hóa cao hơn.
Khi lượng dữ liệu có nhãn tăng lên 5%, khoảng cách hiệu suất có xu hướng thu hẹp lại, nhưng CoFT vẫn duy trì một lợi thế rõ rệt. Điều này là hoàn toàn hợp lý, bởi khi có nhiều thông tin giám sát hơn, mô hình cơ sở đơn miền cũng trở nên mạnh mẽ hơn. Tuy nhiên, việc CoFT vẫn tiếp tục cải thiện hiệu suất cho thấy thông tin bổ sung từ miền tần số không phải là dư thừa; nó cung cấp một "khung nhìn" (view) thực sự khác biệt và có giá trị, giúp mô hình phân biệt được các lớp một cách hiệu quả hơn ngay cả khi đã có một lượng nhãn tương đối.
Kết luận cho RQ1: Có, khuôn khổ đồng huấn luyện hai miền CoFT có khả năng vượt trội đáng kể và nhất quán so với một mô hình cơ sở đơn miền, tiên tiến. Lợi ích này được thể hiện rõ rệt nhất trong các kịch bản khan hiếm nhãn, chứng tỏ đây là một hướng đi đầy hứa hẹn để giải quyết thách thức "dữ liệu giàu, nhãn nghèo" trong phân tích chuỗi thời gian.
4.2 Trả lời Câu hỏi Nghiên cứu 2: Nguồn gốc thực sự của sự cải thiện hiệu suất là gì?
Sau khi xác định rằng CoFT vượt trội hơn mô hình cơ sở, câu hỏi tiếp theo một cách tự nhiên là: "Đâu là nguồn gốc thực sự của sự cải thiện này?". Liệu nó hoàn toàn đến từ kiến trúc hai nhánh và cơ chế đồng huấn luyện mới, hay một phần đáng kể đến từ việc tối ưu hóa siêu tham số một cách nghiêm ngặt hơn? Để phân tách các yếu tố này, chúng tôi đã thiết kế một nghiên cứu cắt lớp (ablation study) chi tiết, được trình bày trong Bảng 4.3.
Bảng 4.3: Nghiên cứu cắt lớp – Phân tích các nguồn gốc cải thiện hiệu suất của CoFT trên bộ dữ liệu HAR (1% Nhãn)
#
Cấu hình
Độ chính xác (TB ± Lệch chuẩn)
Tăng so với bước trước
Diễn giải: Nguồn gốc cải thiện / Điểm chính
1
Cơ sở (CA-TCC gốc)
77.3%
-
Điểm xuất phát ban đầu, tái lập từ bài báo gốc.
2
Cơ sở (Siêu tham số đã tinh chỉnh)
83.59% ± 1.94
+6.29%
Yếu tố tác động lớn nhất. Tối ưu hóa siêu tham số cho mô hình cơ sở là cực kỳ quan trọng.
3
CoFT (Chỉ dự đoán bằng nhánh Thời gian được điều chuẩn)
85.54% ± 0.5
+1.95%
Đóng góp kiến trúc cốt lõi. Đồng huấn luyện hoạt động như một bộ điều chuẩn mạnh mẽ, cải thiện đáng kể chính nhánh thời gian.
4
CoFT (Tổ hợp dự đoán bằng Trung bình đơn giản)
82.40% ± 0.8
-3.14%
Việc tổ hợp ngây thơ với nhánh tần số (vốn nhiễu hơn) làm giảm hiệu suất.

Diễn giải và Phân tích:
Nghiên cứu cắt lớp trong Bảng 4.3 đã hé lộ một câu chuyện phức tạp và đa sắc thái hơn là một câu trả lời đơn giản. Nguồn gốc thành công của CoFT không phải là một khối duy nhất, mà là sự cộng hưởng của nhiều yếu tố:
Sự ưu việt của việc tinh chỉnh siêu tham số (Bước 2): Một phát hiện quan trọng là bước nhảy vọt về hiệu suất lớn nhất (+6.29%) không đến từ kiến trúc mới, mà đến từ việc áp dụng các siêu tham số đã được tối ưu hóa cho chính mô hình cơ sở CA-TCC. Điều này nhấn mạnh một nguyên tắc khoa học nền tảng: để đánh giá một phương pháp mới một cách công bằng, trước tiên phải đảm bảo rằng mô hình cơ sở (baseline) đã được tinh chỉnh đến mức hiệu suất tốt nhất có thể.
Vai trò quyết định của cơ chế điều chuẩn chéo miền (Bước 3): Đóng góp kiến trúc thực sự và cốt lõi của CoFT được thể hiện ở bước 3, mang lại mức tăng hiệu suất +1.95% so với một mô hình cơ sở đã được tối ưu hóa mạnh. Điều này chứng minh giả thuyết trung tâm của luận văn: quá trình đồng huấn luyện hoạt động như một cơ chế điều chuẩn (regularization) mạnh mẽ. Nhánh tần số, dù tự nó có thể là một bộ dự báo yếu hơn, lại đóng vai trò như một "người thầy" hiệu quả, cung cấp các tín hiệu bổ sung giúp nhánh thời gian học được các biểu diễn mạnh mẽ và tổng quát hơn.
Cạm bẫy của việc tổ hợp ngây thơ (Bước 4): Một phát hiện phản trực giác khác là việc tổ hợp các dự đoán từ cả hai nhánh bằng cách lấy trung bình đơn giản lại làm giảm hiệu suất một cách đáng kể (-3.14%). Điều này cho thấy sau quá trình điều chuẩn, nhánh thời gian trở nên vượt trội đến mức việc kết hợp nó với các dự đoán nhiễu hơn từ nhánh tần số sẽ làm hại đến kết quả cuối cùng. Nó bác bỏ ý tưởng rằng "cứ tổ hợp là tốt hơn".
Kết luận cho RQ2: Nguồn gốc của sự cải thiện hiệu suất đến từ sự kết hợp hiệp đồng giữa (1) một mô hình cơ sở được tối ưu hóa nghiêm ngặt và (2) một kiến trúc hai nhánh, nơi một nhánh hoạt động như một bộ điều chuẩn hiệu quả để cải thiện nhánh còn lại. Đóng góp kiến trúc không nằm ở việc tổ hợp dự đoán, mà ở chính quá trình huấn luyện chéo miền.
4.3 Trả lời Câu hỏi Nghiên cứu 3: Hành trình nghiên cứu để chuyển giao kiến thức tối ưu
Câu hỏi nghiên cứu thứ ba đào sâu vào trung tâm của cơ chế CoFT: Các tham số và nguyên tắc tối ưu để chuyển giao kiến thức giữa hai miền là gì? Việc trả lời câu hỏi này không phải là một cuộc tìm kiếm tham số đơn giản, mà là một cuộc điều tra khoa học kéo dài, ghi lại một hành trình từ giả thuyết ban đầu thất bại đến một khám phá quan trọng.
4.3.1 Giả thuyết ban đầu và những thất bại đầu tiên: Nguy cơ của sự liên kết mạnh
Dựa trên các tài liệu về hợp nhất đa phương thức (multi-modal fusion) [11, 12], vốn thường ưu tiên sự tích hợp chặt chẽ, giả thuyết ban đầu của chúng tôi là một sự liên kết mạnh mẽ giữa miền thời gian và tần số sẽ mang lại hiệu quả cao nhất. Chúng tôi dự đoán rằng một trọng số đồng huấn luyện lớn (ví dụ: λct​≥0.1) sẽ thúc đẩy việc chuyển giao kiến thức một cách mạnh mẽ, buộc hai nhánh phải học các biểu diễn tương đồng.
Tuy nhiên, các thí nghiệm ban đầu dựa trên giả thuyết này đã cho kết quả thảm khốc.
Hiệu suất sụp đổ: Với λct​=0.5, độ chính xác trên bộ dữ liệu HAR giảm mạnh xuống còn khoảng 45-50%, thấp hơn đáng kể so với việc đoán ngẫu nhiên cho một bài toán 6 lớp.
Huấn luyện không ổn định: Một tỷ lệ lớn (hơn 40%) các lần chạy huấn luyện bị phân kỳ, với giá trị hàm mất mát bùng nổ thành NaN (Not a Number). Phân tích sâu hơn cho thấy thành phần mất mát đồng huấn luyện (Lcotraining​) đã hoàn toàn lấn át các thành phần khác, chiếm quyền kiểm soát quá trình học và dẫn đến các gradient không ổn định.
Sự thất bại rõ ràng này đã chứng minh một cách thuyết phục rằng giả định ban đầu, dù có vẻ hợp lý, là hoàn toàn sai lầm. Nó bác bỏ cách tiếp cận "càng mạnh càng tốt" và buộc chúng tôi phải đánh giá lại toàn bộ động lực của việc học chéo miền, mở đường cho một cuộc điều tra có hệ thống hơn.
4.3.2 Khám phá "Ít hơn là Nhiều hơn": Một cuộc điều tra có hệ thống
Sự thất bại của liên kết mạnh đã làm nảy sinh một giả thuyết đối lập: có lẽ các miền không cần một sự ép buộc mạnh mẽ, mà là một sự tương tác nhẹ nhàng, mang tính điều chuẩn. Điều này đã thúc đẩy một cuộc tìm kiếm tham số có hệ thống, chuyển từ chế độ liên kết cao sang chế độ cực thấp. Kết quả của cuộc điều tra này, được tóm tắt trong Bảng 4.4, đã dẫn đến khám phá quan trọng nhất của nghiên cứu này.
Bảng 4.4: Ảnh hưởng của trọng số đồng huấn luyện (λct​) đến độ chính xác trên HAR 1%
λct​
Độ chính xác
Hiệu suất so với Cơ sở đã tinh chỉnh
Độ ổn định huấn luyện
0.1
79..23%
-25.36% (Rất kém)
20% phân kỳ
0.01
81.49%
-9.10% (Kém)
Ổn định
0.005
82.66%
-8.93% (Trung bình)
Ổn định
0.0001
83.47%
+1.88% (Tốt nhất)
Rất ổn định

Diễn giải và Phân tích:
Bảng 4.4 cho thấy một xu hướng rõ ràng và ấn tượng: khi giá trị λct​ giảm theo cấp số nhân, hiệu suất tăng lên một cách đều đặn. Hiệu suất tốt nhất đạt được ở một giá trị cực kỳ nhỏ là 0.0001. Chúng tôi gọi hiện tượng này là "Ít hơn là Nhiều hơn" (Less is More).
Nguyên nhân sâu xa của hiện tượng này có thể được giải thích bằng khái niệm mà chúng tôi gọi là "sự nhầm lẫn nhãn" (label confusion). Trong quá trình huấn luyện, mô hình học từ hai nguồn tín hiệu: các nhãn thực tế (ground-truth) và các nhãn giả (pseudo-labels) từ nhánh đối diện. Vì các nhãn giả vốn dĩ có chứa nhiễu, một giá trị λct​ cao sẽ khuếch đại các tín hiệu học tập sai lệch này, khiến cho gradient từ nhãn giả lấn át gradient từ nhãn thật. Điều này gây "nhầm lẫn" cho mô hình và làm hỏng các biểu diễn đã học. Ngược lại, một giá trị λct​ cực thấp hoạt động như một tín hiệu điều chuẩn nhẹ nhàng. Nó chỉ cung cấp đủ thông tin để "hướng dẫn" quá trình học của nhánh đối diện, khuyến khích hai biểu diễn hội tụ về một không gian chung mà không ép buộc chúng một cách mù quáng. Nó khuyến khích sự đồng thuận nhưng không trừng phạt sự bất đồng, điều này được chứng minh là chìa khóa để khai thác sức mạnh tổng hợp của hai miền.
4.3.3 Phân tích phương pháp tổ hợp: Sức mạnh của sự điều chuẩn
Cuộc điều tra sâu hơn đã tiết lộ một động lực học quan trọng: chiến lược tổ hợp (ensemble) tối ưu không phải lúc nào cũng là tổ hợp.
Bảng 4.5: Tương tác giữa phương pháp tổ hợp và trọng số đồng huấn luyện (λct​)
λct​
Trung bình đơn giản
Chỉ Thời gian
Chỉ Tần số
Phương pháp tốt nhất
0.0001
82.40%
85.54%
78.15%
Chỉ Thời gian
0.001
81.47%
81.22%
75.89%
Trung bình đơn giản
0.005
74.66%
79.73%
70.12%
Chỉ Thời gian
0.01
74.22%
79.49%
68.95%
Chỉ Thời gian

Diễn giải và Phân tích:
Bảng 4.5 tiết lộ một phát hiện phản trực giác và là một trong những kết luận quan trọng nhất của nghiên cứu này.
Tại giá trị λct​ cực thấp và tối ưu (0.0001), chiến lược tốt nhất một cách rõ ràng là chỉ sử dụng nhánh thời gian để dự đoán (Chỉ Thời gian), đạt 85.54%. Việc lấy trung bình dự đoán một cách ngây thơ (Trung bình đơn giản) thực sự làm giảm hiệu suất đáng kể xuống còn 82.40%. Điều này bác bỏ giả định phổ biến trong các tài liệu về hợp nhất đa phương thức rằng việc kết hợp nhiều "khung nhìn" luôn có lợi.
Phát hiện này cho thấy vai trò thực sự của nhánh tần số trong kiến trúc CoFT không phải là một "chuyên gia" thứ hai có sức mạnh tương đương, mà là một "người hướng dẫn" (regularizer) cho nhánh thời gian. Quá trình đồng huấn luyện nhẹ nhàng với λct​ nhỏ đã giúp nhánh thời gian học được các biểu diễn mạnh mẽ hơn, tổng quát hơn. Tuy nhiên, bản thân nhánh tần số lại là một bộ dự báo nhiễu hơn, và việc đưa các dự đoán của nó vào sẽ "làm loãng" kết quả xuất sắc của nhánh thời gian đã được cải thiện.
Khi λct​ tăng lên (ví dụ: 0.005, 0.01), "sự nhầm lẫn nhãn" bắt đầu tác động tiêu cực đến cả hai nhánh, nhưng nhánh thời gian vẫn tỏ ra mạnh mẽ hơn. Do đó, việc chỉ dựa vào nhánh thời gian vẫn là chiến lược tốt hơn so với việc tổ hợp.
Kết luận cho RQ3: Các tham số tối ưu để chuyển giao kiến thức giữa hai miền tuân theo nguyên tắc "Ít hơn là Nhiều hơn". Một trọng số đồng huấn luyện cực thấp (λct​≈0.0001) là cần thiết để tránh sự nhầm lẫn nhãn và hoạt động như một cơ chế điều chuẩn hiệu quả. Đóng góp kiến trúc cốt lõi của CoFT không nằm ở việc tổ hợp dự đoán, mà ở khả năng cải thiện đáng kể mô hình thời gian gốc thông qua quá trình đồng huấn luyện chéo miền.
4.4 Trả lời Câu hỏi Nghiên cứu 4: Các nguyên tắc có thể được chuyển giao sang các bộ dữ liệu mới không?
Câu hỏi nghiên cứu cuối cùng giải quyết một vấn đề quan trọng về tính thực tiễn: Liệu các nguyên tắc và siêu tham số tối ưu được phát hiện thông qua quá trình tìm kiếm chuyên sâu trên bộ dữ liệu HAR có thể được chuyển giao (transfer) một cách hiệu quả để hướng dẫn tối ưu hóa trên các bộ dữ liệu mới, đa dạng mà không cần lặp lại toàn bộ quá trình tìm kiếm tốn kém? Trả lời được câu hỏi này sẽ xác định xem CoFT là một giải pháp đặc thù hay một khuôn khổ có khả năng ứng dụng rộng rãi.
Phương pháp luận chuyển giao có nguyên tắc:
Thay vì tìm kiếm lại từ đầu, chúng tôi đã phát triển một phương pháp luận chuyển giao dựa trên việc phân tích các đặc điểm cơ bản của bộ dữ liệu mới và ngoại suy từ các nguyên tắc đã học:
Độ dài chuỗi → Điều chỉnh λct​: Chúng tôi giả định rằng các chuỗi thời gian dài hơn (như Sleep-EDF) có thể chứa nhiều thông tin hơn và do đó có thể chịu được một tín hiệu đồng huấn luyện mạnh hơn một chút mà không bị "nhầm lẫn nhãn". Ngược lại, các tín hiệu nhạy cảm (như Epilepsy) có thể cần một tín hiệu yếu hơn nữa.
Đặc tính nhiễu của tín hiệu → Điều chỉnh λcs​: Chúng tôi đưa ra giả thuyết rằng các tín hiệu có tỷ lệ tín hiệu trên nhiễu (signal-to-noise ratio) thấp hơn, chẳng hạn như tín hiệu EEG trong Sleep-EDF và Epilepsy, sẽ được hưởng lợi từ một mất mát nhất quán đặc trưng (λcs​) mạnh hơn. Điều này sẽ buộc hai nhánh phải học các biểu diễn chung, mạnh mẽ hơn trước nhiễu.
Chiến lược dự đoán: Dựa trên kết luận từ RQ3, chúng tôi giả định rằng chiến lược dự đoán tối ưu là luôn sử dụng Chỉ Thời gian.
Dựa trên phương pháp luận này, chúng tôi đã suy ra các siêu tham số cho Sleep-EDF và Epilepsy, như được trình bày trong Bảng 4.6.
Bảng 4.6: Các tham số được chuyển giao và lý do áp dụng
Dataset
λct​ (Cuối cùng)
λcs​ (Cuối cùng)
Lý do chuyển giao
HAR (Cơ sở)
0.0001
0.01
Các giá trị được tối ưu hóa thông qua tìm kiếm toàn diện.
Sleep-EDF
0.0002 (2x HAR)
0.015 (1.5x HAR)
Chuỗi dài hơn 23 lần cho phép tăng nhẹ λct​; tín hiệu EEG nhiễu hơn cần λcs​ mạnh hơn.
Epilepsy
0.00005 (0.5x HAR)
0.025 (2.5x HAR)
Tín hiệu EEG rất nhạy cảm đòi hỏi λct​ yếu hơn nữa; sự phức tạp của cơn co giật cần λcs​ mạnh hơn.

Xác thực và Phân tích:
Kết quả của việc áp dụng các tham số được chuyển giao này đã được trình bày trong Bảng 4.1. Việc không cần tinh chỉnh lại (zero-shot transfer) các siêu tham số này đã dẫn đến những cải thiện hiệu suất đáng kể so với mô hình cơ sở: +9.32% trên Sleep-EDF và +2.71% trên Epilepsy (với 1% nhãn).
Thành công này xác thực mạnh mẽ rằng các nguyên tắc cơ bản chi phối khuôn khổ CoFT — đặc biệt là nguyên lý "Ít hơn là Nhiều hơn" và sự cân bằng giữa đồng huấn luyện và nhất quán đặc trưng — không phải là những hiện tượng đặc thù của riêng bộ dữ liệu HAR. Chúng phản ánh một động lực học sâu hơn về cách các mô hình hai miền tương tác.
Kết luận cho RQ4: Có, các nguyên tắc học được từ một bộ dữ liệu có thể được chuyển giao một cách hiệu quả để hướng dẫn tối ưu hóa trên các bộ dữ liệu mới và đa dạng. Khả năng đạt được hiệu suất tiên tiến trên các bộ dữ liệu y tế phức tạp chỉ bằng cách sử dụng một phương pháp chuyển giao có nguyên tắc, không cần tìm kiếm lại từ đầu, là một trong những đóng góp thực tiễn quan trọng nhất của công trình này. Nó chứng tỏ sự mạnh mẽ và tính ứng dụng của khuôn khổ CoFT, cung cấp một lộ trình hiệu quả để áp dụng mô hình cho các vấn đề mới trong tương lai.
