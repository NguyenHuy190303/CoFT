# CA-TCC Few-Shot Data Generation với 5 Random Seeds

## 📋 Tổng quan

Document này mô tả chi tiết phương pháp tạo few-shot training data với **5 random seeds** cho CA-TCC, kết hợp ưu điểm của phương pháp tác giả gốc và cải tiến hiện đại.

## 🎯 Mục tiêu

1. **Tái tạo methodology** theo bài báo gốc CA-TCC/TS-TCC
2. **Đảm bảo reproducibility** với 5 random seeds khác nhau  
3. **Cải thiện robustness** của kết quả few-shot learning
4. **Tương thích hoàn toàn** với pipeline CA-TCC hiện có

## 📖 Phương pháp Tác giả Gốc (Từ bài báo)

> "chúng tôi điều tra hiệu suất TS-TCC của chúng tôi dưới các phần khác nhau của một vài nhãn, bằng cách tinh chỉnh mô hình đã được huấn luyện trước bằng cách sử dụng 1%, 5%, 10%, 50% và 75% các mẫu được chọn ngẫu nhiên của dữ liệu huấn luyện."

> "đối với các bộ dữ liệu UCR, họ đảm bảo rằng việc lựa chọn mẫu cho phép tất cả các lớp đều có mặt trong tập huấn luyện, ngay cả với chỉ 1% số mẫu. Toàn bộ các thử nghiệm được lặp lại 5 lần với 5 "seed" (hạt giống ngẫu nhiên) khác nhau để đảm bảo tính ổn định của kết quả."

### Đặc điểm phương pháp gốc:
- ✅ **Random sampling** (không cân bằng classes)
- ✅ **Ensure all classes present** (minimum requirement)
- ✅ **5 random seeds** [0, 1, 2, 3, 4]
- ✅ **Repeatability** để test statistical significance

## 🚀 Phương Pháp Cải Tiến (Implementation hiện tại)

### **Algorithm: Random Sampling with Class Guarantee**

```python
def random_sample_with_class_guarantee(X, y, percentage, random_seed):
    """
    Phương pháp cải tiến: Random + Class Guarantee + Multiple Attempts
    """
    
    # 1. Setup
    unique_classes = np.unique(y)
    n_samples = len(y)
    target_size = max(len(unique_classes), int(n_samples * percentage / 100.0))
    
    # 2. Multiple Attempts Loop  
    for attempt in range(max_attempts=50):
        # 2.1 Random Sampling
        _, X_sample, _, y_sample = train_test_split(
            X, y,
            test_size=target_size,
            random_state=random_seed + attempt,  # Vary seed each attempt
            shuffle=True                         # Pure random (NO stratify)
        )
        
        # 2.2 Check Class Coverage
        if ensure_all_classes_present(y_sample, y):
            return X_sample, y_sample  # SUCCESS!
            
        # 2.3 Adaptive Target Size
        target_size = min(target_size + len(unique_classes), n_samples)
    
    # 3. Fallback Strategy
    if still_failed:
        # Stratified backup để đảm bảo có data
        return stratified_split(X, y, percentage, random_seed)
```

### **Key Improvements:**

1. **🎲 Pure Random Sampling**
   - Sử dụng `shuffle=True`, `stratify=None`  
   - Giống phương pháp tác giả gốc

2. **🔒 Class Guarantee Mechanism**
   - Đảm bảo 100% classes có mặt
   - Multiple attempts với varied seeds

3. **📈 Adaptive Target Size**
   - Tự động tăng sample size nếu thiếu classes
   - Minimum = số classes, Maximum = full dataset

4. **🛡️ Robust Fallback**
   - Stratified backup nếu random fails
   - Đảm bảo luôn có valid output

5. **🔄 5 Seeds Implementation**
   - Seeds = [0, 1, 2, 3, 4] như tác giả
   - File naming: `train_{percentage}perc_seed{seed}.pt`

## 📊 Kết quả Implementation

### **✅ Datasets đã tạo thành công:**

#### **Epilepsy Dataset:**
```
data/epilepsy/5seeds/
├── train_1perc_seed0.pt   (73 samples)
├── train_1perc_seed1.pt   (73 samples) 
├── train_1perc_seed2.pt   (73 samples)
├── train_1perc_seed3.pt   (73 samples)
├── train_1perc_seed4.pt   (73 samples)
├── train_5perc_seed0.pt   (368 samples)
├── train_5perc_seed1.pt   (368 samples)
├── train_5perc_seed2.pt   (368 samples)
├── train_5perc_seed3.pt   (368 samples)
├── train_5perc_seed4.pt   (368 samples)
└── ... (10%, 50%, 75% tương tự)
```

#### **HAR Dataset:**
```
data/HAR/5seeds/
├── train_1perc_seed0.pt   (58 samples, 6 classes)
├── train_1perc_seed1.pt   (58 samples, 6 classes)
├── train_1perc_seed2.pt   (58 samples, 6 classes)
├── train_1perc_seed3.pt   (58 samples, 6 classes)
├── train_1perc_seed4.pt   (58 samples, 6 classes)
└── ... (5%, 10%, 50%, 75%)
```

#### **SleepEDF Dataset:**
```
data/SleepEDF/5seeds/
├── train_1perc_seed0.pt   (256 samples, 5 classes)
├── train_1perc_seed1.pt   (256 samples, 5 classes)
├── train_1perc_seed2.pt   (256 samples, 5 classes)
├── train_1perc_seed3.pt   (256 samples, 5 classes)
├── train_1perc_seed4.pt   (256 samples, 5 classes)
└── ... (5%, 10%, 50%, 75%)
```

## 🔍 Phân tích Class Distribution

### **Epilepsy 1% - Seed Variation:**
```
Seed 0: Class 0: 19/1456 (1.3%), Class 1: 54/5904 (0.9%)
Seed 1: Class 0: 15/1456 (1.0%), Class 1: 58/5904 (1.0%)  
Seed 2: Class 0: 15/1456 (1.0%), Class 1: 58/5904 (1.0%)
Seed 3: Class 0: 18/1456 (1.2%), Class 1: 55/5904 (0.9%)
Seed 4: Class 0: 17/1456 (1.2%), Class 1: 56/5904 (0.9%)

📈 Variation: Class 0: [15-19], Class 1: [54-58]
📊 Random distribution như expected!
```

### **HAR 1% - Cross-Class Balance:**
```
Seed 0: [11, 9, 14, 5, 7, 12] samples per class
Seed 1: [6, 10, 8, 11, 7, 16] samples per class  
Seed 2: [7, 8, 7, 12, 9, 15] samples per class
Seed 3: [11, 6, 7, 10, 5, 19] samples per class
Seed 4: [10, 8, 7, 15, 11, 7] samples per class

📈 Variation: Natural random imbalance
🎯 All 6 classes present in every seed!
```

## ⚖️ So sánh với Phương pháp Trước đây

| Aspect | Phương pháp Stratified (trước) | Phương pháp 5-Seeds (hiện tại) |
|--------|--------------------------------|--------------------------------|
| **Sampling** | Stratified (perfect balance) | Random (natural imbalance) |
| **Class guarantee** | Perfect proportional | Minimum presence guarantee |
| **Seeds** | Single seed (42) | 5 seeds [0,1,2,3,4] |
| **Reproducibility** | High (fixed) | Statistical (5 runs) |
| **Paper alignment** | Methodology different | Methodology aligned |
| **Scientific robustness** | High quality, không theo paper | Lower quality per run, theo paper |

### **Ví dụ cụ thể - Epilepsy 1%:**

**Stratified Method (cũ):**
```
74 samples: Class 0: 15/1456 (1.0%), Class 1: 59/5904 (1.0%)
→ Perfect 1% từ mỗi class
```

**5-Seeds Random Method (mới):**
```
Seed 0: 73 samples: Class 0: 19 (1.3%), Class 1: 54 (0.9%)
Seed 1: 73 samples: Class 0: 15 (1.0%), Class 1: 58 (1.0%) 
Seed 2: 73 samples: Class 0: 15 (1.0%), Class 1: 58 (1.0%)
Seed 3: 73 samples: Class 0: 18 (1.2%), Class 1: 55 (0.9%)
Seed 4: 73 samples: Class 0: 17 (1.2%), Class 1: 56 (0.9%)

→ Natural variation, all classes present
```

## 🎯 Lợi ích của Phương pháp 5-Seeds

### **1. Paper Alignment ✅**
- Tuân thủ chính xác methodology trong bài báo
- Random sampling như tác giả mô tả
- 5 seeds để statistical testing

### **2. Scientific Robustness 📊**
- Statistical significance testing
- Mean ± Standard deviation reporting
- Confidence intervals
- Outlier detection

### **3. Research Reproducibility 🔄**
- Fixed seeds [0,1,2,3,4] như paper
- Deterministic results
- Cross-lab reproducible

### **4. Real-world Simulation 🌍**
- Natural class imbalance (như real data)
- Không artificial perfect balance
- More realistic few-shot scenarios

### **5. CA-TCC Pipeline Compatible 🔧**
- Tương thích 100% với dataloader
- Metadata tracking
- Easy switching between seeds

## 🚀 Usage trong CA-TCC

### **Training với Single Seed:**
```bash
# Sử dụng seed 0
python main.py --experiment_description "ft_1per_seed0" --dataset "epilepsy"

# Dataloader sẽ load: data/epilepsy/5seeds/train_1perc_seed0.pt
```

### **Statistical Experiment (5 Seeds):**
```bash
# Chạy experiment với 5 seeds
for seed in 0 1 2 3 4; do
    python main.py --experiment_description "ft_1per_seed${seed}" --dataset "epilepsy"
done

# Analysis: Mean ± Std across 5 runs
```

### **Comparison Study:**
```bash
# So sánh phương pháp cũ vs mới
python main.py --experiment_description "ft_1per" --dataset "epilepsy"        # Stratified
python main.py --experiment_description "ft_1per_seed0" --dataset "epilepsy"  # Random seed 0
```

## 📈 Expected Research Outcomes

### **1. Statistical Reporting:**
```
Few-shot Learning Results (1% labeled data):
- Stratified method: 85.2% accuracy
- 5-seeds random method: 82.1 ± 2.3% accuracy (n=5)
  └ Seeds: [80.1%, 83.5%, 81.9%, 84.2%, 80.8%]
```

### **2. Method Comparison:**
- **Stratified**: Higher individual performance, không theo paper
- **5-Seeds**: Lower nhưng có statistical confidence, theo paper

### **3. Paper Submission:**
- Có thể cite methodology chính xác
- Statistical significance testing
- Comparable với other papers sử dụng cùng approach

## 🔧 Technical Implementation Details

### **File Structure:**
```
data/
├── epilepsy/
│   ├── train.pt              # Original full training data
│   ├── train_1perc.pt        # Stratified 1% (compatibility)  
│   └── 5seeds/
│       ├── train_1perc_seed0.pt   # Random 1% seed 0
│       ├── train_1perc_seed1.pt   # Random 1% seed 1
│       ├── train_1perc_seed2.pt   # Random 1% seed 2
│       ├── train_1perc_seed3.pt   # Random 1% seed 3
│       ├── train_1perc_seed4.pt   # Random 1% seed 4
│       └── stats_1perc_5seeds.json # Statistics
├── HAR/5seeds/...
└── SleepEDF/5seeds/...
```

### **Metadata trong .pt files:**
```python
{
    "samples": torch.Tensor,     # Training samples
    "labels": torch.Tensor,      # Training labels  
    "metadata": {
        "seed": 0,               # Random seed used
        "percentage": 1,         # Percentage of data
        "sampling_method": "random_with_class_guarantee",
        "original_samples": 7360,
        "selected_samples": 73
    }
}
```

### **Statistics Tracking:**
```json
{
    "dataset": "epilepsy",
    "percentage": 1,
    "full_samples": 7360,
    "full_classes": 2,
    "seeds_results": {
        "0": {
            "samples": 73,
            "actual_percentage": 1.0,
            "classes_present": 2,
            "all_classes_present": true,
            "class_distribution": {"0": 19, "1": 54}
        }
    }
}
```

## 🎯 Khuyến nghị Sử dụng

### **Cho Research Reproducibility:**
✅ **Sử dụng 5-seeds method** để:
- Tuân thủ paper methodology
- Statistical significance testing  
- Cross-lab reproducible results

### **Cho Quick Experiments:**
✅ **Có thể dùng stratified method** để:
- Faster iteration
- Higher individual performance
- Development và debugging

### **Cho Paper Submission:**
✅ **Bắt buộc 5-seeds method** để:
- Cite methodology chính xác
- Statistical confidence
- Reviewer acceptance

## 📋 Kết luận

Phương pháp **5 Random Seeds** đã được implement thành công với:

1. ✅ **Alignment hoàn toàn** với tác giả gốc
2. ✅ **Cải tiến robustness** với class guarantee  
3. ✅ **Tương thích 100%** với CA-TCC pipeline
4. ✅ **Statistical rigor** với 5 independent runs
5. ✅ **Reproducible** với fixed seeds

**Trade-off được chấp nhận:**
- Individual performance có thể thấp hơn stratified
- Nhưng có statistical confidence và paper compliance

**Phương pháp này là OPTIMAL cho research publication trong lĩnh vực Few-shot Time Series Learning.** 🚀 