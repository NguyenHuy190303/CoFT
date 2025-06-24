# New Features Usage Guide

**Date:** 2025-06-24  
**Author:** Leo  
**Features:** Label Percentage Configuration + InfoTS Command Line Control

## 🚀 Overview

Đã thêm 2 tính năng mới vào CoFT để tăng tính linh hoạt:

1. **Label Percentage Control**: Điều chỉnh tỷ lệ label training (1%, 5%, 75%)
2. **InfoTS Command Line**: Enable InfoTS cho bất kỳ dataset nào qua command line

## 📊 1. Label Percentage Configuration

### ✅ Sử dụng

```bash
# 1% Labels (Default)
python3 main.py --training_mode full_run --selected_dataset HAR --label_percentage 1

# 5% Labels  
python3 main.py --training_mode full_run --selected_dataset HAR --label_percentage 5

# 75% Labels (Full dataset)
python3 main.py --training_mode full_run --selected_dataset HAR --label_percentage 75
```

### 🔧 Chi tiết kỹ thuật

- **Valid values**: `1, 5, 75` (tương ứng với train_1perc.pt, train_5perc.pt, train_75perc.pt)
- **Pipeline update**: Tự động update tất cả training modes 
  - `train_linear_1p` → `train_linear_5p`
  - `ft_1p` → `ft_5p` 
  - `train_linear_SupCon_1p` → `train_linear_SupCon_5p`
- **Model loading**: Tự động load correct checkpoint paths

## 🎨 2. InfoTS Command Line Control

### ✅ Sử dụng

```bash
# Enable InfoTS cho HAR dataset
python3 main.py --training_mode full_run --selected_dataset HAR --enable_infots

# Enable InfoTS cho Sleep dataset  
python3 main.py --training_mode full_run --selected_dataset sleep --enable_infots

# Enable InfoTS cho Epilepsy dataset
python3 main.py --training_mode full_run --selected_dataset Epilepsy --enable_infots

# Enable InfoTS cho pFD dataset
python3 main.py --training_mode full_run --selected_dataset pFD --enable_infots
```

### 🔧 Chi tiết kỹ thuật

- **Override behavior**: Command line flag `--enable_infots` sẽ override config file setting
- **Default behavior**: Nếu không specify `--enable_infots`, sử dụng setting từ config file
- **Status display**: Hiển thị InfoTS status trong pipeline summary

## 🔥 3. Combined Usage Examples

### Trả lời câu hỏi của user:

**Câu hỏi 1: Test trên 5% thay vì 1% label**
```bash
python3 main.py --training_mode full_run --selected_dataset HAR --seed 42 --experiment_description "infots_test" --run_description "infots_enabled" --label_percentage 5
```

**Câu hỏi 2: Dùng InfoTS cho các dataset khác**
```bash
# Sleep dataset với InfoTS
python3 main.py --training_mode full_run --selected_dataset sleep --seed 42 --experiment_description "infots_test" --run_description "infots_enabled" --enable_infots

# Epilepsy dataset với InfoTS
python3 main.py --training_mode full_run --selected_dataset Epilepsy --seed 42 --experiment_description "infots_test" --run_description "infots_enabled" --enable_infots

# pFD dataset với InfoTS  
python3 main.py --training_mode full_run --selected_dataset pFD --seed 42 --experiment_description "infots_test" --run_description "infots_enabled" --enable_infots
```

### Advanced combinations:

```bash
# CoFT + InfoTS + 5% labels cho tất cả datasets
python3 main.py --training_mode full_run --selected_dataset HAR --enable_coft --enable_infots --label_percentage 5
python3 main.py --training_mode full_run --selected_dataset sleep --enable_coft --enable_infots --label_percentage 5
python3 main.py --training_mode full_run --selected_dataset Epilepsy --enable_coft --enable_infots --label_percentage 5
python3 main.py --training_mode full_run --selected_dataset pFD --enable_coft --enable_infots --label_percentage 5
```

## 📋 4. Quick Reference

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--label_percentage` | int | 1 | Label percentage: 1, 5, or 75 |
| `--enable_infots` | flag | False | Enable InfoTS for ANY dataset |
| `--enable_coft` | flag | False | Enable CoFT co-training |
| `--selected_dataset` | str | HAR | Dataset: HAR, sleep, Epilepsy, pFD |

### Status Display

Pipeline sẽ hiển thị thông tin configuration:
```
🎯 Starting Full Training Pipeline
📋 Pipeline: self_supervised → train_linear_5p → ft_5p → gen_pseudo_labels → SupCon → train_linear_SupCon_5p
🗂️ Dataset: HAR
📊 Label Percentage: 5%
🔄 CoFT: Enabled
🎨 InfoTS: Enabled
⏰ Start Time: 2025-06-24 16:38:55
```

## ⚡ 5. Quick Examples Script

Chạy script để xem tất cả examples:
```bash
./quick_examples.sh
```

## 🔍 6. Validation & Error Handling

### Label Percentage Validation
```bash
# ❌ Invalid percentage
python3 main.py --label_percentage 10
# Error: Invalid label percentage 10%
# Valid options: [1, 5, 75]
```

### InfoTS Fallback
- Nếu InfoTS không available → Tự động fallback về CoFT augmentation
- Hiển thị warning và tiếp tục training

## 📈 7. Expected Performance Impact

### Label Percentage
- **1%**: Fastest training, lower accuracy
- **5%**: Balanced training time vs accuracy  
- **75%**: Longest training, highest accuracy

### InfoTS vs CoFT Augmentation
- **InfoTS**: Advanced learned augmentations, potentially better accuracy
- **CoFT**: Proven baseline augmentations, reliable results

## 🎯 8. Best Practices

1. **Start with 5%** labels for good balance of speed vs accuracy
2. **Always enable InfoTS** for potential accuracy boost
3. **Combine with CoFT** for maximum performance:
   ```bash
   python3 main.py --training_mode full_run --selected_dataset HAR --enable_coft --enable_infots --label_percentage 5
   ```
4. **Use 75%** labels only for final production runs
5. **Monitor resource usage** with higher percentages

## 📚 9. Implementation Details

### Pipeline Updates
- Training modes now dynamically generated based on `--label_percentage`
- Model loading paths automatically adjusted
- Checkpoint compatibility maintained

### Config Override Logic
```python
# InfoTS override in execute_training_mode()
if args.enable_infots:
    configs.augmentation.use_infots_augmentation = True
    print(f"🎨 InfoTS augmentation ENABLED via command line for {data_type} dataset")
```

### Backward Compatibility
- ✅ Existing commands work unchanged (default 1% + config InfoTS settings)
- ✅ All existing features preserved
- ✅ No breaking changes

---

**Status:** ✅ Complete  
**Tested:** Command line validation, pipeline generation, config override  
**Ready for:** Production use 