# Quick Answers - Your Specific Questions

**Date:** 2025-06-24  
**Requestor:** User - CoFT Training Configuration  

## ❓ Your 2 Questions & Solutions

### 🎯 **Question 1: Làm sao để test trên 5% thay vì 1% label?**

**✅ Answer:** Thêm `--label_percentage 5` vào command của bạn

**Your Original Command:**
```bash
python3 main.py --training_mode full_run --selected_dataset HAR --seed 42 --experiment_description "infots_test" --run_description "infots_enabled"
```

**Updated Command with 5% labels:**
```bash
python3 main.py --training_mode full_run --selected_dataset HAR --seed 42 --experiment_description "infots_test" --run_description "infots_enabled" --label_percentage 5
```

**🔧 What happens:**
- Pipeline tự động thay đổi: `train_linear_1p` → `train_linear_5p`
- Sử dụng file `train_5perc.pt` thay vì `train_1perc.pt`
- Accuracy sẽ cao hơn nhưng training time lâu hơn

---

### 🎯 **Question 2: Làm sao để dùng InfoTS cho các dataset khác chỉ bằng lệnh run tương tự?**

**✅ Answer:** Thêm `--enable_infots` cho bất kỳ dataset nào

**Sleep Dataset:**
```bash
python3 main.py --training_mode full_run --selected_dataset sleep --seed 42 --experiment_description "infots_test" --run_description "infots_enabled" --enable_infots
```

**Epilepsy Dataset:**
```bash
python3 main.py --training_mode full_run --selected_dataset Epilepsy --seed 42 --experiment_description "infots_test" --run_description "infots_enabled" --enable_infots
```

**pFD Dataset:**
```bash
python3 main.py --training_mode full_run --selected_dataset pFD --seed 42 --experiment_description "infots_test" --run_description "infots_enabled" --enable_infots
```

**🔧 What happens:**
- Override config file setting (bất kể dataset config có InfoTS = False)
- Hiển thị: `🎨 InfoTS augmentation ENABLED via command line for [dataset] dataset`
- Nếu InfoTS failed → Tự động fallback về CoFT augmentation

---

## 🔥 **Combined: Both Features Together**

```bash
# HAR with 5% + InfoTS
python3 main.py --training_mode full_run --selected_dataset HAR --seed 42 --experiment_description "infots_test" --run_description "infots_enabled" --label_percentage 5 --enable_infots

# Sleep with 5% + InfoTS
python3 main.py --training_mode full_run --selected_dataset sleep --seed 42 --experiment_description "infots_test" --run_description "infots_enabled" --label_percentage 5 --enable_infots
```

---

## 📊 **Current Status Verification**

**Tested and Working:** ✅ All features validated

1. **Label percentage validation:**
   ```
   📊 Label Percentage: 5%
   📋 Pipeline: self_supervised → train_linear_5p → ft_5p → gen_pseudo_labels → SupCon → train_linear_SupCon_5p
   ```

2. **InfoTS override validation:**
   ```
   🎨 InfoTS augmentation ENABLED via command line for sleep dataset
   🎨 InfoTS: Enabled
   ```

3. **Error validation:**
   ```bash
   python3 main.py --label_percentage 10  # ❌ Invalid - shows error message
   ```

---

## ⚡ **Quick Copy-Paste Commands**

```bash
# Your exact needs - ready to use:

# 1. HAR với 5% labels (thay vì 1%)
python3 main.py --training_mode full_run --selected_dataset HAR --seed 42 --experiment_description "infots_test" --run_description "infots_enabled" --label_percentage 5

# 2. Sleep với InfoTS enabled
python3 main.py --training_mode full_run --selected_dataset sleep --seed 42 --experiment_description "infots_test" --run_description "infots_enabled" --enable_infots

# 3. Epilepsy với InfoTS enabled  
python3 main.py --training_mode full_run --selected_dataset Epilepsy --seed 42 --experiment_description "infots_test" --run_description "infots_enabled" --enable_infots

# 4. pFD với InfoTS enabled
python3 main.py --training_mode full_run --selected_dataset pFD --seed 42 --experiment_description "infots_test" --run_description "infots_enabled" --enable_infots
```

---

**Implementation Status:** ✅ Complete and Ready to Use 