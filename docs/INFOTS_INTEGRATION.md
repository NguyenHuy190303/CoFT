# InfoTS Integration trong CoFT

## 🎯 Tổng quan

CoFT giờ đây hỗ trợ tích hợp **InfoTS augmentations** như một tùy chọn để thay thế cho augmentations mặc định.

- **Mặc định**: Sử dụng CoFT augmentations (weak/strong với scaling, jitter, permutation)
- **Tùy chọn**: Bật InfoTS augmentations với learnable AutoAUG

## 🚀 Cách sử dụng

### Sử dụng CoFT augmentations (mặc định)
```python
# Không cần làm gì - đây là behavior mặc định
config = HAR_Config()
# config.augmentation.use_infots_augmentation = False  # Mặc định
```

### Bật InfoTS augmentations
```python
# Bật InfoTS augmentations
config = HAR_Config()
config.augmentation.use_infots_augmentation = True

# Tùy chọn: điều chỉnh parameters
config.augmentation.infots_aug_p1 = 0.7     # Xác suất augmentation đầu tiên
config.augmentation.infots_aug_p2 = 0.0     # Xác suất augmentation thứ hai  
config.augmentation.infots_temperature = 1.0  # Temperature cho learnable weights
```

## 📊 So sánh

| Tính năng | CoFT (Mặc định) | InfoTS |
|-----------|----------------|--------|
| **Tốc độ** | ⚡ Nhanh | 🐌 Chậm hơn |
| **Bộ nhớ** | 💾 Thấp | 💾💾 Cao hơn |
| **Augmentation types** | 3 loại | 7+ loại |
| **Learning** | ❌ Cố định | ✅ Learnable |

## 🔧 Technical details

### CoFT Augmentations (mặc định)
- **Weak**: Scaling (ratio=2) + Light Jitter (0-0.1)
- **Strong**: Permutation + Heavy Jitter (0.1-1.0)

### InfoTS Augmentations
- **7+ types**: subsequence, cutout, jitter, scaling, time_warp, window_slice, window_warp
- **Learnable weights**: AutoAUG với meta-learning optimization
- **Temperature control**: Điều chỉnh độ "focused" của learnable weights

## 💡 Khi nào sử dụng

### Sử dụng CoFT (mặc định)
- Production deployments cần tốc độ
- Môi trường hạn chế tài nguyên
- Baseline experiments
- Training lớn scale

### Sử dụng InfoTS  
- Research experiments
- Khám phá augmentation strategies mới
- Môi trường GPU mạnh
- Datasets nhỏ cần augmentation phức tạp

## 🛡️ Fallback mechanism

Nếu InfoTS không available, hệ thống tự động fallback về CoFT augmentations:

```
⚠️  InfoTS augmentation failed: No module named 'InfoTS'
   Falling back to CoFT augmentations
```

## 📋 Implementation

### Trong DataTransform function:
```python
def DataTransform(sample, config):
    use_infots = getattr(config.augmentation, 'use_infots_augmentation', False)
    
    if use_infots and INFOTS_AVAILABLE:
        return _apply_infots_augmentation(sample, config)  # InfoTS
    else:
        return weak_augmentation(sample, config), strong_augmentation(sample, config)  # CoFT
```

### Backward compatibility
- **100% backward compatible**: Code hiện tại hoạt động không thay đổi
- **Zero breaking changes**: Không cần modify code cũ
- **Default behavior unchanged**: Mặc định vẫn là CoFT augmentations

---

**Kết luận**: Integration đơn giản này cho phép dễ dàng thử nghiệm với InfoTS augmentations trong khi vẫn duy trì tính ổn định và hiệu suất của CoFT baseline. 