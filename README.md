# finalhw
# 🚗 Vehicle Classifier using Transfer Learning

This project is a **vehicle image classification model** built using **TensorFlow** and **Transfer Learning** with `MobileNetV2`. It identifies vehicle types (e.g., cars, bikes, trucks, etc.) from images using a custom dataset.

---

## 🧠 Model Summary

- ✅ Pretrained Base: `MobileNetV2` (from ImageNet)
- ✅ Top Layers: Custom Dense layers
- ✅ Loss Function: `SparseCategoricalCrossentropy`
- ✅ Accuracy Metric: `Accuracy`
- ✅ Optimizer: `Adam`

---

## 🗂 Dataset

- 📍 Location: `/Users/prason/Downloads/Dataset`
- 🖼️ Format: Images organized into folders per class
- 🔄 Splitting: 80% training, 20% validation
- 📏 Image Size: `224x224`

