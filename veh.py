import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Streamlit app title
st.title("🚗 Vehicle Image Classifier with MobileNetV2")

# Set your dataset path
data_dir = "//Users/prason/Downloads/Dataset"
image_size = (224, 224)
batch_size = 32

# Load the dataset
@st.cache_resource
def load_data():
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )
    return train_ds, val_ds, train_ds.class_names

train_ds, val_ds, class_names = load_data()
st.success(f"✅ Classes loaded: {class_names}")

# Model architecture
@st.cache_resource
def build_model():
    base_model = MobileNetV2(input_shape=image_size + (3,),
                             include_top=False,
                             weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()

# Training
if st.button("👨‍🏫 Train Model (Takes time)"):
    history = model.fit(train_ds, validation_data=val_ds, epochs=10)
    model.save("vehicle_classifier_model.keras")
    st.success("Model trained and saved!")

# Upload image for prediction
uploaded_file = st.file_uploader("📸 Upload an image to classify", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img_resized = img.resize(image_size)
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.image(img, caption=f"Predicted: {predicted_class}", use_column_width=True)
    st.write(f"Prediction Confidence: {np.max(prediction) * 100:.2f}%")

# Visualize random predictions from training set
if st.button("🔍 Show Sample Predictions"):
    for images, labels in train_ds.take(1):
        preds = model.predict(images)
        predicted_labels = tf.argmax(preds, axis=1)

        st.write("### Sample Predictions")
        for i in range(5):
            true_label = class_names[labels[i].numpy()]
            pred_label = class_names[predicted_labels[i].numpy()]
            st.image(images[i].numpy().astype("uint8"), caption=f"True: {true_label} | Pred: {pred_label}", width=200)
