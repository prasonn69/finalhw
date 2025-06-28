import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import os
data_dir = "//Users/prason/Downloads/Dataset"  

image_size = (224, 224)
batch_size = 32

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

class_names = train_ds.class_names
print("Classes:", class_names)
base_model = MobileNetV2(input_shape=image_size + (3,),
                         include_top=False,
                         weights='imagenet')

base_model.trainable = False  # Freeze the base
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

model.summary()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
model.save("vehicle_classifier_model.keras")  
import matplotlib.pyplot as plt
import numpy as np

# Take one batch from the dataset
for images, labels in train_ds.take(1):
    preds = model.predict(images)
    predicted_labels = tf.argmax(preds, axis=1)  # Get index of highest prob

    for i in range(5):
        plt.imshow(images[i].numpy().astype("uint8"))
        true_label = class_names[labels[i].numpy()]
        pred_label = class_names[predicted_labels[i].numpy()]
        plt.title(f"True: {true_label} | Pred: {pred_label}")
        plt.axis('off')
        plt.show()



