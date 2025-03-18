# Thyroid Disease Classification using ResNet50 

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Assuming dataset directory structure:
# dataset_thyroid/
#     train/
#         normal/
#         Benign/
#         Malignant/
#     test/
#         normal/
#         Benign/
#         Malignant/

# 1. Load Numerical Data (CSV)
numerical_data = pd.read_csv('./numerical_data.csv')  # Example CSV with columns: ['TSH', 'T3', 'T4']
print("Sample Numerical Data:")
print(numerical_data.head())

# 2. Image Preprocessing and Augmentation
data_dir = '.'  # root path

def preprocess_images(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    img = img / 255.0
    return img

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir + '/dataset_thyroid/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Multi-class!
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir + '/dataset_thyroid/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Multi-class!
    subset='validation'
)

# 3. Define Image-Only CNN Model
image_input = tf.keras.Input(shape=(224, 224, 3))
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)

for layer in base_model.layers:
    layer.trainable = False  # Freeze ResNet50 base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # Multi-class output!

model = Model(inputs=image_input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train CNN Model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
print(f"Train samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

history = model.fit(train_generator, validation_data=val_generator, epochs=15, callbacks=[early_stop])

# 5. Save CNN Model
model.save('resnet50_image_model.h5')

# 6. Optional: Plot Loss & Accuracy
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()
