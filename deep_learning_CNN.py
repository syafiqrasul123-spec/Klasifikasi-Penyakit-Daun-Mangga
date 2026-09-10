import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import ResNet50

print("===== MEMULAI TRAINING CNN FAMILY =====")

# ==================================
# LOAD DATASET
# ==================================

img_size = (128,128)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names

print("\nClass :")
print(class_names)

# ==================================
# NORMALISASI
# ==================================

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(
    lambda x,y:(normalization_layer(x),y)
)

val_ds = val_ds.map(
    lambda x,y:(normalization_layer(x),y)
)

# ==================================
# CNN SEDERHANA
# ==================================

print("\n===== MODEL CNN =====")

cnn_model = Sequential([

    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,128,3)
    ),

    MaxPooling2D(),

    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D(),

    Conv2D(
        128,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D(),

    Flatten(),

    Dense(
        128,
        activation='relu'
    ),

    Dropout(0.3),

    Dense(
        1,
        activation='sigmoid'
    )
])

cnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_cnn = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

loss_cnn, acc_cnn = cnn_model.evaluate(val_ds)

print("\nAccuracy CNN :", acc_cnn)

# ==================================
# MOBILENETV2
# ==================================

print("\n===== MODEL MobileNetV2 =====")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128,128,3)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(128,128,3))

x = base_model(inputs, training=False)

x = GlobalAveragePooling2D()(x)

x = Dense(128, activation='relu')(x)

outputs = Dense(1, activation='sigmoid')(x)

mobilenet_model = Model(inputs, outputs)

mobilenet_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_mobile = mobilenet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

loss_mobile, acc_mobile = mobilenet_model.evaluate(val_ds)

print("\nAccuracy MobileNetV2 :", acc_mobile)

# ==================================
# RESNET50
# ==================================

print("\n===== MODEL ResNet50 =====")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(128,128,3)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(128,128,3))

x = base_model(inputs, training=False)

x = GlobalAveragePooling2D()(x)

x = Dense(128, activation='relu')(x)

outputs = Dense(1, activation='sigmoid')(x)

resnet_model = Model(inputs, outputs)

resnet_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_resnet = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

loss_resnet, acc_resnet = resnet_model.evaluate(val_ds)

print("\nAccuracy ResNet50 :", acc_resnet)

# ==================================
# GRAFIK AKURASI CNN
# ==================================

plt.figure(figsize=(8,5))

plt.plot(
    history_cnn.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history_cnn.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# ==================================
# GRAFIK LOSS CNN
# ==================================

plt.figure(figsize=(8,5))

plt.plot(
    history_cnn.history['loss'],
    label='Train Loss'
)

plt.plot(
    history_cnn.history['val_loss'],
    label='Validation Loss'
)

plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

# ==================================
# PERBANDINGAN MODEL
# ==================================

model_names = [
    "CNN",
    "MobileNetV2",
    "ResNet50"
]

accuracies = [
    acc_cnn,
    acc_mobile,
    acc_resnet
]

plt.figure(figsize=(8,5))

plt.bar(
    model_names,
    accuracies
)

plt.title("Perbandingan Akurasi CNN Family")
plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.ylim(0.1,1.0)

plt.show()

print("\n===== HASIL AKHIR =====")

print("CNN         :", acc_cnn)
print("MobileNetV2 :", acc_mobile)
print("ResNet50    :", acc_resnet)

print("\n===== SELESAI =====")