import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset Path
data_dir_train = pathlib.Path("../dataset/Train")

# Data Augmentation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_ds = datagen.flow_from_directory(data_dir_train, target_size=(180,180), batch_size=32, class_mode='categorical', subset="training")
val_ds = datagen.flow_from_directory(data_dir_train, target_size=(180,180), batch_size=32, class_mode='categorical', subset="validation")

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(180,180,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(len(train_ds.class_indices), activation='softmax')
])

# Compile & Train Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=25)

# Save Model
model.save("model.keras")
