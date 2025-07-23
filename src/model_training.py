import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import cv2
import numpy as np
import json
import random
from tensorflow.keras.utils import Sequence

# ==============================
# Data Generator with Multi-Scale Training
# ==============================

class TextMapGenerator(Sequence):
    def __init__(self, image_dir, annotation_dir, batch_size=4, min_size=320, max_size=1280, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.batch_size = batch_size
        self.min_size = min_size
        self.max_size = max_size
        self.shuffle = shuffle
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        target_size = random.randint(self.min_size, self.max_size)
        self.input_size = (target_size, int(target_size * 0.7))  # Maintain aspect ratio

        batch_filenames = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(batch_filenames)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_filenames)

    def __data_generation(self, batch_filenames):
        target_h, target_w = self.input_size
        batch_images = np.zeros((self.batch_size, target_h, target_w, 1), dtype=np.float32)

        # Ensure maps are downsampled by 8x to match model output
        output_h, output_w = target_h // 8, target_w // 8
        batch_text_maps = np.zeros((self.batch_size, output_h, output_w, 1), dtype=np.float32)
        batch_height_maps = np.zeros((self.batch_size, output_h, output_w, 1), dtype=np.float32)

        for i, filename in enumerate(batch_filenames):
            image_path = os.path.join(self.image_dir, filename)
            annotation_path = os.path.join(self.annotation_dir, filename.replace(".png", ".json"))

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (target_w, target_h)) / 255.0
            batch_images[i, ..., 0] = image

            with open(annotation_path, "r") as f:
                annotation_data = json.load(f)

            text_map, height_map = self.generate_maps(annotation_data, (target_h, target_w), filename)

            # Downsample the supervision maps to match model output
            text_map = cv2.resize(text_map, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
            height_map = cv2.resize(height_map, (output_w, output_h), interpolation=cv2.INTER_NEAREST)

            batch_text_maps[i, ..., 0] = text_map
            batch_height_maps[i, ..., 0] = height_map

        return batch_images, (batch_text_maps, batch_height_maps)

    def generate_maps(self, annotation_data, input_size, json_filename):
        text_map = np.zeros(input_size, dtype=np.uint8)
        height_map = np.zeros(input_size, dtype=np.float32)

        image_filename = json_filename.replace(".json", ".png")
        image_path = os.path.join(self.image_dir, image_filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_h, img_w = image.shape

        for item in annotation_data["form"]:
            if "box" in item:
                x1, y1, x2, y2 = item["box"]
                x1 = int(x1 * input_size[1] / img_w)
                y1 = int(y1 * input_size[0] / img_h)
                x2 = int(x2 * input_size[1] / img_w)
                y2 = int(y2 * input_size[0] / img_h)

                cv2.rectangle(text_map, (x1, y1), (x2, y2), 255, thickness=-1)
                text_height = y2 - y1
                cv2.rectangle(height_map, (x1, y1), (x2, y2), text_height, thickness=-1)

        text_map = text_map / 255.0
        height_map = height_map / np.max(height_map) if np.max(height_map) > 0 else height_map

        return text_map, height_map

# ==============================
# Fully Convolutional FPN Model
# ==============================

def build_fpn_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)  # Final downsampling by 8

    text_kernel_map = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid', name='text_kernel_map')(x)
    text_height_map = tf.keras.layers.Conv2D(1, (1,1), activation='linear', name='text_height_map')(x)

    model = tf.keras.Model(inputs, [text_kernel_map, text_height_map])
    return model

image_dir = "data/images"
annotation_dir = "data/annotations"

batch_size = 1
train_generator = TextMapGenerator(image_dir, annotation_dir, batch_size=batch_size, min_size=320, max_size=1280)

input_shape = (None, None, 1)
model = build_fpn_model(input_shape)

optimizer = Adam(learning_rate=0.001)
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

model.compile(
    optimizer=optimizer,
    loss={'text_kernel_map': custom_loss, 'text_height_map': custom_loss}
)

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    filepath="/content/drive/MyDrive/fpn_fcn_checkpoint_epoch_{epoch:02d}.h5",
    save_best_only=False,
    save_weights_only=False,
    verbose=1
)

import gc

tf.keras.backend.clear_session()
gc.collect()

# Train model
epochs = 50
model.fit(train_generator, epochs=epochs, callbacks=[checkpoint_callback])

# Save model
model.save("fpn_fcn_text_detection.h5")

this is the code for training the model
