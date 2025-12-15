import tensorflow as tf
import PIL
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

#SOURCE OF THIS CODE: https://www.tensorflow.org/tutorials/images/classification

train_dir = pathlib.Path('data/train')
val_dir = pathlib.Path('data/val')
test_dir = pathlib.Path('data/test')

# image_count = len(list(data_dir.glob('*/*.jpg')))
image_count = len(list(train_dir.glob('*.jpg'))) + len(list(val_dir.glob('*.jpg'))) + len(list(test_dir.glob('*.jpg')))
print(image_count)

batch_size = 16
img_height = 128
img_width = 128

# images
train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  labels=None,
  color_mode='rgb',
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  val_dir,
  labels=None,
  color_mode='rgb',
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=False
)

# building labels from the filename prefixes since tf can't infer labels automatically 
def labels_from_dir(dir_path):
  paths = sorted([str(p) for p in dir_path.glob('*.jpg')])
  labels = []
  for p in paths:
    name = os.path.basename(p).lower()
    # binary classification healthy -> 0 and esca -> 1
    labels.append(1 if name.startswith("esca_") else 0)
  
  return paths, np.array(labels, dtype=np.int32)

train_paths, train_labels = labels_from_dir(train_dir)
val_paths, val_labels = labels_from_dir(val_dir)

print("train label counts (healthy, esca): ", np.bincount(train_labels, minlength=2))
print("val label counts (healthy, esca): ", np.bincount(val_labels, minlength=2))

class_names = ("healthy","esca")

# (images, labels)
train_labels_ds = tf.data.Dataset.from_tensor_slices(train_labels).batch(batch_size)
train_labeled_ds = tf.data.Dataset.zip((train_ds, train_labels_ds))
val_labels_ds = tf.data.Dataset.from_tensor_slices(val_labels).batch(batch_size)
val_labeled_ds = tf.data.Dataset.zip((val_ds, val_labels_ds))

# plt.figure(figsize=(10,10))
# for images, labels in train_labeled_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i+1)
#     plt.imshow(images[i].numpy().astype('uint8'))
#     plt.title(class_names[int(labels[i].numpy())])
#     plt.axis("off")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_labeled_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_labeled_ds.prefetch(buffer_size=AUTOTUNE)

#rescale rgb to [0,1] to decrease range of values
normalization_layer = layers.Rescaling(1./255)

#apply this to our dataset
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


num_classes = len(class_names)

#create our model
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation='softmax')
])

#compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

#view summary
model.summary()

epochs=5
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# loss plot
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()

# accuracy plot
plt.figure()
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve.png")
plt.close()

print("Saved plots: loss_curve.png, accuracy_curve.png")


# saving the training model to gather metrics
model.save("grape_model.keras")
print("saved model to grape_model.keras")