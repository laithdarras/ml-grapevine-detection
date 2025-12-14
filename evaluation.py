import tensorflow as tf
import pathlib
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score

# reduce load on cpu
batch_size = 16
img_height = 128
img_width = 128

test_dir = pathlib.Path("data/test")

def labels_from_dir(dir_path):
    paths = []
    labels = []
    for img_path in sorted(dir_path.glob("*.jpg")):
        filename = img_path.name.lower()
        label = 1 if filename.startswith("esca_") else 0
        paths.append(str(img_path))
        labels.append(label)
    return paths, np.array(labels, dtype=np.int32)

# reading images in tf
test_images = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels=None,
    color_mode="rgb",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

# creating labels for images
_, test_labels = labels_from_dir(test_dir)

# loading saved trained model
model = tf.keras.models.load_model("grape_model.keras")

# compute loss
loss = model.evaluate(
    tf.data.Dataset.zip((
        test_images,
        tf.data.Dataset.from_tensor_slices(test_labels).batch(batch_size)
    )),
    verbose=0
)

# predictions
y_pred = np.argmax(model.predict(test_images, verbose=0), axis=1)

# metrics
precision = precision_score(test_labels, y_pred, zero_division=0)
recall = recall_score(test_labels, y_pred, zero_division=0)

print("============== Test Evaluation ==============")
print(f"loss:      {loss[0]:.4f}")
print(f"precision: {precision:.4f}")
print(f"recall:    {recall:.4f}")