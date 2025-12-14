import tensorflow as tf
import pathlib
import numpy as np
import os

batch_size = 32
img_height = 180
img_width = 180

test_dir = pathlib.Path('data/test')

def labels_from_dir(dir_path):
    paths = []
    labels = []

    for img_path in sorted(dir_path.glob('*.jpg')):
        filename = img_path.name.lower()

        # healthy -> 0, esca -> 1
        if filename.startswith('esca_'):
            label = 1
        else:
            label = 0
        
        paths.append(str(img_path))
        labels.append(label)
    
    return paths, np.array(labels, dtype=np.int32)


# images only
test_images = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels=None,
    color_mode="rgb",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

# label extraction
train_paths, train_labels = labels_from_dir(train_dir)
val_paths, val_labels     = labels_from_dir(val_dir)
test_paths, test_labels   = labels_from_dir(test_dir)

test_labels_ds = tf.data.Dataset.from_tensor_slices(test_labels).batch(batch_size)
test_ds = tf.data.Dataset.zip((test_images, test_labels_ds)).prefetch(tf.data.AUTOTUNE)

# load the trained model
model = tf.keras.models.load_model("grape_model.keras")

# compile and evaluate
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
)

print("============== Test Evaluation ==============")
results = model.evaluate(test_ds, verbose=2)

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.2f}")