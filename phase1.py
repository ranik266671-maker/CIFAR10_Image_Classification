# Step 1: Import Libraries

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.datasets import cifar10


# Create Output Folder

output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs"
)

os.makedirs(output_dir, exist_ok=True)

# Step 2: Load CIFAR-10 Dataset

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Dataset Loaded Successfully!")

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Step 3: Define Class Names

class_names = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]


# Step 4: Display 10 Random Images

plt.figure(figsize=(15, 8), dpi=150)

random_indices = np.random.choice(
    len(X_train), 10, replace=False
)

for i, idx in enumerate(random_indices):

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        X_train[idx],
        interpolation='nearest'
    )

    label = int(y_train[idx][0])

    plt.title(
        class_names[label],
        fontsize=12,
        fontweight='bold'
    )

    plt.axis('off')

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "random_images.png"),
    dpi=200,
    bbox_inches='tight'
)

plt.show()

# Step 5: Data Preprocessing
# Normalization (0 to 1)

X_train = X_train.astype('float32') / 255.0

X_test = X_test.astype('float32') / 255.0

print("\nNormalization Completed!")

print("Minimum Pixel Value:", X_train.min())

print("Maximum Pixel Value:", X_train.max())
 

# Step 6: Data Augmentation

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.04),

    tf.keras.layers.RandomTranslation(
        height_factor=0.1,
        width_factor=0.1
    )

])

print("\nData Augmentation Created Successfully!")


# Step 7: Display Augmented Images

plt.figure(figsize=(15, 8), dpi=150)

sample_image = X_train[0]

for i in range(10):

    augmented_image = data_augmentation(
        tf.expand_dims(sample_image, 0),
        training=True
    )

    plt.subplot(2, 5, i + 1)

    plt.imshow(
        augmented_image[0].numpy(),
        interpolation='nearest'
    )

    plt.title(
        f"Augmented {i+1}",
        fontsize=11
    )

    plt.axis('off')

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "augmented_images.png"),
    dpi=200,
    bbox_inches='tight'
)

plt.show()


# Step 8: Final Output

print("\n====================================")

print("PHASE 1 COMPLETED SUCCESSFULLY!")


print("Dataset Loading: Completed")

print("Dataset Inspection: Completed")

print("Image Visualization: Completed")

print("Normalization: Completed")

print("Data Augmentation: Completed")

print("Augmented Images: Completed")

print("\nAll Output Images Saved Successfully!")

print("Output Folder:", output_dir)

print("====================================")

