import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Step 1: Find project and output folders
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

output_dir = script_dir


# Step 2: Class names
class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]


# Step 3: Find saved model
possible_paths = [
    os.path.join(output_dir, "cifar10_model.keras"),
    os.path.join(output_dir, "best_model.keras"),
    os.path.join(project_dir, "cifar10_model.keras"),
    os.path.join(project_dir, "best_model.keras")
]

model_path = None

for path in possible_paths:
    if os.path.isfile(path):
        model_path = path
        break

if model_path is None:
    raise FileNotFoundError(
        "Model file not found! Check cifar10_model.keras "
        "or best_model.keras in your project folders."
    )

print("Model found:", model_path)


# Step 4: Load CIFAR-10 test dataset
(_, _), (X_test, y_test) = cifar10.load_data()

print("Test Dataset Loaded Successfully!")


# Step 5: Normalize images
X_test = X_test.astype("float32") / 255.0


# Step 6: Load trained model
model = tf.keras.models.load_model(model_path)

print("Trained Model Loaded Successfully!")


# Step 7: Predict images
y_pred = model.predict(X_test, batch_size=64)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test.flatten()

print("Predictions Completed Successfully!")


# Step 8: Generate confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(10, 10))

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45
)

plt.title("CIFAR-10 Confusion Matrix")
plt.tight_layout()


# Step 9: Save confusion matrix
save_path = os.path.join(output_dir, "confusion_matrix.png")

plt.savefig(
    save_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("Confusion Matrix Saved Successfully!")
print("File Location:", save_path)