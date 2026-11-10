import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


# Create output folder
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(output_dir, exist_ok=True)


# Class names
class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]


# Step 1: Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Dataset Loaded Successfully!")


# Step 2: Normalize images
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


# Step 3: Data Augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.04),
    layers.RandomTranslation(0.1, 0.1)
])


# Step 4: Build CNN Model
model = models.Sequential([

    layers.Input(shape=(32, 32, 3)),

    data_augmentation,

    # First CNN Block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    # Second CNN Block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # Third CNN Block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    # Classification Layer
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(10, activation='softmax')
])


# Step 5: Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Display model architecture
model.summary()


# Step 6: Callbacks
checkpoint = ModelCheckpoint(
    os.path.join(output_dir, "best_model.keras"),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3
)


# Step 7: Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=64,
    validation_split=0.2,
    callbacks=[checkpoint, early_stop, reduce_lr]
)


# Step 8: Evaluate Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", test_accuracy * 100, "%")
print("Test Loss:", test_loss)


# Step 9: Plot Accuracy Graph
plt.figure(figsize=(8, 5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig(os.path.join(output_dir, "accuracy_graph.png"))
plt.show()


# Step 10: Plot Loss Graph
plt.figure(figsize=(8, 5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig(os.path.join(output_dir, "loss_graph.png"))
plt.show()


# Step 11: Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = y_test.flatten()



# Step 12: Confusion Matrix
print("Generating Confusion Matrix...")

print("Confusion Matrix Saved Successfully!")

cm = confusion_matrix(y_true, y_pred_classes)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(10, 10))

disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)

plt.title("CIFAR-10 Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "confusion_matrix.png"),
    dpi=200,
    bbox_inches='tight'
)

plt.show()
plt.close()