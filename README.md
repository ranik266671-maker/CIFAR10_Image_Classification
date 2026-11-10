# CIFAR-10 Image Classification using Deep Learning

## 📌 Project Overview

This project focuses on classifying images into 10 different categories using Deep Learning and Convolutional Neural Networks (CNN).

The CIFAR-10 dataset contains 60,000 color images of size 32×32 pixels, divided into 10 classes.

The main objective is to build and train a CNN model that can recognize and classify different objects in images.

## 🎯 Dataset Classes

The project uses the following 10 classes:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- PIL (Pillow)

## 📂 Project Structure

```text
CIFAR10_Image_Classification/
│
├── phase1.py
├── phase2.py
├── phase3.py
├── confusion_matrix.py
├── accuracy_graph.png
├── loss_graph.png
├── augmented_images.png
├── random_images.png
├── confusion_matrix.png
├── best_model.keras
└── README.md
```

## 🚀 Project Phases

### Phase 1: Data Loading and Preprocessing

- Load the CIFAR-10 dataset.
- Display 10 random images with class labels.
- Normalize image pixel values.
- Apply data augmentation techniques.
- Perform random horizontal flips and rotations.

### Phase 2: CNN Model Training

- Build a Convolutional Neural Network.
- Use convolutional and max-pooling layers.
- Apply batch normalization and dropout.
- Train the model using the Adam optimizer.
- Plot training and validation accuracy.
- Plot training and validation loss.
- Generate a confusion matrix and classification report.
- Save the trained model.

### Phase 3: Image Prediction using Streamlit

- Create a web application using Streamlit.
- Upload an image.
- Resize and preprocess the image.
- Predict the image category.
- Display the predicted class and confidence score.
- Display prediction probabilities for all 10 classes.

## ⚙️ Installation

Install the required libraries:

```bash
pip install tensorflow numpy matplotlib scikit-learn streamlit pillow
```

## ▶️ How to Run

### Run Phase 1

```bash
python phase1.py
```

### Run Phase 2

```bash
python phase2.py
```

### Run Phase 3

```bash
python -m streamlit run phase3.py
```

Note: The trained model file must be available at the path specified in `phase3.py`.

## 📊 Project Outputs

The project generates:

- Random CIFAR-10 images
- Augmented images
- Training accuracy graph
- Training loss graph
- Confusion matrix
- Image classification predictions

## 🎓 Learning Outcomes

- Understanding image classification.
- Learning CNN architecture.
- Working with TensorFlow and Keras.
- Understanding image preprocessing and augmentation.
- Training and evaluating deep learning models.
- Building a simple image prediction web application using Streamlit.

## 👩‍💻 Author

**Rani Kumari**

B.Tech – Artificial Intelligence and Machine Learning

## 📌 Conclusion

This project demonstrates how Deep Learning and CNNs can be used for image classification. It covers data preprocessing, model training, evaluation, and real-time image prediction using Streamlit.
