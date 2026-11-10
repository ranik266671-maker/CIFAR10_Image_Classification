import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path


# ----------------------------------
# STEP 1: PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide"
)


# ----------------------------------
# STEP 2: CLASS NAMES
# ----------------------------------

class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]


# ----------------------------------
# STEP 3: MODEL FILE LOCATION
# ----------------------------------

project_dir = Path(__file__).resolve().parent

model_path = (
    project_dir
    / "outputs"
    / "outputs"
    / "best_model.keras"
)


# ----------------------------------
# STEP 4: LOAD TRAINED MODEL
# ----------------------------------

@st.cache_resource
def load_trained_model():

    return tf.keras.models.load_model(
        str(model_path),
        compile=False
    )


st.title("CIFAR-10 Image Classification")

st.write(
    "Upload an image and let the CNN model "
    "predict its class."
)


if not model_path.exists():

    st.error("Trained model not found!")

    st.write("Expected model location:")

    st.code(str(model_path))

    st.stop()


try:

    model = load_trained_model()

    st.success("Trained Model Loaded Successfully!")

except Exception as e:

    st.error(f"Model Loading Error: {e}")

    st.stop()


# ----------------------------------
# STEP 5: IMAGE UPLOAD
# ----------------------------------

st.subheader("Upload Your Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:

        # Open uploaded image
        original_image = Image.open(
            uploaded_file
        ).convert("RGB")


        # Display image
        st.image(
            original_image,
            caption="Uploaded Image",
            width=300
        )


        # ----------------------------------
        # STEP 6: IMAGE PREPROCESSING
        # ----------------------------------

        resized_image = original_image.resize(
            (32, 32)
        )

        image_array = np.array(
            resized_image,
            dtype=np.float32
        )

        # Normalize image
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        st.success("Image Preprocessing Completed!")


        # ----------------------------------
        # STEP 7: PREDICTION
        # ----------------------------------

        if st.button("Predict Image"):

            with st.spinner("Predicting Image..."):

                predictions = model.predict(
                    image_array,
                    verbose=0
                )


            # Get predicted class
            predicted_index = int(
                np.argmax(predictions[0])
            )

            predicted_class = class_names[
                predicted_index
            ]

            confidence = float(
                predictions[0][predicted_index] * 100
            )


            # ----------------------------------
            # STEP 8: DISPLAY RESULT
            # ----------------------------------

            st.success("Prediction Completed!")

            st.subheader("Prediction Result")

            st.write(
                f"### Predicted Class: {predicted_class}"
            )

            st.write(
                f"### Confidence: {confidence:.2f}%"
            )


            # ----------------------------------
            # STEP 9: CONFIDENCE GRAPH
            # ----------------------------------

            st.subheader(
                "Confidence Scores for All Classes"
            )

            confidence_scores = predictions[0] * 100

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.bar(
                class_names,
                confidence_scores
            )

            ax.set_xlabel("Classes")

            ax.set_ylabel("Confidence (%)")

            ax.set_title(
                "CIFAR-10 Prediction Confidence"
            )

            ax.set_ylim(0, 100)

            plt.xticks(rotation=45)

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


            # ----------------------------------
            # STEP 10: ALL CLASS PREDICTIONS
            # ----------------------------------

            st.subheader("All Class Predictions")

            for i in range(10):

                st.write(
                    f"{class_names[i]}: "
                    f"{confidence_scores[i]:.2f}%"
                )


    except Exception as e:

        st.error(f"Image Processing Error: {e}")


# ----------------------------------
# FOOTER
# ----------------------------------

st.write("---")

st.write(
    "CIFAR-10 Image Classification Project"
)

st.write(
    "Developed using TensorFlow and Streamlit"
)