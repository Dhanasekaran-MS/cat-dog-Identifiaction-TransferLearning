import pickle
import cv2
import streamlit as st
import numpy as np
import tensorflow as tf

# Load the model
model = pickle.load(open('cat_dog_model.sav', 'rb'))

# Set page title
st.set_page_config(page_title="Cat Dog Identification App", layout="wide")

# Streamlit Interface
st.title("Cat Dog Identification App")

# Create tabs
tabs = st.tabs(["Home", "Application"])

# Home tab
with tabs[0]:
    st.title("Home")
    st.header("Welcome to My Streamlit App")
    st.write("""
    This Streamlit application is designed to Identify if a picture is of a Dog or Cat.
    - **Home:** This tab contains details about the project.
    - **Application:** This tab contains the main application interface.
    """)

    st.subheader("Project Details")
    st.write("""
    In this project, I have used the Dogs vs Cats Kaggle dataset and applied Transfer Learning Technique [dataset](https://www.kaggle.com/c/dogs-vs-cats/data).
    - Refer to this notebook to learn how I built this project on [GitHub](https://github.com/Dhanasekaran-MS/cat-dog-Identifiaction-TransferLearning/blob/main/dog_cat_classification_Transfer_Leraning.ipynb).
    - I have used the MobileNet V2 model, which is a deep learning model, and I trained it with selected data from the collected dataset.
    - Compiled the model and saved it as a pickle file for predictive modeling.
    - Created a Streamlit app with the saved model.
    - How to use the application:
        - Upload an image and Press Let's Find.        
    """)

# Application tab
with tabs[1]:
    st.title("Application")

    st.write("""
    Upload an image, let's find whether the image represents a Dog or a Cat.
    """)

    # Upload an Image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "webp"])
    if uploaded_file is not None:
        if st.button("Let's Find"):
            # Read the image file
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            input_image = cv2.imdecode(file_bytes, 1)

            # Resize and preprocess the image
            input_image_resize = cv2.resize(input_image, (224, 224))
            input_image_scaled = input_image_resize / 255.0
            image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])

            # Predict the label
            input_prediction = model.predict(image_reshaped)
            input_pred_label = np.argmax(input_prediction)

            if input_pred_label == 0:
                st.balloons()
                st.success('The image represents a Cat')
            else:
                st.balloons()
                st.success('The image represents a Dog')
