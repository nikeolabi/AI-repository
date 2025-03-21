# =============================================================================
# Streamlit application for MNIST tarining model. 
# Created by Nike Olabiyi, v.1.0
# =============================================================================

import numpy as np
import pandas as pd

import streamlit as st
from sklearn.datasets import fetch_openml
import joblib

from sklearn.preprocessing import StandardScaler
from io import BytesIO
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import os
import zipfile

import logging
logging.basicConfig(level=logging.INFO)
logging.info("Current working directory: %s", os.getcwd())

# Predefined model names (this could be your model names like 'Random Forest', 'SVM', etc.)
model_names = ["ExtraTreesClassifier", "Random Forest", "SVM Classifier (non linear)", "SVM Classifier (pca)"]
extraction_folder = 'extracted_models/'
saved_models_folder = 'SavedModels/'
current_model = ''
threshold = 230

print("Current dir = ", os.getcwd())
# Load the scalers
scaler = joblib.load(saved_models_folder + 'scaler.pkl')
pca_scaler = joblib.load(saved_models_folder + 'pca_scaler.pkl')

# =============================================================================
# Loading The Model 
# =============================================================================
def load_the_model_and_predict(image_to_predict):
    # Define model configurations
    model_configs = {
        "ExtraTreesClassifier": {
            "zip_file": saved_models_folder + 'extra_trees_clf.zip',
            "model_file": 'extra_trees_clf.pkl',
            "scaler_required": False,
            "pca_scaler_required": False
        },
        "Random Forest": {
            "zip_file": saved_models_folder + 'best_rf_model_non_scaled.zip',
            "model_file": 'best_rf_model_non_scaled.pkl',
            "scaler_required": False,
            "pca_scaler_required": False
        },
        "SVM Classifier (non linear)": {
            "zip_file": saved_models_folder + 'svm_classifier.zip',
            "model_file": 'svm_classifier.pkl',
            "scaler_required": True,
            "pca_scaler_required": False
        },        
        "SVM Classifier (pca)": {
            "zip_file": saved_models_folder + 'svm_classifier_pca.zip',
            "model_file": 'svm_classifier_pca.pkl',
            "scaler_required": True,
            "pca_scaler_required": True
        }
    }

    # Validate selected model and get configuration
    if selected_model_name not in model_configs:
        raise ValueError("Invalid choice! Please select a valid model.")
    
    config = model_configs[selected_model_name]
    zip_file = config["zip_file"]
    current_model = config["model_file"]
    
    # Preprocess image if scaler is required
    if config["scaler_required"]:
        flattened_image_2d = image_to_predict.reshape(1, -1)  # Shape: (1, 784)
        image_to_predict = scaler.transform(flattened_image_2d)  # Transform with scaler for SVC
        if config["pca_scaler_required"]:
            image_to_predict = pca_scaler.transform(image_to_predict)  # Apply PCA transformation
        image_to_predict = np.array(image_to_predict).flatten()

    # Ensure extraction folder exists
    if not os.path.exists(extraction_folder):
        os.makedirs(extraction_folder)  # Create folder if it doesn't exist

    # Check if the model has been extracted
    extracted_files = os.listdir(extraction_folder)
    if current_model not in extracted_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extraction_folder)  # Extract the model
            print(f"Model {selected_model_name} extracted successfully.")
    else:
        print(f"Model {selected_model_name} is already extracted.")

    # Load the model from the .pkl file
    model_path = os.path.join(extraction_folder, current_model)
    model = joblib.load(model_path)  # Load the model

    # Predict the probabilities for each class (digit 0-9)
    probabilities = model.predict_proba([image_to_predict])[0]
    predicted_class = np.argmax(probabilities)
    st.write(f"Predicted number: {predicted_class}")

    # Display the probabilities as a DataFrame and bar chart
    prob_df = pd.DataFrame({
        "Digit": list(range(10)),
        "Probability": probabilities
    })
    
    st.write("Probabilities for each digit (0-9):") # Display the probabilities
    st.bar_chart(prob_df.set_index("Digit"))  # Display probabilities as a bar chart
  

# =============================================================================
# Process the image
# =============================================================================
def process_the_image(canvas_result):
    # Convert canvas image to NumPy array
    image_array = np.array(canvas_result.image_data)

    # Convert to grayscale (extract any channel, since it's black-and-white)
    grayscale_image = image_array[:, :, 0]  # Extract the red channel

    # Invert colors (MNIST has black background and white digit)
    inverted_image = 255 - grayscale_image  # Make digit white and background black
    inverted_image = (inverted_image > threshold).astype(np.uint8) * 255

    # Convert to PIL Image
    image = Image.fromarray(inverted_image)

    # Automatically detect bounding box (where the digit is)
    bbox = image.getbbox()  # No need to invert again

    if bbox:
        # Crop the image to remove excess white space
        image = image.crop(bbox)

    # Get the dimensions of the cropped image
    width, height = image.size

    # Resize the digit to a smaller size (e.g., 20x20) while preserving aspect ratio
    target_size = 20  # Size of the digit within the 28x28 canvas
    scale = min(target_size / width, target_size / height)  # Preserve aspect ratio
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)  # High-quality resizing

    # Create a new blank (28x28) black image
    new_image = Image.new("L", (28, 28), 0)  # "L" mode = grayscale, black background

    # Calculate the offset to center the resized digit in the 28x28 canvas
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    # Paste the resized digit into the center of the new 28x28 image
    new_image.paste(image, (x_offset, y_offset))

    # Shape: (784,) 1D array for Forrests/Trees
    flattened_image = np.array(new_image).flatten() 
    
    return new_image, flattened_image  # Return processed image


# =============================================================================
# Check if canvas is empty
# =============================================================================

def is_canvas_empty(canvas_result):
    if canvas_result is None or canvas_result.image_data is None:
        return True  # No data means empty

    # Convert image to NumPy array
    image_array = np.array(canvas_result.image_data)

    # Convert to grayscale (use only one color channel, assuming white drawing on black)
    grayscale_image = image_array[:, :, 0]  # Extract the first channel (red/grayscale)

    # Check if all pixels are the same (black background)
    return np.all(grayscale_image == 255)


# =============================================================================
# Creating the streamlit application views
# =============================================================================

# Creating a navigation menu with three different sections the user can choose. 
nav = st.sidebar.radio("Navigation Menu",["About", "Predict"])

# =============================================================================
# ABOUT view
# =============================================================================
if nav == "About":
    st.title("Streamlit - a number prediction")
    st.header("About")
    st.write("Created by Nike Olabiyi, 28 Feb. 2025, v.1.0")
    st.write("""The purpose of this application is to predict/guess a number 
    written by a user. Please navigate to Predict and submit a picture of a number.""")
    st.write("")
    st.write("If you want to know about the data set the model was tarined on, press About MNIST button below")
    # Button to trigger prediction
    if st.button("About MNIST"):
        mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
        st.write(mnist.DESCR)

# =============================================================================
# PREDIC view
# =============================================================================
if nav == "Predict":
    st.title("Let the model guess your own written number")
    st.write('In this section you can download your own written number and let the model to guess it!\nPlease, write just one number at a time!')
    st.write("")
    st.write("SVM Classifier was trained on 10 000 instances only and is the smallest one. However it competes well with Forrest and Trees models.")
    st.write("Numbers 6 and 9 are seldom predicted correctly by either model. They are mostly predicted as 5 and 9, respectively.")
    st.write("")
    st.write("")

    # Create a two-column layout: Left for canvas, Right for prcessed image
    col1, col2 = st.columns(2)

    with col1:
        # Drawing canvas
        col1.write("Draw a number:")
        canvas_result = st_canvas(
            fill_color="none",
            stroke_width=10,
            stroke_color="black",
            background_color="white",
            height=200,
            width=200,
            drawing_mode="freedraw",
            key="canvas"
        )

        # Dropdown to select the model by name
        selected_model_name = st.selectbox("Choose a model:", model_names, key="unique_model_select")

        # Button to trigger prediction
        if st.button("Predict"): 
            
             # (1) Check if the canvas is empty
             if is_canvas_empty(canvas_result):
                 st.warning("Please draw a number in the canvas before predicting.")
                 # Button to acknowledge the warning
                 if st.button("OK"):
                    # When clicked, set the warning to be hidden
                    st.session_state.warning_shown = False
             else: # (2) if not proceed with processing the image
                processed_image, flattened_image = process_the_image(canvas_result)
                
                 # (3) visualize the processed image
                if processed_image:
                    with col2:
                        col2.write("Processed Image:")
                        # Create the figure directly in col2
                        fig, ax = plt.subplots()  # Set figure size
                        ax.imshow(processed_image, cmap="gray")  # Show the grayscale processed image
                        ax.axis("off")  # Hide axes
                        col2.pyplot(fig)  # Display Matplotlib figure in col2
                else:
                    st.warning("Something went wrong in image procesing!")

                # (4) load the model and predict the image
                load_the_model_and_predict(flattened_image)
