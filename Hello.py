# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model_path = 'model.h5'
model = tf.keras.models.load_model(model_path)

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Adjust the size based on your model's input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    # img_array /= 255.0  # Normalize pixel values
    return img_array

# Set page title and favicon
st.set_page_config(
    page_title="Food Vision App",
    page_icon="🍔",
    layout="wide",  # Wide layout for a more stylish appearance
)

# Streamlit App
st.title('Food Vision Model Deployment')

# Sidebar with custom styling
st.sidebar.title("Settings")
st.sidebar.subheader("Choose an image to make predictions")

# Main content area with custom styling
main_column = st.beta_container()  # Use the newer st.beta_container for improved layout control

with main_column:
    # Upload file section with custom styling
    uploaded_file = st.file_uploader("Upload an image...", type="jpg", key="fileUploader", help="Only JPG files are supported.")

    if uploaded_file is not None:
        # Display the uploaded image with rounded corners
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True, output_format="auto", clamp=True, channels="RGB")

        # Preprocess the image
        img_array = preprocess_image(uploaded_file)

        # Make predictions
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        confidence = predictions[0][class_index]

        # Display the results with custom styling
        st.subheader("Prediction:")
        st.write(f"Class: {class_index}")
        st.write(f"Confidence: {confidence:.2%}")

        # Add some additional styling for better visual appeal
        st.success("Prediction successfully made!")
        st.balloons()

        # Customize the appearance of the success message
        success_html = """
        <div style="background-color:#d9f7be;padding:10px;border-radius:10px;">
            <h3 style="color:#4caf50;">Success!</h3>
            <p>Your prediction is ready.</p>
        </div>
        """
        st.markdown(success_html, unsafe_allow_html=True)


