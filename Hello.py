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
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Set page title and favicon
st.set_page_config(
    page_title="🌈 Food Vision App",
    page_icon="🍔",
    layout="wide",
)

# Streamlit App
st.title('🌮 Food Vision Model Deployment')
st.markdown("Upload an image to get predictions on the type of food!")

# Choose a color scheme
primary_color = "#f63366"  # Custom pink color
secondary_color = "#00bcd4"  # Custom teal color

# Set the color scheme for the sidebar
st.markdown(f"""
        <style>
            .reportview-container .sidebar .sidebar-content {{
                background: {primary_color};
                color: #ffffff;
            }}
            .reportview-container .main .block-container {{
                background: #ffffff;
            }}
        </style>
        """, unsafe_allow_html=True)

# Upload file section with colorful styling
uploaded_file = st.file_uploader("Choose an image...", type="jpg", key="fileUploader", help="Only JPG files are supported.")
if uploaded_file is not None:
    # Display the uploaded image with colorful borders
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True, output_format="auto", clamp=True, channels="RGB", style=f"border: 5px solid {secondary_color}; border-radius: 10px;")

    # Preprocess the image
    img_array = preprocess_image(uploaded_file)

    # Make predictions
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]

    # Display the results with colorful styling
    st.subheader("Prediction:")
    st.markdown(f"<span style='color:{primary_color}; font-weight:bold;'>Class:</span> {class_index}", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{primary_color}; font-weight:bold;'>Confidence:</span> {confidence:.2%}", unsafe_allow_html=True)

    # Add some colorful styling for better visual appeal
    st.success("🎉 Prediction successfully made!")
    st.balloons()

