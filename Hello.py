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
    #img_array /= 255.0  # Normalize pixel values
    return img_array

# Set page title and favicon
st.set_page_config(page_title="Food Vision App", page_icon="🍔")

# Streamlit App
st.title('Food Vision Model Deployment')
st.markdown("Upload an image to get predictions on the type of food!")

uploaded_file = st.file_uploader("Choose an image...", type="jpg", key="fileUploader")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(uploaded_file)

    # Make predictions
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]

    # Display the results
    st.subheader("Prediction:")
    st.write(f"Class: {class_index}")
    st.write(f"Confidence: {confidence:.2%}")

    # Add some styling for better visual appeal
    st.success("Prediction successfully made!")
    st.balloons()


