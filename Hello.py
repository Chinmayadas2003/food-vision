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
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)




import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image



# Load your trained model
model_path = 'weights.best.Resnet50.hdf5'
model = tf.keras.models.load_model('weights.best.Resnet50.hdf5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((1, 1))  # Adjust the size based on your model's input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    #img_array /= 255.0  # Normalize pixel values
    return img_array

# Streamlit App
st.title('Food Vision Model Deployment')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

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
    st.write("Prediction:")
    st.write(f"Class: {class_index}")
    st.write(f"Confidence: {confidence:.2%}")

