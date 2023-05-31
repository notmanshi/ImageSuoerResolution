import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load the SRCNN model from the .h5 file
model = tf.keras.models.load_model('SRCNN.h5')

# Set up the app
st.set_page_config(page_title="Image Super Resolution with SRCNN",
                   page_icon=":paintbrush:")
st.title("Image Super Resolution with SRCNN")
st.write("Welcome to the Image Super Resolution with SRCNN. This app uses the SRCNN model to enhance the quality of your images. To get started, simply upload an image using the file uploader below.")
st.write("")

# Define the file uploader
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Check if the user has uploaded a file
if file is not None:
    # Read the image
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), 1)

    # Convert the image to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to a smaller size
    img_small = cv2.resize(img, (256, 256))

    # Normalize the pixel values between 0 and 1
    img_small = img_small.astype('float32') / 255.0

    # Define the enhancement button
    if st.button("Enhance Image"):
        # Use the SRCNN model to upscale the image
        img_high_res = model.predict(np.expand_dims(img_small, axis=0))[0]

        # Resize the image to its original size
        img_high_res = cv2.resize(img_high_res, (img.shape[1], img.shape[0]))

        # Convert the image back to BGR format
        img_high_res = cv2.cvtColor(img_high_res, cv2.COLOR_RGB2BGR)

        # Display the high-resolution image
        with st.spinner("Enhancing image..."):
            st.image(img, clamp=True, caption="Low-Resolution Image")
            st.image(img_high_res, clamp=True, caption="High-Resolution Image")
