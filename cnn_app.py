import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

model = tf.keras.models.load_model('cnn.h5')


# Streamlit app
st.title('CIFAR-10 IMAGE CLASSIFICATION')

# Show the model summary
if st.checkbox('SHOW MODEL INPUT SUMMARY'):
    st.text('....THIS MODEL IS CREATED USING CIFAR10 DATASET SO PLEASE GIVE INPUT IMAGE COMES UNDER THE CIFAR10 DATASET.... ')

st.write("UPLOAD AN IMAGE OF CIFAR-10 TO CLASSIFY")

# Upload image
uploaded_image = st.file_uploader("CHOOSE AN IMAGE...", type="jpg")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")

    # Preprocess the image for prediction
    image = image.resize((32, 32))
    image = np.array(image)
    image = image / 255.0  # Normalizing the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    
    class_names = ['AIR PLANE', 'AUTOMOBILE', 'BIRD', 'CAT', 'DEER', 'DOG', 'FROG', 'HORSE', 'SHIP', 'TRUCK']
    
    st.write(f'PREDICTION: THE ABOVE IMAGE ARE :   "{class_names[predicted_class[0]]}"')




    st.title(' THIS CNN MODEL IS CREATED BY SANJAY KRISHNAN ')
