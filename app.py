import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras
from keras.preprocessing import image
from img_classification import teachable_machine_classification

st.title("Image Classification with Google's Teachable Machine")
st.header("Malaysian Banknote Classification")
st.text("Upload a banknote for image classification to identify what banknote it is")

uploaded_file = st.file_uploader("Upload a banknote image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'keras_model.h5')
    if label == 0:
        st.write("BANKNOTE RM50")
    elif label == 1:
        st.write("BANKNOTE RM5")
    elif label == 2:
        st.write("BANKNOTE RM10")
    else:
        st.write("UNABLE TO CLASSIFY")