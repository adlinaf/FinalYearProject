import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras
from keras.preprocessing import image
from img_classification import teachable_machine_classification

st.title("Image Classification with Teachable Machine")
st.header("Malaysian Banknote Classification")
st.header("Project's Objectives:")
st.header("1. To investigate Malaysian banknotes differences and features.")
st.header("1. To investigate Malaysian banknotes differences and features.")

st.text("Upload a banknote for image classification to identify what banknote it is")
st.markdown(
   f”””
   <style>
   p {
   background-image: url(‘sky.jpg’);
   }
   </style>
   ”””,
   unsafe_allow_html=False)

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
