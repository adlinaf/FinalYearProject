import streamlit as st
import base64
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
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('sky.jpg') 

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
