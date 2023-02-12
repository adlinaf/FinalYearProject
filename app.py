import streamlit as st
import base64
from PIL import Image, ImageOps
from tensorflow import keras
from keras.preprocessing import image
from img_classification import teachable_machine_classification

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('new wallpaper.png')    

st.title("Image Classification with Teachable Machine")
st.header("Malaysian Banknote Classification")
st.subheader("Project's Objectives")
st.markdown(
"""
- To investigate Malaysian banknotes differences and features.
- To build an image classification model using transfer learning.
- To evaluate the accuracy of the classification model.
- To develop a web application by incorporating image classification model that can successfully classify Malaysian banknotes.
"""
)
st.text("Upload a banknote for image classification to identify what banknote it is")

uploaded_file = st.file_uploader("Upload a banknote image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label, accuracy = teachable_machine_classification(image, 'keras_model.h5')
    if accuracy > 0.9:
        if label == 0:
            st.subheader("BANKNOTE RM50")
            st.write(f'(Accuracy: {accuracy:.2f})')
            st.markdown(
    """
    - Fourth Series of Banknotes
    - Currency: RM50
    - RM50 = $11.78
    - Material: Paper
    - Size (mm): 145 x 69s
    - Date: 2012 – present
    - Color: Cyan
    
    Did you know..?
    - RM50 represents oil palm trees as our country earned its status as the second-largest producer of sustainable palm oil. 
    """
    )
        elif label == 1:
            st.subheader("BANKNOTE RM5")
            st.write(f'(Accuracy: {accuracy:.2f})')
            st.markdown(
    """
    - Fourth Series of Banknotes
    - Currency: RM5
    - RM5 = $1.15
    - Material: Polymer
    - Size (mm): 135 x 65
    - Date: 2012 – present
    - Color: Green
    
    Did you know..?
    - RM5 represents the Sarawak's official magnificent bird, Rhinoceros Hornbill.
    """
    )
        elif label == 2:
            st.subheader("BANKNOTE RM10")
            st.write(f'(Accuracy: {accuracy:.2f})')
            st.markdown(
    """
    - Fourth Series of Banknotes
    - Currency: RM10
    - RM10 = $2.31
    - Material: Paper
    - Size (mm): 140 x 65
    - Date: 2012 – present
    - Color: Red
    
    Did you know..?
    - RM10 represents the LARGEST flower in the whole world, Rafflesia.
    """
    )
    else:
        st.write("UNABLE TO CLASSIFY")
