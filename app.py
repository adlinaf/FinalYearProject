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
add_bg_from_local('test.png')    

st.title("Image Classification with Teachable Machine")
st.header("Malaysian Banknote Classification")
st.subheader("Project's Objectives")
st.markdown(
"""
- To investigate Malaysian banknotes differences and features.
- To collect Malaysian banknotes images.
- To build an image classification model using transfer learning.
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
    - Material: Paper
    - Size (mm): 145 x 69
    - Date: 2012 – present
    - Color: Cyan
    """
    )
        elif label == 1:
            st.subheader("BANKNOTE RM5")
            st.write(f'(Accuracy: {accuracy:.2f})')
            st.markdown(
    """
    - Fourth Series of Banknotes
    - Currency: RM5
    - Material: Polymer
    - Size (mm): 135 x 65
    - Date: 2012 – present
    - Color: Green
    """
    )
        elif label == 2:
            st.subheader("BANKNOTE RM10")
            st.write(f'(Accuracy: {accuracy:.2f})')
            st.markdown(
    """
    - Fourth Series of Banknotes
    - Currency: RM10
    - Material: Paper
    - Size (mm): 140 x 65
    - Date: 2012 – present
    - Color: Red
    """
    )
    else:
      st.write("UNABLE TO CLASSIFY")
