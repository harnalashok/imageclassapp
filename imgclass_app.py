# Last amended: 19th October, 2021
# streamlit file to classify images
#
# Ref: https://www.pluralsight.com/guides/deploying-image-classification-on-the-web-with-streamlit-and-heroku
#      streamlit:  https://docs.streamlit.io/library/api-reference
#
#  Objectives:
#            a. Build a simple CNN model for image classfication
#            b. Save the model, host it on github and use it for
#               developing a webapp.
#            c. CNN model building file is:
#                    intel_images_classification.ipynb
#               on github under deeplearning repo.
#            d. Images are stored on gdrive under:
#                  ..Colab_data_files/intel_images
#            e. And saved model (a saved_model/my_model folder) under:
#                 ..Colab_data_files/intelmodel


# 1.0 Call libraries
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import numpy as np
import streamlit as st


# 1.1 Initial titles
st.title("Image Classification of Intel Images")
st.header("Classify image into six classes")
st.subheader("Author: Ashok K Harnal") 
st.text("Upload an image from one of six classes")

# 1.2 Constants to be decided, case by case
size = (1, 64, 64, 3)
img_classes =  ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

#2.0 Our predict function:
def predict(img, model_file, size):
    # 2.1 Load the model
    model = keras.models.load_model(model_file)

    # 2.2 Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=size, dtype=np.float32)
    image = img
    # 2.3 Image re-sizing
    resizeTo = (size[1],size[2])
    image = ImageOps.fit(image, resizeTo, Image.ANTIALIAS)

    # 2.4 Turn the image into a numpy array
    image_array = np.asarray(image)
    # 2.5 Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # 2.6 Load the image into the array
    data[0] = normalized_image_array

    # 2.7 Run the inference
    prediction = model.predict(data)
    return np.argmax(prediction)   # return position of the highest probability


# 3.0 Upload an image file:
uploaded_file = st.file_uploader("Drag and drop an image file ...", type="jpg")

# 3.1 Display and make predictions
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(
              image,
              caption='Uploaded Image.',
              use_column_width=True
             )
    st.write("")
    st.write("Classifying...")
    label = predict(
                    image,
                    'my_model',
                     size
                    )

    st.write("The image is of: ", img_classes[label])
