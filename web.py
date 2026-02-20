import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
import tensorflow_hub as hub

st.cache(allow_output_mutation=True)
CLASS_NAMES = ['actinic keratosis', 
               'basal cell carcinoma',
               'dermatofibroma', 
               'melanoma',
               'nevus',
               'pigmented benign keratosis',
               'seborrheic keratosis',
               'squamous cell carcinoma',
               'vascular lesion']
def load_model():
  model=tf.keras.models.load_model('./Models/Melanoma_Detection7.h5',
                                   custom_objects={'KerasLayer':hub.KerasLayer}
                                   )
  return model
model = load_model()

st.write("""# Melanoma Detection""")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data, model):
        size = (180,180)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img_reshape = np.expand_dims(image,0)
        prediction = model.predict(img_reshape)
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[index]
    confidence = np.max(predictions[0])
    st.write(predicted_class)
    st.write(confidence)
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(CLASS_NAMES[np.argmax(confidence)], 100 * np.max(confidence))
)