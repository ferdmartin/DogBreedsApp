#!streamlit/bin/python
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from GDownload import download_file_from_google_drive

@st.cache(allow_output_mutation=True)
def load_model():
  model_location = '1-q1R5dLfIFW7BbzKuYTjolAoqpjVClsb'
  save_dest = Path('saved_model')
  save_dest.mkdir(exist_ok=True)
  saved_model = Path("saved_model/FerNet_EfficientNet.h5")
  if not saved_model.exists():
      download_file_from_google_drive(model_location, saved_model)
  saved_model = tf.keras.models.load_model(saved_model)
  return saved_model

@st.cache
def load_classes():
    with open(str(Path().parent.absolute())+'/App/classes_dict.json') as classes:
        class_names = json.load(classes)
    return class_names

def load_and_prep_image(filename, img_shape=260):
  #img = tf.io.read_file(filename)
  img = np.array(filename)#tf.io.decode_image(filename, channels=3)
  # Resize our image
  img = tf.image.resize(img, [img_shape,img_shape])
  # Scale
  return img # don't need to resclae images for EfficientNet models in Tensorflow

if __name__ == '__main__':

  saved_model = load_model()
  class_names = load_classes()

  st.header("Dog Breeds Detector")
  st.write("Choose any dog image and get the corresponding breed:")

  uploaded_image = st.file_uploader("Choose an image...")
    
  if uploaded_image:
    uploaded_image = Image.open(uploaded_image)
    image_for_the_model = load_and_prep_image(uploaded_image)
    prediction = saved_model.predict(tf.expand_dims(image_for_the_model, axis=0), verbose=0)
    
    top_k_proba, top_k_indices = tf.nn.top_k(prediction,k=5)
    top_5_classes = {top_n+1:class_names[str(top_k)] for top_n, top_k in enumerate(list(tf.squeeze(top_k_indices).numpy()))}
    top_k_proba = tf.squeeze(top_k_proba).numpy()
    top_5_classes = pd.DataFrame({"Top-k":top_5_classes.keys(), "Dog Breed": top_5_classes.values(), "Probability": top_k_proba})
    top_5_classes.set_index("Top-k", inplace=True)
    
    print(tf.argmax(prediction, axis=1).numpy())
    predicted_breed = class_names[str(tf.argmax(prediction, axis=1).numpy()[0])]
    predicted_breed = ' '.join(predicted_breed.split('_'))
    predicted_breed = predicted_breed.title()
    st.title(f'This dog looks like a {predicted_breed}')
    
    col1, col2 = st.columns(2)
    
    col1.image(uploaded_image,use_column_width=True)
    col2.bar_chart(top_5_classes, y="Dog Breed", x="Probability")
