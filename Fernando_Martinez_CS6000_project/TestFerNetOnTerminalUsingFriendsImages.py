import numpy as np
import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mping

with open("classes_dict.json",) as classes: # Read clean list
    class_names = json.load(classes)

fernet_StanfordDogs = tf.keras.models.load_model("saved_model/FerNetEfficientNetB2") # Read model

def load_and_predict(model, filename, img_shape=260):
    img = mping.imread(filename)
    prediction = model.predict(tf.expand_dims(tf.image.resize(img,(img_shape,img_shape)), axis=0), verbose=0)
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
    plt.imshow(img)
    plt.axis(False)
    plt.title(f"I think that in this image we have a: {class_names[str(predicted_class)]}")
    plt.show()

image_to_use = "real_images/"+np.random.choice(os.listdir("real_images")) # Random sample from list of images provided by friends and relatives
load_and_predict(fernet_StanfordDogs, image_to_use, 260) # Load, predict, and show image with prediction