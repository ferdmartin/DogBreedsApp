# Fernando_Martinez-Lopez_CS6000_FinalProject

##Fernando's Final Project for Deep Learning Class.

**Student name:** Fernando Martinez Lopez


This folder contains the following files:

* *README.md*: This file.

* *FerNetPaper.pdf*: Final report of this project.

* *FerNetPresentation.pdf*: Presentation slides.

* *StanfordExperiments.ipynb*: Jupyter Notebook file contains all the Stanford Dogs experiments I ran and the results shown during my presentation and found in my paper.

* *StanfordExperiments.html*: HTML version of *StanfordExperiments.ipynb*.

* *TsinghuaExperiments.ipynb*: Jupyter Notebook file contains all the Tsinghua Dogs experiments I ran and the results shown during my presentation and found in my paper.

* *TsinghuaExperiments.html*: HTML version of *TsinghuaExperiments.ipynb*.

* *TestFerNetOnTerminalUsingFriendsImages.py*: Python file to test our best-trained model using the Stanford Dogs dataset. To test our model, I provided a folder that contains images that friends and relatives shared with me to test the model. This program loads the best-trained model from *saved_model*, randomly selects one image from *real_images*, and then predicts the corresponding dog breed in the image.

* *classes_dict.json*: JSON file used has all the classes' names cleaned. The classes provided come from the Stanford Dogs. *TestFerNetOnTerminalUsingFriendsImages.py* reads this file to label its predictions.

* *real_images*: This is a folder/directory. This directory contains some images provided by friends and relatives of their dogs.

* *saved_model*: This is a folder/directory. This directory contains the best-trained FerNet versions trained using both Stanford and Tsinghua datasets.

* *FerNetArchitectureOverview.png*: Representation of the developed architecture.


## How to run *TestFerNetOnTerminalUsingFriendsImages.py*?

**Required libraries**
To run *TestFerNetOnTerminalUsingFriendsImages.py*, it is mandatory to run the following libraries:
* NumPy
* pandas
* json
* tensorflow
* matplotlib

**Instructions to use this code**
1. To run this code, we first need to import the previously mentioned libraries:

```
import numpy as np
import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mping
```

2. Load *classes_dict.json*
3. Load our model. By default, our program reads the best FerNet trained using Stanford Dogs dataset: 
```
fernet_StanfordDogs = tf.keras.models.load_model("saved_model/FerNetEfficientNetB2") # Read model
```

3. Randomly selects one image from *real_images*.
4. Load the selected image and predict!
5. This code can be executed using a terminal.


## StanfordExperiments.ipynb and TsinghuaExperiments.ipynb
Both notebooks created during my experimentation are configured to execute parallel distributed learning. Additionally, I uploaded both datasets to Kaggle to directly perform my experiments in an environment with multiple powerful GPUs and a faster loading process.

**Datasets information**

Given that both datasets are enormous, I didn't provide the files; however, our model can be tested using the images provided on *real_images*. Find some information about the used datasets for benchmarking **FerNet**:

* Stanford Dogs Dataset: 
    * Size: 757MB
    * Num. Images: 20,580
    * Link: http://vision.stanford.edu/aditya86/ImageNetDogs/main.html
* Tsinghua Dogs Dataset: 
    * Size: 2.5GB
    * Num. Images: 70,428
    * Link: https://cg.cs.tsinghua.edu.cn/ThuDogs/

**Required libraries to run the experiments**

To run *TestFerNetOnTerminalUsingFriendsImages.py*, it is mandatory to run the following libraries:

* os
* NumPy
* Pandas
* tensorflow
* matplotlib
* random

**Important note**: Both JupyterNotebooks take significant time to retrain both models; nonetheless, all outputs can be found in the notebooks, and both models are provided in TensorFlow saved model format.

## Additional Information

You can find the application I developed to share with family and friends. This application uses the latest version of **FerNet** which was trained using the Stanford Dogs data.

Link to app: https://femartinez-dogbreedpredictor.streamlit.app 
