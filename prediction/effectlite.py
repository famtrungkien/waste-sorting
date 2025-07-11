import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

model = load_model('models/effectlite_model.h5')


def predict_fun(img_path):
    # Upload and process images
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x / 255.0  # Apply the same rescale step
    x = np.expand_dims(x, axis=0)  # Add batch size
    
    # Predict
    prediction = model.predict(x)
    
    # Class determination based on threshold 0.5
    if prediction[0][0] >= 0.5:
        print("The predicted image is INORGANIC")
        return 'Recyclable Waste'
    else:
        print("The predicted image is ORGANIC")
        return 'Organic Waste'




fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()  
all_time = 0

for i in range(6):
    # Create figure to display
    img_path = 'DATASET/TEST/O/O_' + str(12568+i) + '.jpg'
    start = time.time()
    prediction = predict_fun(img_path)
    end = time.time()
    time_process = end - start
    if i!=0:
        all_time+= time_process
    print(f"Time process: {time_process} s")
    #image2 = 'DATASET/TEST/R/R_' + str(10000+i) + '.jpg'
    #result = predict_image(img_path)
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(prediction)
    axes[i].axis('off')



plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()  

for i in range(6):
    # Create figure to display
    img_path = 'DATASET/TEST/R/R_' + str(10000+i) + '.jpg'
    start = time.time()
    prediction = predict_fun(img_path)
    end = time.time()
    time_process = end - start
    if i!=0:
        all_time+= time_process
    print(f"Time process: {time_process} s")
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(prediction)
    axes[i].axis('off')  



plt.tight_layout()
plt.show()




print(f"Average processing time per image: {all_time/10} s")
