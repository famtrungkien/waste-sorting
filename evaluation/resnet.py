import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Load model
model = load_model('models/resnet50_waste_classifier.h5')

# test folders
folder_O = 'DATASET/TEST/O'
folder_R = 'DATASET/TEST/R'

# load image
def load_y_test(folder, label):
    y_test = []
    y_pred = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path is not None:
            # Resize  224x224
            prediction = predict_fun(img_path)
            y_test.append(label)
            y_pred.append(prediction)
    return y_test, y_pred

class_indices = {'organic': 0, 'recyle': 1} 

# Prediction function from an image
def predict_fun(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # tạo batch 1
    pred = model.predict(img_array, verbose=None)[0][0]
    class_labels = {v: k for k, v in class_indices.items()}
    
    if pred <= 0.5:
        label = 1 
    else:
        label = 0 
    #confidence = pred if pred >= 0.5 else 1 - pred
    #print(f"Prediction: {label} ({confidence*100:.2f}%)")
    return label

y_test_0, y_pred_0 =  load_y_test(folder_O, 0)
y_test_1, y_pred_1 =  load_y_test(folder_R, 1)

y_test = np.array(y_test_0 + y_test_1)
y_pred = np.array(y_pred_0 + y_pred_1)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# confusion
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# report
report = classification_report(y_test, y_pred, target_names=['Organic', 'Recyle'])
print('Classification Report:')
print(report)