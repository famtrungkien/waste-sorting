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
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load model
model = load_model('models/vgg16_O_R_classifier.keras')

# Testing dataset
folder_O = 'DATASET/TEST/O'
folder_R = 'DATASET/TEST/R'

# Image function
def load_y_test(folder, label):
    y_test = []
    y_pred = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path is not None:
            # Resize 224x224
            prediction = predict_fun(img_path)
            y_test.append(label)
            y_pred.append(prediction)
    return y_test, y_pred
    
def predict_fun(img_path):
    # load image
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x / 255.0  
    x = np.expand_dims(x, axis=0)  # add batch size
    
    # Predict
    prediction = model.predict(x, verbose=None)
    
    if prediction[0][0] >= 0.5:
        return 1
    else:
        return 0
            
y_test_0, y_pred_0 =  load_y_test(folder_O, 0)
y_test_1, y_pred_1 =  load_y_test(folder_R, 1)

y_test = np.array(y_test_0 + y_test_1)
y_pred = np.array(y_pred_0 + y_pred_1)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# report
report = classification_report(y_test, y_pred, target_names=['Organic', 'Recycle'])
print('Classification Report:')
print(report)