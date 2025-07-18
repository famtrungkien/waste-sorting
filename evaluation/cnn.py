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

# Load model
model = load_model('models/my_cnn_model.h5')

# Test path folders
folder_O = 'DATASET/TEST/O'
folder_R = 'DATASET/TEST/R'

# load image 
def load_y_test(folder, label):
    y_test = []
    y_pred = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Resize to 224x224
            prediction = predict_fun(img)
            y_test.append(label)
            y_pred.append(prediction)
    return y_test, y_pred
    
def predict_fun(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to (224x224)
    resized_img = cv2.resize(image_rgb, (224, 224))
    # onvert to float32
    resized_img = resized_img.astype('float32') / 255.0  # Mô hình được huấn luyện với dữ liệu chuẩn hóa 0-1
    # add batch dimension
    input_img = np.expand_dims(resized_img, axis=0)
    # predict
    prediction = model.predict(input_img, verbose=None)
    result = np.argmax(prediction)

    
    if result == 0:
        #print('The image is Organic Waste')
        return 0
    elif result == 1:
        #print('The image is Recycle Waste')
        return 1
        
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

# report classification
report = classification_report(y_test, y_pred, target_names=['Organic', 'Recycle'])
print('Classification Report:')
print(report)