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

# Test path folders
folder_O = 'DATASET/TEST/O'
folder_R = 'DATASET/TEST/R'

model_path = 'models/model_cnn_quantized.tflite'

# Load model TFLite
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to preprocess images
def preprocess_image(image_path, input_size):
    # Read image
    img = cv2.imread(image_path)
    # Convert to RGB 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize image
    img_resized = cv2.resize(img, input_size)
    # Normalize the image 
    img_normalized = img_resized.astype(np.float32) / 255.0
    # Add batch dimension
    input_data = np.expand_dims(img_normalized, axis=0)
    return input_data

# predict function
def predict_image(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Assign data to model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get the prediction results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

interpreter = load_tflite_model(model_path)

def predict_fun(image_path):
    # Get the input dimensions of the model
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_size = (input_shape[1], input_shape[2])  # (width, height)

    # Image preprocessing
    input_data = preprocess_image(image_path, input_size)

    # Predict
    predictions = predict_image(interpreter, input_data)

    predicted_class = np.argmax(predictions)
    if predicted_class == 0:
        return 0
    else:
        return 1

# load image 
def load_y_test(folder, label):
    y_test = []
    y_pred = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)

        if img_path is not None:
            # Resize to 224x224
            prediction = predict_fun(img_path)
            y_test.append(label)
            y_pred.append(prediction)
    return y_test, y_pred
          
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