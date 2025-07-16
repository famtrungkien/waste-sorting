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

model_path = 'models/model_quantized.tflite'
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
    
def predict_fun(image_path):
    # Load model
    interpreter = load_tflite_model(model_path)

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
        print("The image is INORGANIC")
        return 'Organic Waste';
    else:
        print("The image is ORGANIC")
        return 'Inorganic Waste';

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
    if i != 0:
        all_time+= time_process
    print(f"Time process: {time_process} s")
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(prediction)
    axes[i].axis('off')  

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten() 

for i in range(6):
    # Tạo figure để hiển thị
    img_path = 'DATASET/TEST/R/R_' + str(10000+i) + '.jpg'
    start = time.time()
    prediction = predict_fun(img_path)
    end = time.time()
    time_process = end - start
    if i != 0:
        all_time+= time_process
    print(f"Time process: {time_process} s")
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(prediction)
    axes[i].axis('off')  

plt.tight_layout()
plt.show()
print(f"Average processing time per image: {all_time/10} s")
