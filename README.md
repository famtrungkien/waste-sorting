# Organic Vs Recyclable Waste Classifier

In this notebook, I will deal with a dataset containing 25,000 images of waste. The task is to build a model to classify this waste into organic waste and recyclable waste. I will experiment with this on Raspberry Pi hardware using the following classifiers:

* Pruning-Quantization-CNN
* ConvneXt
* Resnet
* VGG
* EfficientNet
* MobileNet

Kaggle notebook is available [here](https://www.kaggle.com/code/phamkien999/evaluation-of-models/) 

## Image Visualization & Processing¶

```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

train_path0 = "DATASET/TRAIN/O"
train_path1 = "DATASET/TRAIN/R"

test_path0 = "DATASET/TEST/O"
test_path1 = "DATASET/TEST/R"

import fnmatch

print(len(fnmatch.filter(os.listdir(train_path0), '*.jpg')) +  len(fnmatch.filter(os.listdir(train_path1), '*.jpg')))
print(len(fnmatch.filter(os.listdir(test_path0), '*.jpg')) +  len(fnmatch.filter(os.listdir(test_path1), '*.jpg')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense , BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import plot_model
from glob import glob

def load_data(path):
    x_data = []
    y_data = []
    for category in glob(path+'/*'):
        for file in tqdm(glob(category+'/*')):
            img_array = cv2.imread(file)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            x_data.append(img_array)
            y_data.append(category.split('/')[-1])
    data = pd.DataFrame({'image' : x_data, 'label' : y_data})
    return data

train_path = "DATASET/TRAIN/"
test_path = "DATASET/TEST/"

# Load train data
train_data = load_data(train_path)
# Load test data
test_data = load_data(test_path)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
colors = ['green', 'red']
plt.pie(train_data['label'].value_counts(), 
        labels=['Organic', 'Recyclable'], 
        autopct='%0.2f%%', 
        colors=colors, 
        startangle=90, 
        explode=[0.05, 0.05])
plt.rcParams.update({'font.size': 24})
plt.title('Train Dataset')

plt.subplot(1, 2, 2)
plt.pie(test_data['label'].value_counts(), 
        labels=['Organic', 'Recyclable'], 
        autopct='%0.2f%%', 
        colors=colors, 
        startangle=90, 
        explode=[0.05, 0.05])
plt.title('Test Dataset')
plt.rcParams.update({'font.size': 24})
plt.tight_layout()
plt.show()
```

![alt text](https://i.postimg.cc/fbqrP32Z/dataset.png)


# 1. CNN Model
## 1.1 Evaluation of CNN Model

```python
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
```

![alt text](https://github.com/famtrungkien/waste-sorting/blob/main/images/evaluations/orginil_cnn_model.PNG)


## 1.2. Evaluation of PrQ-CNN model
```python
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
```
![alt text](https://github.com/famtrungkien/waste-sorting/blob/main/images/evaluations/cnn_model.PNG)

## 1.4. Compare the sizes of CNN, pruned, quantized models.
```python
import os
def getsize(path):
    return os.path.getsize(path)

path_cnn = 'my_cnn_model.h5'
path_quantized_cnn = 'model_cnn_quantized.tflite'
path_pruned_cnn= 'pruned_cnn_model.h5'

print(f"Size of CNN model: {getsize(path_cnn)/1000000}MB")
print(f"Size of Pruned CNN model: {getsize(path_pruned_cnn)/1000000}MB")
print(f"Size of Quantized CNN model: {getsize(path_quantized_cnn)/1000000}MB")

print(f"Quantized CNN model reduce: {getsize(path_cnn)/getsize(path_quantized_cnn)} times")
```
![alt text](https://github.com/famtrungkien/waste-sorting/blob/main/images/evaluations/compare%20cnn.PNG)

# 2. ConvNext Model

## 2.1. Evaluation of ConvNext Model
```python
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
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'models/convnext_rac_model.pth'

# Path of dataset
train_dir = "DATASET/TRAIN/"
test_dir = "DATASET/TEST/"

# Transform dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Testing dataset
folder_O = 'DATASET/TEST/O'
folder_R = 'DATASET/TEST/R'

# Load saved model
model = models.convnext_tiny(pretrained=False)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Predict function
def predict_image(image_path, model):
    image = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])(Image.open(image_path).convert('RGB'))

    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        class_name = train_dataset.classes[pred_class]
        confidence = probs[0][pred_class].item()
    #return class_name, confidence

# Class 0 is organic, class 1 is inorganic
    if class_name == 'O':
        #print('The image is Organic Waste')
        return 0
    elif class_name == 'R':
        #print('The image is Inorganic Waste')
        return 1
    
from PIL import Image

# Image loading and preprocessing function
def load_y_test(folder, label):
    y_test = []
    y_pred = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        #img = cv2.imread(img_path)
        if img_path is not None:
            # Resize ảnh về kích thước phù hợp, ví dụ 224x224
            prediction = predict_image(img_path, model)
            y_test.append(label)
            y_pred.append(prediction)
    return y_test, y_pred
       
y_test_0, y_pred_0 =  load_y_test(folder_O, 0)
y_test_1, y_pred_1 =  load_y_test(folder_R, 1)

y_test = np.array(y_test_0 + y_test_1)
y_pred = np.array(y_pred_0 + y_pred_1)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Classification report
report = classification_report(y_test, y_pred, target_names=['Organic', 'Recycle'])
print('Classification Report:')
print(report)
```
![alt text](https://github.com/famtrungkien/waste-sorting/blob/main/images/evaluations/convnext.PNG)


# 3. VGG16 
## 3.1. Evaluation of VGG16 Model
```python
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

# Test data directory
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
```

![alt text](https://github.com/famtrungkien/waste-sorting/blob/main/images/evaluations/vgg16.PNG)

# 4. MobileNetV2
## 4.1. Evaluation of MobileNetV2 Model
```python
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
model = load_model('models/my_model_mobilenetv2.h5')

# test folders
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
            # Resize  224x224
            prediction = predict_fun(img)
            y_test.append(label)
            y_pred.append(prediction)
    return y_test, y_pred
    
def predict_fun(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize image (224x224)
    resized_img = cv2.resize(image_rgb, (224, 224))
    # Convert float32
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
        #print('The image is Inorganic Waste')
        return 1
        
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
report = classification_report(y_test, y_pred, target_names=['Organic', 'Recycle'])
print('Classification Report:')
print(report)
```

![alt text](https://github.com/famtrungkien/waste-sorting/blob/main/images/evaluations/mobilenetv2.PNG)

# 4. ResNet
## 4.1. Evaluation of ResNet Model
```python
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

# predict function from an image
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
```
![alt text](https://github.com/famtrungkien/waste-sorting/blob/main/images/evaluations/resnet.PNG)

# 5. EfficientNet
## 5.1 Evaluation of EfficientNet Model
```python
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

model = load_model('models/my_model_effectlite.h5')

folder_O = 'DATASET/TEST/O'
folder_R = 'DATASET/TEST/R'

def load_y_test(folder, label):
    y_test = []
    y_pred = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Resize  224x224
            prediction = predict_fun(img)
            y_test.append(label)
            y_pred.append(prediction)
    return y_test, y_pred
    
def predict_fun(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize (224x224)
    resized_img = cv2.resize(image_rgb, (224, 224))
    # Cconvert float32
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
        #print('The image is Inorganic Waste')
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

# report
report = classification_report(y_test, y_pred, target_names=['Organic', 'Recycle'])
print('Classification Report:')
print(report)
```
![alt text](https://github.com/famtrungkien/waste-sorting/blob/main/images/evaluations/EfficientNet.PNG)

# 6. Evaluation of Models.
## 6.1 Confusion Matrix of Models
```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

# Define the confusion matrices
cmats = [
    np.array([[1340, 61], 
              [141, 971]]),   # CNN
    np.array([[1385, 16],
              [43, 1069]]), #ConvNext
    np.array([[1343, 58],
              [217, 895]]), #VGG16
    np.array([[1359, 42],
              [247, 865]]), #MobileNetV2
    np.array([[1368, 33],
              [190, 922]]), #RESNet
    np.array([[1362, 39],
              [187, 925]]) #EffectLiteNet
]

# Create a 3x2 grid for subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 12))
axes = axes.flatten()  # Flatten to easily iterate

# Plot each confusion matrix
for i, cm in enumerate(cmats):
    disp = ConfusionMatrixDisplay(cm)
    
    disp.plot(ax=axes[i], colorbar=False)
    
    if i== 0:
        axes[i].set_title(f'PQ-CNN')
    if i== 1:
        axes[i].set_title(f'ConvNeXt')
    if i== 2:
        axes[i].set_title(f'VGG16')
    if i== 3:
        axes[i].set_title(f'MobileNetV2')
    if i== 4:
        axes[i].set_title(f'ResNet')
    if i== 5:
        axes[i].set_title(f'EfficientNet')

# Remove the unused subplot (since only 5 matrices for 6 subplots)
#fig.delaxes(axes[-1])  # delete the last subplot
plt.rcParams["figure.figsize"] = (20,3)
plt.rcParams.update({'font.size': 30})
plt.tight_layout()
plt.savefig('bieu_do_confusion.png')  
plt.show()
```

![alt text](https://github.com/famtrungkien/waste-sorting/blob/main/images/results/bieu_do_confusion.png)

## 6.2. Comparison of Models
```python
import numpy as np

# confusion matrix
cmats = [
    np.array([[1340, 61], 
              [141, 971]]),   # CNN
    np.array([[1385, 16],
              [43, 1069]]), #ConvNext
    np.array([[1343, 58],
              [217, 895]]), #VGG16
    np.array([[1359, 42],
              [247, 865]]), #MobileNetV2
    np.array([[1368, 33],
              [190, 922]]), #RESNet
    np.array([[1362, 39],
              [187, 925]]) #EffectLiteNet
]
results = []
# calculate MDR, FDR
for idx, cm in enumerate(cmats):
    TP = cm[1, 1]
    FN = cm[1, 0]
    FP = cm[0, 1]

    
    
    MDR = FN / (TP + FN) if (TP + FN) != 0 else 0
    FDR = FP / (TP + FP) if (TP + FP) != 0 else 0
    
    results.append({
            'model_index': idx + 1,
            'MDR': MDR,
            'FDR': FDR
        })

for res in results:
    print(f"Model {res['model_index']}: MDR = {res['MDR']:.4f}, FDR = {res['FDR']:.4f}")
```
![alt text](https://github.com/famtrungkien/waste-sorting/blob/main/images/evaluations/evaluation2.PNG)

```python
import pandas as pd
import matplotlib.pyplot as plt

# dataset of models
models = [
    {
        'name': 'PrQ-CNN',
        'size': 22.28,
        'accuracy': 92.0,
        'speed_ras': 76.3,
        'speed_kaggle': 54.0,
        'MDR': 0.1268,
        'FDR': 0.0591
    },
    {
        'name': 'ConvNeXt',
        'size': 111.35,
        'accuracy': 98.0,
        'speed_ras': 234.3,
        'speed_kaggle': 71.6,
        'MDR': 0.0387,
        'FDR': 0.0147
    },
    {
        'name': 'VGG16',
        'size': 273.56,
        'accuracy': 89.0,
        'speed_ras': 472.2,
        'speed_kaggle': 90.4,
        'MDR': 0.1951,
        'FDR': 0.0609
    },
    {
        'name': 'MobileNetV2',
        'size': 9.6,
        'accuracy': 88.0,
        'speed_ras': 130.4,
        'speed_kaggle': 78.5,
        'MDR': 0.2221,
        'FDR': 0.0463
    },
    {
        'name': 'ResNet',
        'size': 283.44,
        'accuracy': 91.0,
        'speed_ras': 133.4,
        'speed_kaggle': 72.9,
        'MDR': 0.1709,
        'FDR': 0.0346
    },
    {
        'name': 'EfficientNet',
        'size': 244.22,
        'accuracy': 91.0,
        'speed_ras': 250.0,
        'speed_kaggle': 91.0,
        'MDR': 0.1682,
        'FDR': 0.0405
    }
]


# create DataFrame
df = pd.DataFrame(models)

# draw plots
fig, axes = plt.subplots(2, 3, figsize=(28, 20))
axes = axes.flatten()

parameters = ['size', 'accuracy', 'speed_ras', 'speed_kaggle', 'MDR', 'FDR']
titles = ['Size (MB)', 'Accuracy (%)', 'Speed RAS (ms)', 
          'Speed Kaggle (ms)', 'MDR', 'FDR']

for i, param in enumerate(parameters):
    ax = axes[i]
    ax.bar(df['name'], df[param], color='skyblue')
    ax.set_title(titles[i])
    ax.set_ylabel(param)
    ax.set_xticklabels(df['name'], rotation=45, ha='right')
    ax.grid(axis='y')
    
plt.rcParams.update({'font.size': 50})
plt.tight_layout()

# save image plot
plt.savefig('bieu_do_mo_hinh.png')  
# show image plot
plt.show()
```
![alt text](https://github.com/famtrungkien/waste-sorting/blob/main/images/results/bieu_do_mo_hinh.png)

```python
import pandas as pd

# dataset of models
models = [
    {
        'name': 'PrQ-CNN',
        'size': 22.28,
        'accuracy': 92.0,
        'speed_ras': 76.3,
        'speed_kaggle': 54.0,
        'MDR': 0.1268,
        'FDR': 0.0591
    },
    {
        'name': 'ConvNeXt',
        'size': 111.35,
        'accuracy': 98.0,
        'speed_ras': 234.3,
        'speed_kaggle': 71.6,
        'MDR': 0.0387,
        'FDR': 0.0147
    },
    {
        'name': 'VGG16',
        'size': 273.56,
        'accuracy': 89.0,
        'speed_ras': 472.2,
        'speed_kaggle': 90.4,
        'MDR': 0.1951,
        'FDR': 0.0609
    },
    {
        'name': 'MobileNetV2',
        'size': 9.6,
        'accuracy': 88.0,
        'speed_ras': 130.4,
        'speed_kaggle': 78.5,
        'MDR': 0.2221,
        'FDR': 0.0463
    },
    {
        'name': 'ResNet',
        'size': 283.44,
        'accuracy': 91.0,
        'speed_ras': 133.4,
        'speed_kaggle': 72.9,
        'MDR': 0.1709,
        'FDR': 0.0346
    },
    {
        'name': 'EfficientNet',
        'size': 244.22,
        'accuracy': 91.0,
        'speed_ras': 250.0,
        'speed_kaggle': 91.0,
        'MDR': 0.1682,
        'FDR': 0.0405
    }
]

# convert to DataFrame
df = pd.DataFrame(models)

# parameters
parameters = ['size', 'accuracy', 'speed_ras', 'speed_kaggle', 'MDR', 'FDR']

# get min and max
def get_model_for_value(column, find_max=True):
    if find_max:
        idx = df[column].idxmax()
    else:
        idx = df[column].idxmin()
    return df.iloc[idx]['name']

# create table 1: min size, max accuracy, min speed1, speed2, MDR, FDR
table1_data = {
    'Parameter': parameters,
    'Value': [
        df['size'].min(),
        df['accuracy'].max(),
        df['speed_ras'].min(),
        df['speed_kaggle'].min(),
        df['MDR'].min(),
        df['FDR'].min()
    ],
    'Model': [
        get_model_for_value('size', find_max=False),
        get_model_for_value('accuracy', find_max=True),
        get_model_for_value('speed_ras', find_max=False),
        get_model_for_value('speed_kaggle', find_max=False),
        get_model_for_value('MDR', find_max=False),
        get_model_for_value('FDR', find_max=False)
    ]
}

# create table 2: max size, min accuracy, max speed1, speed2, MDR, FDR
table2_data = {
    'Parameter': parameters,
    'Value': [
        df['size'].max(),
        df['accuracy'].min(),
        df['speed_ras'].max(),
        df['speed_kaggle'].max(),
        df['MDR'].max(),
        df['FDR'].max()
    ],
    'Model': [
        get_model_for_value('size', find_max=True),
        get_model_for_value('accuracy', find_max=False),
        get_model_for_value('speed_ras', find_max=True),
        get_model_for_value('speed_kaggle', find_max=True),
        get_model_for_value('MDR', find_max=True),
        get_model_for_value('FDR', find_max=True)
    ]
}

# Convert to DataFrame
df_table1 = pd.DataFrame(table1_data)
df_table2 = pd.DataFrame(table2_data)

print("Table 1 (Min size, Max accuracy, Min speed_ras, speed_kaggle, MDR, FDR")
print(df_table1)

print("\nTable 2 (Max size, Min accuracy, Max speed_ras, speed_kaggle, MDR, FDR")
print(df_table2)
```
![alt text](https://github.com/famtrungkien/waste-sorting/blob/main/images/results/table.PNG)

## License

[MIT](https://choosealicense.com/licenses/mit/)
