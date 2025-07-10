# Organic Vs Recyclable Waste Classifier

In this notebook we will deal with a dataset containing 25,000 images of waste. This task is to build a model to classify this waste into organic waste and recyclable waste. We will experiment with this to Raspberry Pi hardware with flowing classifiers:

* Pruning-Quantization-CNN
* ConvneXt
* Resnet
* VGG
* EfficientNet
* MobileNet

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

train_path0 = "/kaggle/input/waste-classification-data/DATASET/TRAIN/O"
train_path1 = "/kaggle/input/waste-classification-data/DATASET/TRAIN/R"

test_path0 = "/kaggle/input/waste-classification-data/DATASET/TEST/O"
test_path1 = "/kaggle/input/waste-classification-data/DATASET/TEST/R"

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

train_path = "/kaggle/input/waste-classification-data/DATASET/TRAIN/"
test_path = "/kaggle/input/waste-classification-data/DATASET/TEST/"

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
![alt text](https://i.ibb.co/4wJ56yTR/dataset.png)
# 1. CNN Model
## 1.1 Training off CNN Model
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import random
import tensorflow as tf

# Đường dẫn dữ liệu
train_path = "/kaggle/input/waste-classification-data/DATASET/TRAIN/"
test_path = "/kaggle/input/waste-classification-data/DATASET/TEST/"

seed_value = 30
batch_size = 64  # Thay đổi phù hợp hơn
target_size = (224, 224)
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# Tăng cường dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,             # Quay ngược 30 độ
    width_shift_range=0.2,         # Dịch chuyển ngang 20%
    height_shift_range=0.2,        # Dịch chuyển dọc 20%
    zoom_range=0.2,                # Phóng to/thu nhỏ 20%
    horizontal_flip=True,          # Lật phải/trái
    brightness_range=[0.8, 1.2], # Điều chỉnh độ sáng
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
)

# Load dữ liệu
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    seed=seed_value
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    seed=seed_value
)

# Xây dựng mô hình
model = Sequential()

# Conv Block 1
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

# Conv Block 2
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

# Conv Block 3
model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

# Fully connected layers
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(2, activation='softmax'))  # Dùng softmax cho phân loại đa lớp

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Vẽ mô hình
plot_model(model, to_file='cnn_model.png', show_shapes=True, show_layer_names=True)
from IPython.display import Image
Image('cnn_model.png')

model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Huấn luyện
history = model.fit(
    train_generator,
    epochs=30,  # Tăng số epochs để có thể tinh chỉnh tốt hơn
    validation_data=test_generator,
    callbacks=[early_stop, reduce_lr]
)

# Lưu mô hình
model.save('waste_classifier_model.h5')

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over epochs')
plt.show()
```
![alt text](https://i.ibb.co/ksYfJVJb/cnn-model.png)

## 1.2 Evaluation of CNN Model

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
model = load_model('/kaggle/input/cnn_waste_model/tensorflow2/default/1/waste_classifier_model.h5')

# Test path folders
folder_O = '/kaggle/input/waste-classification-data/DATASET/TEST/O'
folder_R = '/kaggle/input/waste-classification-data/DATASET/TEST/R'

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

# report classification
report = classification_report(y_test, y_pred, target_names=['InOrganic', 'Organic'])
print('Classification Report:')
print(report)
```
![alt text](https://i.ibb.co/ksYfJVJb/cnn-model.png)
## 1.3 Pruning Model 
```python
import numpy as np
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
model = load_model('/kaggle/input/cnn_waste_model/tensorflow2/default/1/waste_classifier_model.h5')


threshold = 0.001

# Iterate through each layer of the model
for layer in model.layers:
    if hasattr(layer, 'weights'):
        weights = layer.get_weights()
        # Update weights
        new_weights = []
        for w in weights:
            w[np.abs(w) < threshold] = 0
            new_weights.append(w)
        layer.set_weights(new_weights)

model.save('pruned_cnn_model.h5')  # save pruned model
print("saved pruned cnn model")
```
![alt text](https://i.ibb.co/kVVFVYRc/pruned-cnn.png)
## 1.4. Quantization of CNN model
```python

import tensorflow as tf
from tensorflow.keras.models import load_model

# Load pruned model
pruned_model = tf.keras.models.load_model('/kaggle/input/pruned_cnn/tensorflow2/default/1/pruned_cnn_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# save quantized model
with open('model_cnn_quantized.tflite', 'wb') as f:
    f.write(tflite_quantized_model)
```
![alt text](https://i.ibb.co/VY3MtCks/quantized-cnn.png)

## 1.5. Evaluation of PQ-CNN model
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
folder_O = '/kaggle/input/waste-classification-data/DATASET/TEST/O'
folder_R = '/kaggle/input/waste-classification-data/DATASET/TEST/R'

model_path = '/kaggle/input/quantized_model/tensorflow2/default/1/quantized_cnn_model.tflite'

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
report = classification_report(y_test, y_pred, target_names=['InOrganic', 'Organic'])
print('Classification Report:')
print(report)
```
![alt text](https://i.ibb.co/wFF9Y3rz/pq-cnn.png)

## 1.6.  Example of PQ-CNN Model Prediction

```python
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

model_path = '/kaggle/input/quantized_model/tensorflow2/default/1/quantized_cnn_model.tflite'
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
    img_path = '/kaggle/input/waste-classification-data/DATASET/TEST/O/O_' + str(12568+i) + '.jpg'
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
    # create figure
    img_path = '/kaggle/input/waste-classification-data/DATASET/TEST/R/R_' + str(10400+i) + '.jpg'
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
```
### Kaggle GPU T4
![alt text](https://i.ibb.co/gbpHp5cv/pq-cnn.jpg)

### Raspberry Pi 5
![alt text](https://i.ibb.co/5W8NsWKh/cnn-log.png)

## 1.7. Compare the sizes of CNN, pruned, quantized models.
```python
import os
def getsize(path):
    return os.path.getsize(path)

path_cnn = 'waste_classifier_model.h5'
path_quantized_cnn = 'quantized_cnn_model.tflite'
path_pruned_cnn= 'pruned_cnn_model.h5'

print(f"Size of CNN model: {getsize(path_cnn)/1000000}MB")
print(f"Size of Pruned CNN model: {getsize(path_pruned_cnn)/1000000}MB")
print(f"Size of Quantized CNN model: {getsize(path_quantized_cnn)/1000000}MB")

print(f"Quantized CNN model reduce: {getsize(path_cnn)/getsize(path_quantized_cnn)} times")
```
![alt text](https://i.ibb.co/PZM8Yb2k/size-compare.png)

# 2. ConvNext Model
## 2.1. Training ConvNext Model
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import numpy as np

# Đặt device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Đường dẫn dữ liệu
train_dir = "/kaggle/input/waste-classification-data/DATASET/TRAIN/"
test_dir = "/kaggle/input/waste-classification-data/DATASET/TEST/"
model_path = 'convnext_rac_model.pth'

# Hàm huấn luyện
def train(model, loader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")

# Hàm đánh giá
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f'Accuracy: {100 * correct / total:.2f}%')

seed_value = 0
torch.manual_seed(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Transform dữ liệu
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Tạo dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
# Tạo dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Khởi tạo mô hình ConvNeXt
model = models.convnext_tiny(pretrained=True)

# Thay đổi lớp cuối để phù hợp với 2 lớp (Vô cơ và Hữu cơ)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)

model = model.to(device)

# Đặt hàm loss và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Đào tạo mô hình
train(model, train_loader, criterion, optimizer, epochs=1)

# Đánh giá mô hình
evaluate(model, test_loader)

# Lưu mô hình
torch.save(model.state_dict(), model_path)
print(f"Mô hình đã lưu tại {model_path}")

# --- Load lại mô hình và dự đoán một ảnh trong thư mục O ---

# Tải mô hình đã lưu
model_loaded = models.convnext_tiny(pretrained=False)
model_loaded.classifier[2] = nn.Linear(model_loaded.classifier[2].in_features, 2)
model_loaded.load_state_dict(torch.load(model_path))
model_loaded = model_loaded.to(device)
model_loaded.eval()

# Hàm dự đoán
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
    return class_name, confidence

from PIL import Image

# Đường dẫn ảnh trong thư mục O
sample_image_path = '/kaggle/input/waste-classification-data/DATASET/TEST/O/O_12568.jpg'  # Thay bằng đường dẫn thực tế

# Dự đoán
predicted_class, confidence = predict_image(sample_image_path, model_loaded)
print(f'Ảnh dự đoán là: {predicted_class} với độ tin cậy {confidence:.2f}')
```
![alt text](https://i.ibb.co/gL6ymdRs/train-convnext.png)
## 2.2. Evaluation of ConvNext Model
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

# Đặt device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '/kaggle/input/convnext98/tensorflow2/default/1/convnext_rac_model_98.pth'

# Đường dẫn dữ liệu
train_dir = "/kaggle/input/waste-classification-data/DATASET/TRAIN/"
test_dir = "/kaggle/input/waste-classification-data/DATASET/TEST/"

# Transform dữ liệu
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Tạo dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Thư mục chứa dữ liệu test
folder_O = '/kaggle/input/waste-classification-data/DATASET/TEST/O'
folder_R = '/kaggle/input/waste-classification-data/DATASET/TEST/R'

# Tải mô hình đã lưu
model = models.convnext_tiny(pretrained=False)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Hàm dự đoán
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

# Lớp 0 là hữu cơ, lớp 1 là vô cơ
    if class_name == 'O':
        #print('The image is Organic Waste')
        return 0
    elif class_name == 'R':
        #print('The image is Inorganic Waste')
        return 1
        
from PIL import Image
# Hàm load ảnh và tiền xử lý
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
# Bảng phân biệt nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Báo cáo phân loại
report = classification_report(y_test, y_pred, target_names=['Vô cơ', 'Hữu cơ'])
print('Classification Report:')
print(report)
```
![alt text](https://i.ibb.co/d06cngYh/convnext.png)

## 2.3 Example of ConvneXt model Prediction
```python
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
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

# Set device to CPU
device = torch.device('cpu')

model_path  = '/kaggle/input/convnext98/tensorflow2/default/1/convnext_rac_model_98.pth'

# Dataset path
train_dir = "/kaggle/input/waste-classification-data/DATASET/TRAIN/"
test_dir = "/kaggle/input/waste-classification-data/DATASET/TEST/"

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

# load model
model_loaded = models.convnext_tiny(pretrained=False)
model_loaded.classifier[2] = nn.Linear(model_loaded.classifier[2].in_features, 2)
model_loaded.load_state_dict(torch.load(model_path, map_location='cpu'))  
model_loaded = model_loaded.to(device)
model_loaded.eval()

# predict
def predict_fun(image_path, model):
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
    if class_name == 'R':
        return 'Inorganic Waste'
    if class_name == 'O':
        return 'Organic Waste'
   
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

all_time = 0  # Initialize timer variable

for i in range(6):
    # Sample image path
    img_path = f'/kaggle/input/waste-classification-data/DATASET/TEST/O/O_{12568 + i}.jpg'
    start = time.time()
    prediction = predict_fun(img_path, model_loaded)
    end = time.time()
    time_process = end - start
    if i!=0:
        all_time += time_process
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
    img_path = f'/kaggle/input/waste-classification-data/DATASET/TEST/R/R_{10400 + i}.jpg'
    start = time.time()
    prediction = predict_fun(img_path, model_loaded)
    end = time.time()
    time_process = end - start
    if i!=0:
        all_time += time_process
    print(f"Time process: {time_process} s")
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(prediction)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

print(f"Average processing time per image: {all_time/10} s")
```
### a. Kaggle
![alt text](https://i.ibb.co/r2xmtRwm/convnext-predict.png)
### b. Raspberrry Pi 5
![alt text](https://i.ibb.co/5xt3crn2/convnext-log.png)

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
model = load_model('/kaggle/input/vgg16_model/tensorflow2/default/1/vgg16_O_R_classifier.keras')

# Thư mục chứa dữ liệu test
folder_O = '/kaggle/input/waste-classification-data/DATASET/TEST/O'
folder_R = '/kaggle/input/waste-classification-data/DATASET/TEST/R'

# Hàm image
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
report = classification_report(y_test, y_pred, target_names=['Inorganic', 'Organic'])
print('Classification Report:')
print(report)
```
![alt text](https://i.ibb.co/yngwgjBp/vgg16.png)
## 3.2 Example of VGG16 model Prediction
```python
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

model = load_model('/kaggle/input/vgg16_model/tensorflow2/default/1/vgg16_O_R_classifier.keras')

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
axes = axes.flatten()  # Convert to 1D for easier handling
all_time = 0

for i in range(6):
    # Create figure to display
    img_path = '/kaggle/input/waste-classification-data/DATASET/TEST/O/O_' + str(12568+i) + '.jpg'
    start = time.time()
    prediction = predict_fun(img_path)
    end = time.time()
    time_process = end - start
    if i != 0:
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
axes = axes.flatten()  # Convert to 1D for easier handling

for i in range(6):
    # Create figure to display
    img_path = '/kaggle/input/waste-classification-data/DATASET/TEST/R/R_' + str(10400+i) + '.jpg'
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
```
### a. Kaggle
![alt text](https://i.ibb.co/6JYvnJ87/vgg16-predict.png)
### b. Raspberry Pi 5
![alt text](https://i.ibb.co/DfPyG5Dj/vgg.png)
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
model = load_model('/kaggle/input/mobilenet/tensorflow2/default/1/my_model_mobilenetv2.h5')

# test folders
folder_O = '/kaggle/input/waste-classification-data/DATASET/TEST/O'
folder_R = '/kaggle/input/waste-classification-data/DATASET/TEST/R'

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
report = classification_report(y_test, y_pred, target_names=['Vô cơ', 'Hữu cơ'])
print('Classification Report:')
print(report)
```

![alt text](https://i.ibb.co/8DmBzh0J/mobilenetv2.png)

## 4.2. Example of MobileNetV2 model Prediction
```python
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

model = load_model('/kaggle/input/mobilenet/tensorflow2/default/1/my_model_mobilenetv2.h5')

def predict_fun(img_path):
    # Upload and process images
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x / 255.0  # Apply same rescale step
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
axes = axes.flatten()  # Convert to 1D for easier handling
all_time = 0

for i in range(6):
    # Create figure to display
    img_path = '/kaggle/input/waste-classification-data/DATASET/TEST/O/O_' + str(12568+i) + '.jpg'
    start = time.time()
    prediction = predict_fun(img_path)
    end = time.time()
    time_process = end - start
    if i !=0:
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
    img_path = '/kaggle/input/waste-classification-data/DATASET/TEST/R/R_' + str(10400+i) + '.jpg'
    start = time.time()
    prediction = predict_fun(img_path)
    end = time.time()
    time_process = end - start
    if i !=0:
        all_time+= time_process
    print(f"Time process: {time_process} s")
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(prediction)
    axes[i].axis('off')  

plt.tight_layout()
plt.show()

print(f"Average processing time per image: {all_time/10} s")
```
### a. Kaggle
![alt text](https://i.ibb.co/N68tcJN8/mobilenet-predict.png)
### b. Raspberry Pi 5
![alt text](https://i.ibb.co/fVKqXQpk/mobilenet.png)
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

# Load model
model = load_model('/kaggle/input/resnet50/tensorflow2/default/1/my_model_resnet50.h5')

# test folder
folder_O = '/kaggle/input/waste-classification-data/DATASET/TEST/O'
folder_R = '/kaggle/input/waste-classification-data/DATASET/TEST/R'

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
    # Resize  (224x224)
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

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# report
report = classification_report(y_test, y_pred, target_names=['Vô cơ', 'Hữu cơ'])
print('Classification Report:')
print(report)
```
![alt text](https://i.ibb.co/mrjWX7kD/resnet.png)

## 4.2 Example of ResNet Model Prediction
```python
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

model = load_model('/kaggle/input/resnet50/tensorflow2/default/1/my_model_resnet50.h5')

def predict_fun(img_path):
    # Upload and process images
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x / 255.0  # Apply the same rescale step
    x = np.expand_dims(x, axis=0)  # Add batch size
    
    # predict
    prediction = model.predict(x)
    
    # Class determination based on threshold 0.5
    if prediction[0][0] >= 0.5:
        print("The predicted image is INORGANIC")
        return 'Recyclable Waste'
    else:
        print("The predicted image is ORGANIC")
        return 'Organic Waste'
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()  # Convert to 1D for easier handling
all_time = 0

for i in range(6):
    # Create figure to display
    img_path = '/kaggle/input/waste-classification-data/DATASET/TEST/O/O_' + str(12568+i) + '.jpg'
    start = time.time()
    prediction = predict_fun(img_path)
    end = time.time()
    time_process = end - start
    if i != 0:
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
axes = axes.flatten()  # Convert to 1D for easier handling

for i in range(6):
    # Create figure to display
    img_path = '/kaggle/input/waste-classification-data/DATASET/TEST/R/R_' + str(10400+i) + '.jpg'
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
```
### a. Kaggle
![alt text](https://i.ibb.co/b50jKwvD/resnet-predict.png)
### b. Raspberry Pi
![alt text](https://i.ibb.co/XkrtcjPL/resnet.png)

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

model = load_model('/kaggle/input/effectlite/tensorflow2/default/1/my_model_effectlite.h5')

folder_O = '/kaggle/input/waste-classification-data/DATASET/TEST/O'
folder_R = '/kaggle/input/waste-classification-data/DATASET/TEST/R'

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
report = classification_report(y_test, y_pred, target_names=['Inorganic', 'organic'])
print('Classification Report:')
print(report)
```
![alt text](https://i.ibb.co/rfG45dg3/Efficient-Net.png)

## 5.2. Example of EfficientNet Model Prediction
```python
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
model = load_model('/kaggle/input/effectlite/tensorflow2/default/1/my_model_effectlite.h5')

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
    img_path = '/kaggle/input/waste-classification-data/DATASET/TEST/O/O_' + str(12568+i) + '.jpg'
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
    img_path = '/kaggle/input/waste-classification-data/DATASET/TEST/R/R_' + str(10400+i) + '.jpg'
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
```
### a. Kaggle
![alt text](https://i.ibb.co/Mk7g0GMF/Efficient-Net-predcit.png)
### b. Raspberry Pi 5
![alt text](https://i.ibb.co/ZRfvFqrL/Effect-Lite.png)

# 6. Evaluation of Models.
## 6.1 Confusion Matrix of Models
```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

# Define the confusion matrices
cmats = [
    np.array([[1331, 70], 
              [165, 947]]),   # CNN
    np.array([[1385, 16],
              [43, 1069]]), #ConvNext
    np.array([[1303, 98],
              [224, 888]]), #VGG16
    np.array([[1357, 44],
              [207, 905]]), #MobileNetV2
    np.array([[1357, 44],
              [207, 905]]), #RESNet
    np.array([[1369, 32],
              [244, 868]]) #EffectLiteNet
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
![alt text](https://i.ibb.co/G3CN7J1K/bieu-do-confusion-1.png)

## 6.2. Comparison of Models
```python
import numpy as np

# Ma trận nhầm lẫn của các mô hình
cmats = [
    np.array([[1331, 70], 
              [165, 947]]),   # CNN
    np.array([[1385, 16],
              [43, 1069]]), #ConvNext
    np.array([[1303, 98],
              [224, 888]]), #VGG16
    np.array([[1357, 44],
              [207, 905]]), #MobileNetV2
    np.array([[1357, 44],
              [207, 905]]), #RESNet
    np.array([[1369, 32],
              [244, 868]]) #EffectLiteNet
]

def tinh_mdr_fdr(cmats):
    results = []
    for idx, cm in enumerate(cmats):
        total = np.sum(cm)
        sai = np.sum(cm) - np.trace(cm)
        false_positive = cm[0,1]  # Dự đoán sai là false positive
        total_sai = sai
        total_fp = false_positive

        # Tính MDR
        mdr = sai / total if total > 0 else 0

        # Tính FDR
        fdr = false_positive / sai if sai > 0 else 0
        results.append({
            'model_index': idx + 1,
            'MDR': mdr,
            'FDR': fdr
        })
    return results

# calculate MDR and FDR 
ket_qua = tinh_mdr_fdr(cmats)

# print results
for res in ket_qua:
    print(f"Model {res['model_index']}: MDR = {res['MDR']:.4f}, FDR = {res['FDR']:.4f}")
```
![alt text](https://i.ibb.co/QvzzRyJ1/evaluation2.png)

```python
import pandas as pd
import matplotlib.pyplot as plt

# dataset of models
models = [
    {
        'name': 'PQ-CNN',
        'size': 22.28,
        'accuracy': 91.0,
        'speed_ras': 84.32,
        'speed_kaggle':  55.0,
        'MDR': 0.0935,
        'FDR': 0.2979
    },
    {
        'name': 'ConvNeXt',
        'size': 111.36,
        'accuracy': 97.0,
        'speed_ras': 378.0,
        'speed_kaggle': 68.0,
        'MDR': 0.0235,
        'FDR': 0.2712
    },
    {
        'name': 'VGG16',
        'size': 273.56,
        'accuracy': 87.0,
        'speed_ras': 630.6,
        'speed_kaggle': 85.5,
        'MDR': 0.1281,
        'FDR': 0.3043
    },
    {
        'name': 'MobileNetV2',
        'size': 9.6,
        'accuracy': 90.0,
        'speed_ras': 157.6,
        'speed_kaggle': 78.4,
        'MDR': 0.0999,
        'FDR': 0.1753
    },
    {
        'name': 'ResNet',
        'size': 9.58,
        'accuracy': 90.0,
        'speed_ras': 151.4,
        'speed_kaggle': 77.6,
        'MDR': 0.0999,
        'FDR': 0.1753
    },
    {
        'name': 'EfficientNet',
        'size': 244.22,
        'accuracy': 89.0,
        'speed_ras': 333.0,
        'speed_kaggle': 85.6,
        'MDR': 0.1098,
        'FDR': 0.1159
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
![alt text](https://i.ibb.co/0ypQqYt4/bieu-do-mo-hinh-6.png)
```python
import pandas as pd

# dataset of models
models = [
    {
        'name': 'PQ-CNN',
        'size': 22.28,
        'accuracy': 91.0,
        'speed_ras': 84.32,
        'speed_kaggle':  55.0,
        'MDR': 0.0935,
        'FDR': 0.2979
    },
    {
        'name': 'ConvNeXt',
        'size': 111.36,
        'accuracy': 97.0,
        'speed_ras': 378.0,
        'speed_kaggle': 68.0,
        'MDR': 0.0235,
        'FDR': 0.2712
    },
    {
        'name': 'VGG16',
        'size': 273.56,
        'accuracy': 87.0,
        'speed_ras': 630.6,
        'speed_kaggle': 85.5,
        'MDR': 0.1281,
        'FDR': 0.3043
    },
    {
        'name': 'MobileNetV2',
        'size': 9.6,
        'accuracy': 90.0,
        'speed_ras': 157.6,
        'speed_kaggle': 78.4,
        'MDR': 0.0999,
        'FDR': 0.1753
    },
    {
        'name': 'ResNet',
        'size': 9.58,
        'accuracy': 90.0,
        'speed_ras': 151.4,
        'speed_kaggle': 77.6,
        'MDR': 0.0999,
        'FDR': 0.1753
    },
    {
        'name': 'EfficientNet',
        'size': 244.22,
        'accuracy': 89.0,
        'speed_ras': 333.0,
        'speed_kaggle': 85.6,
        'MDR': 0.1098,
        'FDR': 0.1159
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
![alt text](https://i.ibb.co/VYztZM6p/table.png)

## License

[MIT](https://choosealicense.com/licenses/mit/)
