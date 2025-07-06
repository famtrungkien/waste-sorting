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

model_path  = 'convnext_rac_model.pth'

# Dataset path
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
    img_path = f'DATASET/TEST/O/O_{12568 + i}.jpg'
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
    img_path = f'DATASET/TEST/R/R_{10000 + i}.jpg'
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
