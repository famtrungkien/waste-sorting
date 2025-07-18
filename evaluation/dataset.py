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