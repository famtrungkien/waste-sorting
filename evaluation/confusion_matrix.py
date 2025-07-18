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