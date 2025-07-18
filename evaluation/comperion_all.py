import pandas as pd
import matplotlib.pyplot as plt

# dataset of models
models = [
    {
        'name': 'PQ-CNN',
        'size': 22.28,
        'accuracy': 92.0,
        'speed_ras': 76.3,
        'speed_kaggle': 54.0,
        'MDR': 0.0804,
        'FDR': 0.3020
    },
    {
        'name': 'ConvNeXt',
        'size': 111.35,
        'accuracy': 98.0,
        'speed_ras': 234.3,
        'speed_kaggle': 71.6,
        'MDR': 0.0235,
        'FDR': 0.2712
    },
    {
        'name': 'VGG16',
        'size': 273.56,
        'accuracy': 89.0,
        'speed_ras': 472.2,
        'speed_kaggle': 90.4,
        'MDR': 0.1094,
        'FDR': 0.2109
    },
    {
        'name': 'MobileNetV2',
        'size': 9.6,
        'accuracy': 88.0,
        'speed_ras': 130.4,
        'speed_kaggle': 78.5,
        'MDR': 0.1150,
        'FDR': 0.1453
    },
    {
        'name': 'ResNet',
        'size': 283.44,
        'accuracy': 91.0,
        'speed_ras': 133.4,
        'speed_kaggle': 72.9,
        'MDR': 0.0887,
        'FDR': 0.1480
    },
    {
        'name': 'EfficientNet',
        'size': 244.22,
        'accuracy': 91.0,
        'speed_ras': 250.0,
        'speed_kaggle': 91.0,
        'MDR': 0.0899,
        'FDR': 0.1726
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