import pandas as pd

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