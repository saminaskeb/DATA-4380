import os
import pandas as pd

data_path = "data/raw"
files = os.listdir(data_path)

data = []
for file in files:
    try:
        age, gender, race, _ = file.split("_", 3)
        data.append({"filename": file, "race": int(race)})
    except:
        continue

df = pd.DataFrame(data)
df.head()

valid_classes = [0, 1, 2, 3, 4]  # White, Black, Asian, Indian, Others
df = df[df['race'].isin(valid_classes)]

subset_df = df.groupby("race").apply(lambda x: x.sample(n=min(100, len(x)), random_state=42))
subset_df.reset_index(drop=True, inplace=True)

import os

output_dir = "data/organized"
os.makedirs(output_dir, exist_ok=True)

# Create a folder for each class (0–4)
for race_id in valid_classes:
    class_dir = os.path.join(output_dir, str(race_id))
    os.makedirs(class_dir, exist_ok=True)


import shutil

raw_path = "data/raw"

for _, row in subset_df.iterrows():
    src = os.path.join(raw_path, row['filename'])
    dst = os.path.join(output_dir, str(row['race']), row['filename'])
    shutil.copy(src, dst)

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
)

train_generator = train_datagen.flow_from_directory(
    'data/organized',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    'data/organized',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)



