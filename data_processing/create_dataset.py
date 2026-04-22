import os
import pandas as pd
import random

IMAGE_DIR = "data/raw/images/train"
SENSOR_FILE = "data/processed_sensor.csv"

sensor_df = pd.read_csv(SENSOR_FILE)

data = []

for label in os.listdir(IMAGE_DIR):
    label_path = os.path.join(IMAGE_DIR, label)

    if not os.path.isdir(label_path):
        continue

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)

        # randomly pick sensor row
        sensor_row = sensor_df.sample(1).iloc[0]

        wave_height = sensor_row['wave_height']
        wind_speed = sensor_row['wind_speed']

        data.append([img_path, wave_height, wind_speed, label])

df = pd.DataFrame(data, columns=['image_path', 'wave_height', 'wind_speed', 'label'])

# Save in ROOT (same level as data folder)
df.to_csv("dataset.csv", index=False)

print("✅ Dataset created with multimodal data")