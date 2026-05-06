import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# Paths
annotations_dir = 'data/annotations'
images_dir = 'data/images'
labels_dir = 'data/labels'
train_images_dir = 'data/train/images'
train_labels_dir = 'data/train/labels'
val_images_dir = 'data/valid/images'
val_labels_dir = 'data/valid/labels'

# Class mapping for YOLO labels
class_map = {
    'local_boat': 0,
    'foreign_boat': 1
}

# Create directories
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Convert XML to YOLO txt
for xml_file in os.listdir(annotations_dir):
    if not xml_file.endswith('.xml'):
        continue
    xml_path = os.path.join(annotations_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    
    txt_filename = filename.replace('.png', '.txt').replace('.jpg', '.txt')
    txt_path = os.path.join(labels_dir, txt_filename)
    
    with open(txt_path, 'w') as f:
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in class_map:
                raise ValueError(
                    f'Unknown class name "{name}" in {xml_file}. '
                    'Use local_boat or foreign_boat in the XML annotation.'
                )
            class_id = class_map[name]
            xmin = int(obj.find('bndbox/xmin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymin = int(obj.find('bndbox/ymin').text)
            ymax = int(obj.find('bndbox/ymax').text)
            
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

# Get list of image files
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Split into train and val (80/20)
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Move files
for file in train_files:
    shutil.copy(os.path.join(images_dir, file), os.path.join(train_images_dir, file))
    txt_file = file.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
    if os.path.exists(os.path.join(labels_dir, txt_file)):
        shutil.copy(os.path.join(labels_dir, txt_file), os.path.join(train_labels_dir, txt_file))

for file in val_files:
    shutil.copy(os.path.join(images_dir, file), os.path.join(val_images_dir, file))
    txt_file = file.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
    if os.path.exists(os.path.join(labels_dir, txt_file)):
        shutil.copy(os.path.join(labels_dir, txt_file), os.path.join(val_labels_dir, txt_file))

print("Data preparation complete.")