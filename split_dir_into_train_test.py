import os
from shutil import copyfile

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define the dataset directory and the classes
dataset_dir = 'images'
classes = os.listdir(dataset_dir)

target_dataset_dir = 'image_dataset'
# Create a train and test directory in the dataset directory
train_dir = os.path.join(target_dataset_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
test_dir = os.path.join(target_dataset_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# Split the dataset into train and test sets
for class_name in classes:
    class_dir = os.path.join(dataset_dir, class_name)
    images = os.listdir(class_dir)
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
    print(f'Copying train images for class : {class_name}')
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    for image in tqdm(train_images):
        copyfile(os.path.join(class_dir, image), os.path.join(train_dir, class_name, image))

    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    print(f'Copying test images for class : {class_name}')
    for image in tqdm(test_images):
        copyfile(os.path.join(class_dir, image), os.path.join(test_dir, class_name, image))