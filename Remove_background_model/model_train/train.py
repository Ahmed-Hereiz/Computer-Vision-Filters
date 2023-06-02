from ImageDataProcessor import ImageDataHandler
from HandleImagesFunctions import *

url = 'https://storage.googleapis.com/kaggle-data-sets/3145890/5442439/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230522%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230522T183243Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=a7980ef172d30c9f39a2b6836a763795842ee19534d544e51bb358fe824060f082e74f82ffef74e2c7b61e286149bac5fcafe24c3cdcae9c42f715d215b53ef34f15ee39740c1880e8e8f6a6df23464a3ce242a70700236485d0f9ca0a68b08024c502ac841b0292375884d35ce4a987a976531a29544ffa3167647c1d67df3091cb6f900bf08502b70d16d4d3816e49f72d7ae71a3930d570edc40303c572389f4d02a6c4be3605d0c4a7a84a6af3bbe714e43fcf7e2384897c0db6636ccbeae4504f94657327d1f45b903ed966fb380d60fb379a4296fb647f897d4fef0273d7182370f94becd7cefbab31826ff72a941275ad7f55755f7f82dc83e0b92314'



data_processor = ImageDataHandler(url=url)
data_processor.download_data()

json_file_path = '/content/datasets/People/annotations/instances.json'
output_folder = '/content/datasets/People/labels'


convert_annotations(json_file_path, output_folder)

import os

# Directory containing the images
image_dir = '/content/datasets/People/images'

for filename in os.listdir(image_dir):
    if filename.endswith(('.png', '.jpeg')):
        new_filename = os.path.splitext(filename)[0] + '.jpg'
        os.rename(os.path.join(image_dir, filename), os.path.join(image_dir, new_filename))

image_dir = '/content/datasets/People/images'

extensions = set()
for filename in os.listdir(image_dir):
    if os.path.isfile(os.path.join(image_dir, filename)):
        ext = os.path.splitext(filename)[1].lower()
        if ext.startswith('.'):
            ext = ext[1:]
        extensions.add(ext)

# Print unique extensions
print("Unique image extensions:")
for ext in extensions:
    print(ext)

labels_dir = '/content/datasets/People/labels'  # Directory containing label files
images_dir = '/content/datasets/People/images'  # Directory containing image files

normalize_label_coordinates(labels_dir, images_dir)

image_folder = '/content/datasets/People/images'
label_folder = '/content/datasets/People/labels'
output_folder = '/content/datasets/People/data'
train_ratio = 0.8 

split_data(image_folder, label_folder, train_ratio, output_folder)

train_dir = '/content/datasets/People/data/train/images'
val_dir = '/content/datasets/People/data/val/images'





