import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import random
import zipfile
import shutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

class ImageDataHandler:
    """
    A utility class for handling image data, including downloading, splitting into train and test sets, 
    and augmenting data using Keras' ImageDataGenerator.

    Parameters:
    -----------
    url : str
        URL to the data zip file to be downloaded.
    cache_dir : str, optional
        Directory to cache the downloaded data, defaults to current working directory.

    Methods:
    --------
    download_data():
        Downloads the data zip file from the given URL and extracts it into a 'datasets' folder.

    split_data(data_dir: str, train_dir: str, test_dir: str, test_size: float):
        Splits the data in the `data_dir` directory into training and testing sets, and saves them in `train_dir`
        and `test_dir` directories respectively. `test_size` is the proportion of data to be used for testing, 
        defaults to 0.2.

    data_augment(train_dir: str, rotation_range: int, width_shift_range: float, height_shift_range: float, 
                 shear_range: float, zoom_range: float):
        Augments the training data in the `train_dir` directory using Keras' ImageDataGenerator. `rotation_range`,
        `width_shift_range`, `height_shift_range`, `shear_range`, and `zoom_range` are the ranges for random image
        transformations, as defined by Keras' ImageDataGenerator.

    delete_folder(folder_path: str):
        Deletes the folder at the given `folder_path`.
    """
    def __init__(self, url, cache_dir='.'):
        
        self.url = url
        self.cache_dir = cache_dir
        
    def download_data(self):
        
        data_path = tf.keras.utils.get_file("data.zip", self.url, cache_dir=self.cache_dir)
        with zipfile.ZipFile(data_path, 'r') as zip_ref:
            zip_ref.extractall('datasets')
            
    def split_data(self, data_dir, train_dir, test_dir, test_size=0.2):

        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)

            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)

            # Create the class folders in the new data directories
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # Loop through the images in the class folder
            image_paths = [os.path.join(class_dir, img_name) for img_name in os.listdir(class_dir)]
            train_paths, test_paths = train_test_split(image_paths, test_size=test_size, random_state=42)

            # Copy the training images to the training directory
            for path in train_paths:
                filename = os.path.basename(path)
                dest_path = os.path.join(train_class_dir, filename)
                shutil.copy2(path, dest_path)

            # Copy the testing images to the testing directory
            for path in test_paths:
                filename = os.path.basename(path)
                dest_path = os.path.join(test_class_dir, filename)
                shutil.copy2(path, dest_path)
                
    def data_augment(self, train_dir, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2):
        
        class_names = os.listdir(train_dir)
        num_images = {}
        for class_name in class_names:
            num_images[class_name] = len(os.listdir(os.path.join(train_dir, class_name)))
        print(f'Number of images in each class: {num_images}')

        # Set the target number of images for each class
        target_num = max(num_images.values())

        # Define an ImageDataGenerator for data augmentation
        data_gen = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=True,
            fill_mode='nearest')

        # Loop through each class and augment the images
        for class_name in class_names:
            # Calculate the number of images to generate
            num_to_generate = target_num - num_images[class_name]
            if num_to_generate <= 0:
                continue
            print('Number of Images needed to be generated : ',num_to_generate)

            # Set the path to the images in the current class
            class_path = os.path.join(train_dir, class_name)
            print('images needs to be generated in : ',class_path)

            # Loop through the images in the current class and generate new images
            for i, img_name in enumerate(os.listdir(class_path)):
                # Load the image and convert it to a numpy array
                img_path = os.path.join(class_path, img_name)
                img = load_img(img_path, target_size=(224, 224))
                x = img_to_array(img)

                # Generate new images
                for j in range(num_to_generate):
                    # Apply random transformations to the image
                    params = data_gen.get_random_transform(x.shape)
                    x_aug = data_gen.apply_transform(x, params)

                    # Save the new image
                    new_img_name = f'{class_name}_{i}_{j}.JPG'
                    new_img_path = os.path.join(class_path, new_img_name)
                    img = array_to_img(x_aug)
                    img.save(new_img_path)

                # Update the number of images in the current class
                num_images[class_name] += num_to_generate
                break

        print(f'Number of images in each class after data augmentation: {num_images}')
        
    def delete_folder(self, folder_path):
        
        shutil.rmtree(folder_path)
        
            
class ImageGenerator:
    """A class for creating image data generators using Keras ImageDataGenerator.

    Attributes:
        train_dir (str): Path to the directory containing the training images.
        test_dir (str): Path to the directory containing the testing images.
        img_height (int): The desired height of the input images.
        img_width (int): The desired width of the input images.
        batch_size (int): The batch size to use for training and testing.

    Methods:
        
    create_train_generator(self):
        Returns the data generator for training images.
        
    create_val_generator(self):
        Returns the data generator for validation images.
        
    create_test_generator(self):
        Returns the data generator for testing images.
    """
    def __init__(self, train_dir, test_dir, img_height=224, img_width=224, batch_size=32):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        
    def create_data_generators(self):
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        # create train and test data generators
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                self.train_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                class_mode='binary')

        val_generator = datagen.flow_from_directory(
                self.train_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                class_mode='binary',
                subset='validation')

        test_generator = test_datagen.flow_from_directory(
                self.test_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                class_mode='binary')
        
        return train_generator, val_generator, test_generator

    
class ImagePlotter:
    """
    Attributes:
        generator: An instance of `tf.keras.preprocessing.image.ImageDataGenerator` used 
    for generating batches of images.
    Methods:
        show_images(num_images): Displays a batch of `num_images` images 
    and
     their corresponding labels.
    """
    def __init__(self, train_generator):
        self.train_generator = train_generator
        self.class_names = list(train_generator.class_indices.keys())
        
    def plot_images(self):
        # Get a batch of images and their corresponding labels from the generator
        images, labels = next(self.train_generator)

        # Plot the images and their corresponding labels
        fig, axes = plt.subplots(6, 5, figsize=(20, 20))
        axes = axes.ravel()
        for i in np.arange(0, 30):
            axes[i].imshow(images[i])
            axes[i].set_title(self.class_names[int(labels[i])], color='r')
            axes[i].axis('off')

        plt.subplots_adjust(wspace=0.01)
        plt.show()