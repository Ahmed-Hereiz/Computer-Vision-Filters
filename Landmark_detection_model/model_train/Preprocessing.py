import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from ImageDataProcessor import ImageDataHandler


url = 'https://storage.googleapis.com/kaggle-data-sets/2598/4327/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230526%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230526T031831Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3e9ffc227bc790fd9b1c9fe9a22fef19523bde5a71646c0d13ec5fb8dd74a4b5f321c24edc6d67ced0509089b40a4c656c421ebc74147f36faacb87a6c25e824fcc7c3bca6ba4cbb472d46be585455bf29c2c44fc7d5cb8b50350ec4365bbc2490bd692027e58c85445298d036642041620f8b949b6edf025e15093cb2f293c40f3b622ed7eb11180c8d10f20ad03b8d69a9c0cf62d4b3362e034537675f7f066d713eadceb79f957d594fde896e6b43b6d207ec2519ee76b43201a956217fe75656a9b1aac179e9ffa7ab7e0a9ea7d09b698b8d3df19e2905c8c6ed3216e8798fff12b39dba9087df0889ca8111c4334d2eff85cfc8a5049752d1b28d29f8e3'

data_processor = ImageDataHandler(url=url)
data_processor.download_data()


data = np.load('/content/datasets/face_images.npz', allow_pickle=True)
images = data['face_images']
keypoints = pd.read_csv('/content/datasets/facial_keypoints.csv')

data.close()

keypoints = keypoints[['left_eye_center_x','left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y',
                      'nose_tip_x', 'nose_tip_y']]


images = np.swapaxes(np.swapaxes(images, 1, 2), 0, 1)
images = images/255.0

keypoints.fillna(keypoints.mean(),inplace=True)
keypoints = np.array(keypoints)

images = images.reshape(7049, 96, 96, 1)
keypoints = keypoints.reshape(-1, 6)


X_train, X_test, y_train, y_test = train_test_split(images, keypoints, test_size=0.2, random_state=42)
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)