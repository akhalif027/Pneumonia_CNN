import tensorflow as tf
import numpy as np
import pandas as pd 
import cv2 
import os 
import matplotlib.pyplot as plt 
from tensorflow import keras 
from PIL import Image

# Data Visualization
normal_image = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0115-0001.jpeg')
plt.imshow(normal_image)
plt.title('Normal Lungs Sample #1')
plt.show()

pneumonia_image1 = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1000_bacteria_2931.jpeg')
plt.imshow(pneumonia_image1)
plt.title('Pneumonia Lungs Sample #1')
plt.show()

# Data Preprocessing

print(f'Normal Lungs #1 Shape: {normal_image.shape}')
print(f'Normal Lungs #2 Shape {normal_image2.shape}')

normal_images = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/')
pneumonia_images = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/')

# Normalizing the dimensions of the dataset's images. 
reprocessed_images = []
def image_reprocessing(path, images):
    for image in images:
        image = Image.open(path + image)
        image = image.resize((128,128))
        image = image.convert('RGB')
        image = np.array(image)
        image = image / 255.0

        reprocessed_images.append(image)

new_normal = image_reprocessing(('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/'),normal_images)
new_pneumonia = image_reprocessing(('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/'),pneumonia_images)

# Checking  if all images are present and the standard dimension is used. 
len(reprocessed_images)
print(f' Adjusted image shape : {reprocessed_images[0].shape}')

# Preparing images for model training via one-hot encoding and array convertion. 
normal_training_labels = [0] * len(normal_images)
pneumonia_training_labels = [1] * len(pneumonia_images)
training_labels = normal_training_labels + pneumonia_training_labels 
len(training_labels)

X = np.array(reprocessed_images)
Y = np.array(training_labels)

