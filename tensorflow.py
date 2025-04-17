import tensorflow as tf
import numpy as np
import os
import cv2 
import matplotlib.pyplot as plt 
from tensorflow import keras 
from PIL import Image

#Data Visualization
normal_image = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0115-0001.jpeg')
plt.title('Normal Lungs Sample')
plt.imshow()

pneumonia_image = cv2.imread('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1000_bacteria_2931.jpeg')
plt.title('Pneumonia Lungs Sample')
plt.imshow()

#Data Preprocessing

print(f'Normal Lungs #1 Shape: {normal_image.shape}')
print(f'Normal Lungs #2 Shape {normal_image2.shape}')

normal_images = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/')
pneumonia_images = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/')
len(normal_images + pneumonia_images)

#Normalizing the dimensions of the dataset's images. 
def image_reprocessing(path, images):
    reprocessed_images = []
    for image in images:
        image = Image.open(path + image)
        image = image.resize((128,128))
        image = image.convert('RGB')
        image = np.array(image)
        image = image / 255.0
        reprocessed_images.append(image)
        
    return reprocessed_images

new_normal = image_reprocessing(('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/'),normal_images)
new_pneumonia = image_reprocessing(('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/'),pneumonia_images)

#Checking if the standard dimension is applied correctly. 
print(f' Adjusted image shape : {reprocessed_images[0].shape}')

#Preparing images for model training via one-hot encoding and array convertion. 
normal_training_labels = [0] * len(normal_images)
pneumonia_training_labels = [1] * len(pneumonia_images)
training_labels = normal_training_labels + pneumonia_training_labels 
len(training_labels)

training_imagews = new_normal + new_pneumonia
len(training_images)

X = np.array(training_images)
Y = np.array(training_labels)

#Building the model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()

model.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),1,activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),1,activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation= 'sigmoid'))
model.summary()

model.compile('adam',loss= tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

#Training the model
hist = model.fit(X, Y, validation_split=0.2, verbose=1, epochs=5)

#Plotting training accuracy and loss rate. 
plt.plot(hist.history['loss'], label= "Training Loss")
plt.plot(hist.history['val_loss'], label= "Validation Loss")
plt.legend()
plt.grid(True)
plt.title('Training Loss VS Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.plot(hist.history['accuracy'], label= 'Training Accuracy')
plt.plot(hist.history['val_accuracy'], label= 'Validation Accuracy')
plt.legend()
plt.grid(True)
plt.title('Training Accuracy VS Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

#Testing the model
#Preprocessing testing images procedure similar to training images. 
normal_images = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/')
pneumonia_images = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/')
        
new_normal = image_reprocessing(('/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/'),normal_images)
new_pneumonia = image_reprocessing(('/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/'),pneumonia_images)
testing_images = new_normal + new_pneumonia
len(testing_images)

normal_testing_labels = len(normal_images) * [0]
pneumonia_testing_labels = len(pneumonia_images) * [1]
testing_labels = normal_testing_labels + pneumonia_testing_labels
len(testing_labels)

X_test = np.array(testing_images)
Y_test = np.array(testing_labels)

loss, acc  = model.evaluate(X_test, Y_test)
print(f' Accuracy: {acc * 100:.2f}%')

#Saving the model to reuse without retraining it. 
model.save('pnuemonia_detector.h5')

