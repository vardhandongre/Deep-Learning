#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 23:00:36 2020

@author: don
reference = https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/
"""

# packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# Argument Parser

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help='path to input datset')
ap.add_argument("-p","--plots",type=str, default='plot.png',help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="covid19.model", help="path to output model")
args = vars(ap.parse_args())

# initialize the hyper-parameters
# initial LR
init_lr = 1e-3
epochs = 25
bs = 8

# Loading the dataset 
print('[INFO] loading images')
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

#loop over the image paths
for imagePath in imagePaths:
    # extract the class label from filename
    label = imagePath.split(os.path.sep)[-2]
    
    # load the image, swap the color channels, resize it to be fixed
    # size = 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    
    # update the data and labels
    data.append(image)
    labels.append(label)
    
# converting the data into numpy arrays and then scaling the pixels
# intensities to range [0,1]
data = np.array(data)/255
labels = np.array(labels)

# One hot encoding the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# train-test split (80%-20%)
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, 
                                                stratify=labels, random_state=42)

# Data Augmentation

trainAug = ImageDataGenerator(
    rotation_range = 15,
    fill_mode = 'nearest')

# Here we are using the VGG16 network as our backbone/base model, and we will be constructing a 
# head of this model to be place on top of this base model

baseModel = VGG16(weights='imagenet', include_top=False, 
                   input_tensor=Input(shape=(224,224,3)))

# head of the model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4,4))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(64, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation='softmax')(headModel)

# Model
model = Model(inputs = baseModel.input, outputs = headModel)

# Freeze the layers of the base model to prevent updates 
for layer in baseModel.layers:
    layer.trainable = False
    
print('[INFO] compiling model....')

opt = Adam(lr=init_lr, decay=init_lr/epochs)
model.compile(loss='binary_crossentropy', optimizer=opt,
              metrics=['accuracy'])

# Training
print('[INFO] training head model....')
H = model.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=bs),
    steps_per_epoch=len(trainX)//bs,
    validation_data=(testX,testY),
    validation_steps = len(testX)//bs,
    epochs=epochs)

# Evaluate mode
print('[INFO] evaluating network')
predIDxs = model.predict(testX, batch_size=bs)

# for each image in the testing set we need to find the index of label
# with corresponding largest predicted probability
predIDxs = np.argmax(predIDxs,axis=1)

# Formatted Classification report
print(classification_report(testY.argmax(axis=1), predIDxs,
                            target_names=lb.classes_))

# Create the confusion matrix which can be used to derive raw accuracy,
# sensitivity and specificity 
cm = confusion_matrix(testY.argmax(axis=1), predIDxs)

total = sum(sum(cm))

acc = (cm[0,0]+cm[1,1])/total
sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
specificity = cm[1,1]/(cm[1,0]+cm[1,1])

# show the confusion matrix, accuracy, sensitivity and specificity 
print(cm)
print('acc: {:.4f}'.format(acc))
print('sensitivity: {:.4f}'.format(sensitivity))
print('specificity: {:.4f}'.format(specificity))


# Plots
N = epochs
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0,N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0,N), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0,N), H.history['val_accuracy'], label='val_accuracy')
plt.title('Training Loss and Acuracy on COVID19 Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='best')
plt.savefig('plot.pdf')


# serialize the model to disc
print('[INFO] saving the model....')
model.save(args['model'], save_format='h5')








