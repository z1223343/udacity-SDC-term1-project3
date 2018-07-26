import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split

# load data
path = './data/'
lines = []
with open(path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
# Split data into training set and validation set
train_lines,validation_lines = train_test_split(lines,test_size=0.2)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Conv2D
from keras.layers import Cropping2D
from keras.layers import MaxPooling2D
import matplotlib.pyplot as plt
import sklearn
import cv2
import os
import sys

# correction factor
steering_correction = 0.2

# Generator
def generator(samples, batch_size=32):

    num_samples = len(samples)
    #print (num_samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            all_images = []
            steering_measurements = []
            for batch_sample in batch_samples:
                filename = os.path.basename(batch_sample[0])
                image = cv2.imread(path+'IMG/'+filename)
                all_images.append(image)
                all_images.append(cv2.flip(image,1))
                measurement = float(batch_sample[3])
                steering_measurements.append(measurement)
                steering_measurements.append(measurement * -1.0)
                filename = os.path.basename(batch_sample[1])
                image = cv2.imread(path+'IMG/'+filename)
                all_images.append(image)
                steering_measurements.append(measurement+steering_correction)
                all_images.append(cv2.flip(image, 1))
                steering_measurements.append((measurement+steering_correction)*-1)
                filename = os.path.basename(batch_sample[2])
                image = cv2.imread(path+'IMG/'+filename)
                all_images.append(image)
                steering_measurements.append(measurement-steering_correction)
                all_images.append(cv2.flip(image,1))
                steering_measurements.append((measurement-steering_correction)*-1)
                #print( batch_size, offset, num_samples)

            X_train = np.array(all_images)
            # Convert the image from BGR to RGB format using [...,::-1]
            #X_train = X_train[...,::-1]
            y_train = np.array(steering_measurements)
            yield sklearn.utils.shuffle(X_train,y_train)
			
# Define the generator function
train_generator = generator(train_lines,batch_size=1)
validation_generator = generator(validation_lines,batch_size=1)

# Define the model
model = Sequential()
# Preprocess the data
model.add(Cropping2D(cropping=((50,20),(0,0)),input_shape=(160,320,3)))
model.add(Lambda(lambda x:(x/255.0)-0.5,input_shape=(90,320,3)))

model.add(Conv2D(24,(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(36,(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(48,(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(64,(3,3),strides=(1,1),activation='relu'))
model.add(Conv2D(64,(3,3),strides=(1,1),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Dropout(0.25))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
					samples_per_epoch=6*len(train_lines),
                    validation_data=validation_generator,
					nb_val_samples=6*len(validation_lines),
                    nb_epoch=10)

model.save('model.h5')

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set','validation set'],loc='upper right')
plt.show
