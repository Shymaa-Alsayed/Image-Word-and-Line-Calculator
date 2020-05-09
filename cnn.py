from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import pickle
#import cv2
#import numpy as np

# Initialize network
cnn=Sequential()

# Convolution
cnn.add(Convolution2D(32,3,3,input_shape=(32,32,1),activation='relu'))

# Max Pooling
cnn.add(MaxPooling2D(pool_size=(2,2)))

# Flattening
cnn.add(Flatten())

# Full Connection
cnn.add(Dense(output_dim=128,activation='relu'))

# Output layer
cnn.add(Dense(output_dim=62,activation='softmax'))

# Compiling cnn
cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Fitting cnn to dataset of images
'''def read_one_channel(img):
        proc_img=cv2.imread(img,cv2.IMREAD_UNCHANGED)
        tensor_proc_img=np.reshape(proc_img,(32,32,1))
        return tensor_proc_img'''

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training = train_datagen.flow_from_directory(
        'dataset/training',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical',
        color_mode='grayscale')

testing= test_datagen.flow_from_directory(
        'dataset/testing',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical',
        color_mode='grayscale')


model=cnn.fit_generator(
        training,
        steps_per_epoch=38440,
        epochs=25,
        validation_data=testing,
        validation_steps=12772)

cnn.save('cnn.h5', model)
label_map = (training.class_indices)
pickle.dump({'label_map':label_map},open('classes','wb'))

