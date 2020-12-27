from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import pickle
from keras import regularizers
#import cv2
#import numpy as np

# Initialize network
cnn=Sequential()

# Convolution
cnn.add(Conv2D(filters=32,kernel_size=3,input_shape=(32,32,1),activation='relu',data_format="channels_last",kernel_regularizer=regularizers.l2(1e-5)))

# Max Pooling
cnn.add(MaxPooling2D(pool_size=(2,2)))

# Flattening
cnn.add(Flatten())

# Full Connection
cnn.add(Dense(128,activation='relu',kernel_regularizer=regularizers.l2(1e-5)))

# Output layer
cnn.add(Dense(62,activation='softmax'))

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
        batch_size=1,
        class_mode='categorical',
        color_mode='grayscale')


model=cnn.fit_generator(
        training,
        steps_per_epoch=38440,
        epochs=1,
        validation_data=testing,
        validation_steps=12772)

cnn.save('cnnv2.h5', model)
label_map = (training.class_indices)
pickle.dump({'label_map':label_map},open('classes','wb'))

