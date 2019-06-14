#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:17:23 2019

@author: admin
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import to_categorical
from keras.regularizers import *
import matplotlib.pyplot as plt
from keras.applications import VGG19
from keras.backend import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint

bestValidationCheckpointer = ModelCheckpoint('train_model_VGG_19.hdf5', 
                                             monitor='val_categorical_accuracy', 
                                             save_best_only=True, verbose=1)
train_datagen = ImageDataGenerator(
                                   rescale = 1./255,
                                   zoom_range = [1,1.2], 
                                   rotation_range = 15, 
                                   height_shift_range = 0.4, 
                                   width_shift_range =0.4,
                                   horizontal_flip = True,
                                   shear_range = 0.05,
                                   fill_mode = 'nearest'
	)
train_generator = train_datagen.flow_from_directory(
	'train-masks',
	target_size = (240,320),
	batch_size = 32,
	class_mode = 'categorical',
	color_mode = 'rgb',
	)
val_datagen = ImageDataGenerator(
	rescale = 1./255)

val_generator = val_datagen.flow_from_directory(
	'val-masks',
	target_size = (240,320),
	batch_size = 32,
	class_mode = 'categorical',
	color_mode = 'rgb')

#img_input = Input(shape=(256,256,1))
#img_conc = Concatenate()([img_input, img_input, img_input])    
base_model = VGG19(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
x = Dropoout(0.4)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(11, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer= 'Adam', loss = 'categorical_crossentropy',
              metrics = ['categorical_accuracy'])
history = model.fit_generator(train_generator,
	steps_per_epoch = 50,
	epochs = 100,
	validation_data = val_generator,
	validation_steps = 20,callbacks = [bestValidationCheckpointer])
# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('acc.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss.png')

  # save the weights
