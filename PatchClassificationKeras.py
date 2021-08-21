# -*- coding: utf-8 -*-
"""
# **بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ**
"""
import pandas as pd

print("Bismillah")

"""# **Imports**"""

import numpy as np
import matplotlib.pyplot as plt

"""# Initializing"""

img_height = 128 #128 is being upscaled cuz resnet50 needs minimum 197
img_width = img_height
batch_size = 45
nb_epochs = 50
patience_val = 8
learning_rate = 0.0003
model_name = "DatasetSKMC_InceptionResNetV2_with_SGDM_lr0_0003"


"""# **ResNet50 Model**"""

# from keras.applications import ResNet101V2
# from keras.applications.vgg19 import VGG19
# from tensorflow.python.keras.applications.vgg19 import VGG19
# from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.layers import Flatten

# load ResNet50 model without classification layers
model = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(img_width, img_height, 3))

# add new classification layers
flat1 = Flatten()(model.layers[-1].output)  # flatten last layer
# class1 = Dense(1024, activation='relu')(flat1)  # add FC layer on previous layer
output = Dense(5, activation='softmax')(flat1)  # add softmax layer

# define the new model
model = Model(inputs=model.inputs, outputs=output)
model.summary()

"""# Data Paths"""

train_dir = "E:/MyPipeline/training_data/train/"
# train_dir = "D:/Downloads/NCT-CRC-HE-100K/Training/"

"""# On the Fly Data Load"""

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

"""Loading Training Images"""

train_datagen = ImageDataGenerator(
    rescale=1. / 255
    # , shear_range=0.2
    # , zoom_range=0.2 # zoom
     , rotation_range= 360 # rotation
    # , width_shift_range=0.2 # horizontal shift
    # , height_shift_range=0.2 # vertical shift
    # , horizontal_flip=True # horizontal flip
    # channel_shift_range = [-0.1, 0.1]
    # )
    , validation_split=0.2)  # set validation split, 0.177 for Kather

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    seed=27,
    class_mode='categorical',
    interpolation="nearest"
    , subset='training')  # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_dir,  # directory for validation data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    seed=27,
    class_mode='categorical',
    subset='validation')  # set as validation data


"""# Compile the model"""

# import tensorflow as tf

# import keras
# # from keras.optimizers import RMSprop
# initial_learning_rate = 0.01
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=7000,
#     decay_rate=0.9,
#     staircase=True)

# opt = keras.optimizers.RMSprop(learning_rate=lr_schedule)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

import tensorflow as tf
# import tensorflow.python.keras
# from keras.optimizers import SGD

# sgd = SGD(lr=0.01, decay=1e-7, momentum=.9)
# sgd = tf.keras.optimizers.SGD(
#     learning_rate=learning_rate, momentum=0.9, nesterov=True, name='SGD',
# )
sgd = tf.keras.optimizers.SGD(
    learning_rate=learning_rate, momentum=0.5, nesterov=False, name='SGD'
)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

"""# **Save Path for the outputs**"""

save_path = 'E:/MyPipeline/training_data/Results/' + model_name

"""# Train the model"""

from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience_val)

mc = ModelCheckpoint(save_path + '_best_model_val_acc_max.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

print("Training Start:")

H = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=nb_epochs,
    verbose=1,
    callbacks=[es, mc]
)

"""Saving Model weights"""

# save the model with trained weights
model.save(save_path + "_transfer_trained_wts.h5")

# model.load_weights(save_path+"_"+model_name+"_transfer_trained_wts.h5")

"""# Some Graphs"""
print(H.history)
pd.DataFrame(H.history).to_csv("E:/MyPipeline/training_data/Results/History"+model_name+".csv")

simple_acc = H.history['accuracy']
plt.plot([acc for acc in simple_acc])
plt.title('Train Accuracy for' + model_name)
plt.ylabel('Train Accuracy')
plt.xlabel('Epoch')
plt.savefig(save_path + '_Train_Acc.png')
plt.show()

simple_val_acc = H.history['val_accuracy']
plt.plot([acc for acc in simple_val_acc])
plt.title('Validation Accuracy for' + model_name)
plt.ylabel('Validation Accuracy')
plt.xlabel('Epoch')
plt.savefig(save_path + '_Validation_Acc.png')
plt.show()

simple_loss = H.history['loss']
plt.plot([los for los in simple_loss])
plt.title('Train Loss for' + model_name)
plt.ylabel('Train Loss')
plt.xlabel('Epoch')
plt.savefig(save_path + '_Train_loss.png')
plt.show()

simple_val_loss = H.history['val_loss']
plt.plot([los for los in simple_val_loss])
plt.title('Validation Loss for' + model_name)
plt.ylabel('Validation Loss')
plt.xlabel('Epoch')
plt.savefig(save_path + '_Validation_Loss.png')
plt.show()

'''LEARNING CURVE'''

import matplotlib.pyplot as plt

N = np.arange(0, 50)
plt.style.use('ggplot')
plt.figure()
plt.plot([los for los in H.history['loss']], label='train_loss')
plt.plot([los for los in H.history['val_loss']], label='val_loss')
plt.plot([los for los in H.history['accuracy']], label='train_accuracy')
plt.plot([los for los in H.history['val_accuracy']], label='val_accuracy')
plt.title('Training loss and accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()

"""Loss/Error Graphs"""

fig, axis = plt.subplots(1, 2, figsize=(20, 4))

axis[0].plot(H.history['accuracy'],
             label='Train accuracy',
             c='tomato', ls='-')
axis[0].plot(H.history['val_accuracy'],
             label='Validation accuracy',
             c='magenta', ls='-')

axis[0].set_xlabel('Epoch')
axis[0].set_ylabel('Accuracy')
axis[0].legend(loc='upper left')

axis[1].plot(H.history['loss'],
             label='Train loss',
             c='tomato', ls='-')
axis[1].plot(H.history['val_loss'],
             label='Validation loss',
             c='magenta', ls='-')

axis[1].set_xlabel('Epoch')
axis[1].set_ylabel('loss')
axis[1].legend(loc='upper left')
plt.savefig(save_path + '_simple_Validation_error&loss.png')
plt.show()

