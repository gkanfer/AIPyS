import tensorflow as tf
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

np.random.seed(42)
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.applications import vgg16
from keras.models import Model
import keras
import pandas as pd


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from tensorflow.keras import optimizers

import glob
import numpy as np

from keras.preprocessing.image import load_img, img_to_array, array_to_img
import os

class model_builder():
    """
    Parameters
    ----------
    IMG_DIM: tuple
        The dimensions of images to be used for training and validation.
    path_training: str
        The path to the directory containing training images.
    path_validation: str
        The path to the directory containing validation images.
    train_imgs_scaled: tf.array
        The scaled training images.
    validation_imgs_scaled:  tf.array
        The scaled validation images.
    path_model: str
        The path to the directory containing the model to be used for training.
    batch: int
        The batch size to be used for training.
    epoch: int
        The number of epochs to be used for training.
    input_shape: int
        The shape of the inputs to be used for training.
    train_labels_enc: tf.array
        The encoded labels associated with the training images.
    validation_labels_enc: tf.array
        The encoded labels associated with the validation images.
    train_imgs: image
        The training images.
    validation_imgs: image
        The validation images.
    steps_per_epoch_sel: int
        The number of steps to take per epoch during training.
    validation_steps: int
        The number of steps to take during validation.
    file_extention: str
        The file extension for the images to be used for training and validation.
    extract_size_train: int
        The size of the training set to be used for training.
    extract_size_val: int
        The size of the validation set to be used for validation.
    imbalance_train: int
        The imbalance ratio of the training set.
    imbalance_val: int
        The imbalance ratio of the validation set.
    model_name: str
        The name of the model to be used for training.
    path_checkpoints: str
        The path to the directory containing the checkpoints of the model.
    """
    def __init__(self, IMG_DIM=None, path_training=None, path_validation=None,
                train_imgs_scaled=None, validation_imgs_scaled=None, path_model=None, batch=None, epoch=None,
                input_shape=None, train_labels_enc=None, validation_labels_enc = None, train_imgs = None,
                 validation_imgs = None, steps_per_epoch_sel = None,validation_steps = None, file_extention = None,
                 extract_size_train = None,extract_size_val = None,imbalance_train = None, 
                 imbalance_val = None, model_name = None, path_checkpoints=None):
        self.IMG_DIM = IMG_DIM
        self.path_training = path_training
        self.path_validation = path_validation
        self.train_imgs_scaled = train_imgs_scaled
        self.validation_imgs_scaled = validation_imgs_scaled
        self.path_model = path_model
        self.batch = batch
        self.epoch = epoch
        self.input_shape = input_shape
        self.train_labels_enc = train_labels_enc
        self.validation_labels_enc = validation_labels_enc
        self.train_imgs = train_imgs
        self.validation_imgs = validation_imgs
        self.steps_per_epoch_sel = steps_per_epoch_sel
        self.validation_steps = validation_steps
        self.file_extention = file_extention
        self.extract_size_train = extract_size_train
        self.extract_size_val = extract_size_val
        self.imbalance_train = imbalance_train
        self.imbalance_val = imbalance_val
        self.model_name = model_name
        self.path_checkpoints = path_checkpoints
        
    def display_data_distribution(self):
        train_files = glob.glob(os.path.join(self.path_training,'*.' + self.file_extention))
        validation_files = glob.glob(os.path.join(self.path_validation,'*.' + self.file_extention))
        train_labels = [fn.split('/')[7].split('_')[0].strip() for fn in train_files]
        validation_labels = [fn.split('/')[7].split('_')[0].strip() for fn in validation_files]
        print('number of training samples norm: {}'.format(train_labels.count('norm')))
        print('number of training samples pheno: {}'.format(train_labels.count('pheno')))
                                                      
    def build_image__sets(self):
        # Training file generator
        train_files = glob.glob(os.path.join(self.path_training,'*.' + self.file_extention))
        if self.extract_size_train is not None:
            train_files = train_files[:self.extract_size_train]
        train_imgs = []
        # method for dealing with imbalance data, cut the norm in to minority
        if self.imbalance_train is not None:
            train_files_new = []
            count_norm = 0  
            for i in range(len(train_files)):
                if train_files[i].split('/')[7].split('_')[0].strip()=='norm':
                        if count_norm < self.imbalance_train:
                            train_files_new.append(train_files[i])
                            count_norm += 1
                else:
                    train_files_new.append(train_files[i])
            train_files = train_files_new # update variable
        print('run training')
        train_imgs = [img_to_array(load_img(train_files[i], target_size=self.IMG_DIM)) for i in tqdm(range(len(train_files)))]
        train_imgs = np.array(train_imgs)
        self.train_imgs = train_imgs
        train_labels = [fn.split('/')[7].split('_')[0].strip() for fn in train_files]
        # Validation file generator
        validation_files = glob.glob(os.path.join(self.path_validation,'*.' + self.file_extention))
        if self.extract_size_val is not None:
            validation_files = validation_files[:self.extract_size_val]
        # method for dealing with imbalance data, cut the norm in to minority
        if self.imbalance_val is not None:
            validation_files_new = []
            count_norm = 0  
            for i in range(len(validation_files)):
                if validation_files[i].split('/')[7].split('_')[0].strip()=='norm':
                        if count_norm < self.imbalance_val:
                            validation_files_new.append(validation_files[i])
                            count_norm += 1
                else:
                    validation_files_new.append(validation_files[i])
            validation_files = validation_files_new # update variable
        print('run valadtion')
        validation_imgs = [img_to_array(load_img(validation_files[i], target_size=self.IMG_DIM)) for i in tqdm(range(len(validation_files)))]
        validation_imgs = np.array(validation_imgs)
        self.validation_imgs = validation_imgs
        validation_labels = [fn.split('/')[7].split('_')[0].strip() for fn in validation_files]
        train_imgs_scaled = train_imgs.astype('float32')
        validation_imgs_scaled = validation_imgs.astype('float32')
        self.train_imgs_scaled = train_imgs_scaled/255
        self.validation_imgs_scaled = validation_imgs_scaled/255
        le = LabelEncoder()
        le.fit(train_labels)
        self.train_labels_enc = le.transform(train_labels)
        self.validation_labels_enc = le.transform(validation_labels)
        report = 'tarin labels:{}, train_labels_enc:{}.'.format(train_labels[10:15],self.train_labels_enc[10:15])
        return self.train_imgs_scaled, self.validation_imgs_scaled,self.train_labels_enc,\
               self.validation_labels_enc,self.train_imgs,self.validation_imgs,report

    def model_cnn_basic(self):
        model = Sequential()

        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.RMSprop(),
                      metrics=['accuracy'])

        model.summary()

        history = model.fit(x=self.train_imgs_scaled, y=self.train_labels_enc,
                            validation_data=(self.validation_imgs_scaled, self.validation_labels_enc),
                            batch_size=self.batch,
                            epochs=self.epoch,
                            verbose=1)

        # save the model
        os.chdir(self.path_model)
        model.save('cnn_basic.h5')
        return history

    def model_cnn_basic_Augmentation(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.3, rotation_range=50,
                                           width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                           horizontal_flip=True, fill_mode='wrap', brightness_range=[0.1, 0.9])

        val_datagen = ImageDataGenerator(rescale=1. / 255)

        # applay new model with droput and aougmentation
        train_generator = train_datagen.flow(self.train_imgs, self.train_labels_enc, batch_size=self.batch)
        val_generator = val_datagen.flow(self.validation_imgs, self.validation_labels_enc, batch_size=self.batch)

        model = Sequential()

        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['accuracy'])

        history = model.fit_generator(train_generator, steps_per_epoch=self.steps_per_epoch_sel, epochs=self.epoch,
                                      validation_data=val_generator, validation_steps=self.validation_steps,
                                      verbose=1)
        # save the model
        os.chdir(self.path_model)
        model.save('cnn_basic_Augmentation.h5')

        return history

    def model_cnn_transfer_learning_Augmentation_freez_all(self):
        vgg = vgg16.VGG16(include_top=False, weights='imagenet',
                          input_shape=self.input_shape)

        output = vgg.layers[-1].output
        output = keras.layers.Flatten()(output)
        vgg_model = Model(vgg.input, output)

        vgg_model.trainable = False
        for layer in vgg_model.layers:
            layer.trainable = False

        # show which layer are frozen
        pd.set_option('max_colwidth', -1)
        layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
        pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

        # flat all the fetures from training and validation set
        # for feeding the last blovk of vg16
        def get_bottleneck_features(model, input_imgs):
            features = model.predict(input_imgs, verbose=0)
            return features

        train_features_vgg = get_bottleneck_features(vgg_model, self.train_imgs_scaled)
        validation_features_vgg = get_bottleneck_features(vgg_model, self.validation_imgs_scaled)

        report = 'Train Bottleneck Features:{},  Validation Bottleneck Features:{}'.format(train_features_vgg.shape,validation_features_vgg.shape)
        '''
        now we will train with aougmentation
        Pre-trained CNN model as a Feature Extractor with Image Augmentation
        '''
        train_datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.3, rotation_range=50,
                                           width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                           horizontal_flip=True, fill_mode='wrap', brightness_range=[0.1, 0.9])

        val_datagen = ImageDataGenerator(rescale=1. / 255)

        # applay new model with droput and aougmentation
        train_generator = train_datagen.flow(self.train_imgs, self.train_labels_enc, batch_size=self.batch)
        val_generator = val_datagen.flow(self.validation_imgs, self.validation_labels_enc, batch_size=self.batch)
        

        model = Sequential()
        model.add(vgg_model)
        model.add(Dense(512, activation='relu', input_dim=self.input_shape))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.RMSprop(learning_rate=2e-5),
                      metrics=['accuracy'])

        history = model.fit_generator(train_generator, steps_per_epoch=self.steps_per_epoch_sel, epochs=self.epoch,
                                      validation_data=val_generator, validation_steps=self.validation_steps,
                                      verbose=1)

        # save the model
        os.chdir(self.path_model)
        model.save('transfer_learning_aug_dropout_freez_all.h5')

        return history

    def model_cnn_transfer_learning_Augmentation_drop_layer_4and5(self,**kwargs):
        vgg = vgg16.VGG16(include_top=False, weights='imagenet',
                          input_shape=self.input_shape)

        output = vgg.layers[-1].output
        output = keras.layers.Flatten()(output)
        vgg_model = Model(vgg.input, output)

        vgg_model.trainable = True

        set_trainable = False
        for layer in vgg_model.layers:
            if layer.name in ['block5_conv1', 'block4_conv1']:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
        pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])

        train_datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.3, rotation_range=50,
                                           width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                           horizontal_flip=True, fill_mode='wrap', brightness_range=[0.1, 0.9])

        val_datagen = ImageDataGenerator(rescale=1. / 255)

        # applay new model with droput and aougmentation
        train_generator = train_datagen.flow(self.train_imgs, self.train_labels_enc, batch_size=self.batch)
        val_generator = val_datagen.flow(self.validation_imgs, self.validation_labels_enc, batch_size=self.batch)
        input_shape = self.input_shape

        model = Sequential()
        model.add(vgg_model)
        model.add(Dense(512, activation='relu', input_dim=input_shape))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))

        learning_rate = kwargs.get('learning_rate', 1e-5)
        if 'learning_rate' in kwargs:
            learning_rate = kwargs['learning_rate']
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.RMSprop(learning_rate=learning_rate),
                      metrics=['accuracy'])
        
        checkpoint_path = self.path_checkpoints + "cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        
        history = model.fit_generator(train_generator, steps_per_epoch=self.steps_per_epoch_sel, epochs=self.epoch,
                                      validation_data=val_generator, validation_steps=self.validation_steps,
                                      verbose=1,callbacks=[cp_callback])

        os.chdir(self.path_model)
        if self.model_name:
            model.save(self.model_name)
            with open(self.model_name + '_history'  , 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
        else:
            model.save('cnn_transfer_learning_Augmentation_drop_layer_4and5.h5')
            with open('history_', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
        return history



