import os
import pandas as pd
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import os
import psutil
import zipfile
import click
from math import ceil


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print("Current process using memoery: {0} MB".format((process.memory_info().rss / 1024/1024)))  # in bytes


def read_image(image_file):
    image = cv2.imread(image_file)
    if image is None:
        raise FileNotFoundError("{0} cannot be read as image".format(image_file))
    return image

def flip_image(image):
    """
    Flip the image left to right
    :param image:
    :return:
    """
    return np.fliplr(image)

def pre_process(image):
    """
    :param image: a image
    :return: a preprocessed image to feed into imagenet
    """
    return image

def read_driving_log(data_directory='/driving_data/data'):
    driving_log_csv_file = os.path.join(data_directory, 'driving_log.csv')
    return pd.read_csv(driving_log_csv_file)

def image_visualization(data_directory='/driving_data/data'):
    driving_log = read_driving_log(data_directory)
    img_file = os.path.join(data_directory, driving_log['center'][0])
    img = read_image(img_file)

    print(img.shape)
    cv2.imshow('image', img)
    cv2.waitKey()

def read_training_data_log(data_directory, num_images=-1):
    """
    Read the csv file and generate full path for convenience
    :param data_directory:
    :param num_images:
    :return:
    """

    driving_log = read_driving_log(data_directory)
    if num_images != -1:
        driving_log = driving_log.head(num_images)

    # Create full paths in the data frame
    convert_to_full = lambda x: os.path.join(data_directory, x.replace(' ',''))
    driving_log['center_full'] = driving_log.loc[:,'center'].apply(convert_to_full)
    driving_log['right_full'] = driving_log.loc[:, 'right'].apply(convert_to_full)
    driving_log['left_full'] = driving_log.loc[:, 'left'].apply(convert_to_full)

    return driving_log

def data_frame_to_training_data(driving_log, augment_lr=False, steering_correction=0.2):
    """
    Given a data frame corresponding to image paths
    todo(qingyouz): the data augmentation step could be done either on the fly or offline.
    Return X_train and y_train
    :param log_df:
    :return:
    """
    # first read the images
    num_images = len(driving_log)
    num_image_per_line = 1
    if augment_lr:
        num_images += len(driving_log) * 2
        num_image_per_line += 2

    # Add in flipped images
    num_images *= 2
    num_image_per_line *= 2

    X_train = np.zeros([num_images] + list(read_image(driving_log['center_full'].values[0]).shape))
    y_train = np.zeros(num_images)

    for idx in range(driving_log.shape[0]):
        cur_idx = num_image_per_line * idx
        df_idx = driving_log.index[idx]

        image_center = read_image(driving_log['center_full'][df_idx])
        image_right = read_image(driving_log['right_full'][df_idx])
        image_left = read_image(driving_log['left_full'][df_idx])

        X_train[cur_idx] = image_center
        y_train[cur_idx] = driving_log['steering'][df_idx]
        X_train[cur_idx+1] = flip_image(image_center)
        X_train[cur_idx+1] = -(driving_log['steering'][df_idx])
        if augment_lr:
            X_train[cur_idx+2] = image_left
            y_train[cur_idx+2] = driving_log['steering'][df_idx] + steering_correction
            X_train[cur_idx+3] = flip_image(image_left)
            y_train[cur_idx+3] = (driving_log['steering'][df_idx] + steering_correction) * -1.0

            X_train[cur_idx+4] = image_right
            y_train[cur_idx+4] = (driving_log['steering'][df_idx] - steering_correction)
            X_train[cur_idx+5] = flip_image(image_right)
            y_train[cur_idx+5] = (driving_log['steering'][df_idx] - steering_correction) * -1.0

    return X_train, y_train

def read_training_data(data_directory, num_images=-1):
    driving_log = read_training_data_log(data_directory, num_images=num_images)
    # read image into data frame
    driving_log['image_center'] = driving_log.loc[:,'center_full'].apply(read_image)
    # driving_log['image_left'] = driving_log.loc[:, 'left_full'].apply(read_image)
    # driving_log['image_right'] = driving_log.loc[:, 'right_full'].apply(read_image)

    # augment data with flipped
    X_train= {}
    y_train = {}
    X_train['center'] = np.zeros([driving_log['image_center'].shape[0]]+list(driving_log['image_center'].values[0].shape))
    y_train['center'] = np.zeros(driving_log['image_center'].shape[0])
    X_train['center_flipped'] = np.zeros([driving_log['image_center'].shape[0]]+list(driving_log['image_center'].values[0].shape))
    y_train['center_flipped'] = np.zeros(driving_log['image_center'].shape[0])

    print("Center has shape {0}".format(X_train['center'].shape))
    print("Center flipped has shape {0}".format(X_train['center_flipped'].shape))

    for idx in range(driving_log['image_center'].shape[-1]):
        if(idx%500==0):
            print_memory_usage()
        X_train['center'][idx] = driving_log['image_center'][idx]
        y_train['center'][idx] = driving_log['steering'][idx]
        X_train['center_flipped'][idx] = flip_image(driving_log['image_center'][idx])
        y_train['center_flipped'][idx] = -driving_log['steering'][idx]

    # create training data by appending needed arrays
    print("Center has shape {0}".format(X_train['center'].shape))
    print("Center flipped has shape {0}".format(X_train['center_flipped'].shape))

    # todo(qingyouz): I need to know more about the difference between
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.append.html
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
    # but forenow, let's just use np.append
    X_train_combined = np.append(X_train['center'], X_train['center_flipped'], axis=0)
    y_train_combined = np.append(y_train['center'], y_train['center_flipped'], axis=0)

    return X_train_combined, y_train_combined


def training_data_generator(driving_log,num_images=-1, batch_size=32):
    """
    Use a genarator pattern to read images into memory instead of reading them into a chunk
    :param data_directory:
    :param num_images:
    :return:
    """
    while 1:
        sample_log = driving_log.sample(batch_size)
        X_train, y_train = data_frame_to_training_data(sample_log, augment_lr=True)
        yield X_train, y_train


#
# Models
#

def image_test_model(X_train, y_train, save_path='model.h5'):

    input_shape = X_train[0].shape
    print('X input_shape = {0}, Y input_shape = {1}'.format(X_train.shape, y_train.shape))
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

    model.save(save_path)

def LeNet_model(X_train, y_train, save_path='model.h5', epochs=5):
    """

    :param X_train: Input training data, an array of 3 channel pictures
    :param y_train: Input training data
    :param save_path:
    :param epochs:
    :return:
    """
    input_shape = X_train[0].shape
    print(input_shape)
    model = Sequential()

    model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=input_shape))

    model.add(Convolution2D(filters=6,kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(filters=16,kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs)

    model.save(save_path)

def add_pre_process_layers(model, input_shape):
    """
    Shared utility to add preprocess model.
    :param model: the Sequentiao model
    :param input_shape:
    :return: the model with pre-process layers
    """

    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    return model

# Transfer learning models


# training models from scratch:


def pre_trained_inception(X_train, y_train, save_path='InceptionV3.h5', epochs=5):
    # create the base pre-trained model
    input_shape = X_train[0].shape
    print(input_shape)


    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    x = Flatten()(base_model.output)
    x = Dense(1024)(x)
    x = Dense(200)(x)
    x = Dense(1)(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x)

    # make the inception model part not trainable
    for layer in base_model.layers:
        layer.trainable = False

    print(model.summary())

    model.compile(loss='mse',optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs)

    model.save(save_path)

def nvidia_net(input_shape):
    """
    https://devblogs.nvidia.com/deep-learning-self-driving-cars/
    :param X_train:
    :param y_train:
    :param save_path:
    :param epochs:
    :return:
    """

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    print(model.summary())

    model.compile(loss='mse',optimizer='adam')

    return model

@click.command()
@click.option('--model-save-path','-s',default='model.h5')
@click.option('--epochs','-e',type=int, default=2)
@click.option('--training-data-directory',default="./driving_data/training_data")
@click.option('--validation-data-directory', default='./driving_data/validation_data')
def main(model_save_path="", epochs=2, training_data_directory='', validation_data_directory=''):
    # read some sample training data for ease of processing
    X_train, y_train = read_training_data(training_data_directory, num_images=10)
    # used to test pipeline

    driving_log = read_training_data_log(training_data_directory)
    print("Using {0} pictures for training".format(len(driving_log)))
    train_generator = training_data_generator(driving_log)

    validation_log = read_training_data_log(validation_data_directory)
    print("Using {0} pictures for validation".format(len(validation_log)))
    validation_generator = training_data_generator(validation_log)

    model = Sequential()
    model = nvidia_net(input_shape=X_train[0].shape)

    # ensure we are using GPUs
    from keras import backend as K
    print(K.tensorflow_backend._get_available_gpus())

    batch_size = 128
    history_object = model.fit_generator(train_generator,
                        steps_per_epoch=ceil(len(driving_log) * 6 / batch_size),
                        validation_data=validation_generator,
                        validation_steps=ceil(len(validation_log) * 6/batch_size),
                        epochs=epochs, verbose=1)

    model.save(model_save_path)


if __name__ == '__main__':
    main()

