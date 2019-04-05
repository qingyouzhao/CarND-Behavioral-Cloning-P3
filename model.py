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


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print("Current process using memoery: {0} MB".format((process.memory_info().rss / 1024/1024)))  # in bytes


def read_image(image_file):
    return cv2.imread(image_file)

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

def read_training_data(data_directory, use_head = False, num_images=-1):
    driving_log = read_driving_log(data_directory)
    if num_images != -1:
        driving_log = driving_log.head(num_images)

    if use_head:
        driving_log = driving_log.head()



    # Create full driving names
    driving_log['center_full'] = driving_log.loc[:,'center'].apply(lambda x: os.path.join(data_directory, x))
    driving_log['right_full'] = driving_log.loc[:, 'right'].apply(lambda x: os.path.join(data_directory, x))
    driving_log['left_full'] = driving_log.loc[:, 'left'].apply(lambda x: os.path.join(data_directory, x))

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

def nvidia_net(X_train, y_train, save_path='nvidia_custom.h5', epochs=5):
    """
    https://devblogs.nvidia.com/deep-learning-self-driving-cars/
    :param X_train:
    :param y_train:
    :param save_path:
    :param epochs:
    :return:
    """

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=X_train[0].shape))
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
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs)

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    model.save(save_path)


def main():
    data_directory = os.path.join('driving_data','custom_data')
    X_train, y_train = read_training_data(data_directory, use_head=False)
    # used to test pipeline
    # image_test_model(X_train,y_train)

    nvidia_net(X_train,y_train)

if __name__ == '__main__':
    main()

