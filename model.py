import os
import pandas as pd
import cv2
import numpy as np
import keras


def read_image(image_file):
    return cv2.imread(image_file)

def pre_process(image):
    """
    :param image: a image
    :return: a preprocessed image to feed into imagenet
    """

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

def read_training_data(data_directory, use_head = False):
    driving_log = read_driving_log(data_directory)
    if use_head:
        driving_log = driving_log.head()
    driving_log['center_full'] = driving_log.loc[:,'center'].apply(lambda x: os.path.join(data_directory, x))
    driving_log['image'] = driving_log.loc[:,'center_full'].apply(read_image)

    X_train = np.zeros([driving_log['image'].shape[0]]+list(driving_log['image'].values[0].shape))
    y_train = np.zeros(driving_log['steering'].shape[0])

    for idx in range(driving_log['image'].shape[-1]):
        X_train[idx] = driving_log['image'][idx]
        y_train[idx] = driving_log['steering'][idx]

    return X_train, y_train



def image_test_model(X_train, y_train, save_path='model.h5'):
    from keras.models import Sequential
    from keras.layers import Flatten, Dense

    input_shape = X_train[0].shape
    print('X input_shape = {0}, Y input_shape = {1}'.format(X_train.shape, y_train.shape))
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

    model.save(save_path)


def main():
    data_directory = os.path.join('driving_data','data')
    X_train, y_train = read_training_data(data_directory, use_head=True)
    image_test_model(X_train,y_train)


if __name__ == '__main__':
    main()

