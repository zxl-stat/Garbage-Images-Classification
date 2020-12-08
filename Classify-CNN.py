"""""
Date: December 7, 2020.
Author: Xiaoliang Zhang(301297782)
"""""

import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


def get_image_array(path, train_image):
    img = image.load_img(path, target_size=(28, 28, 1), grayscale=False)
    img = image.img_to_array(img)
    img = img / 255
    train_image.append(img)

def get_image_array_t(id, test_image):
    img = image.load_img('te/X00'+str(id)+'.jpg', target_size=(28, 28, 1), grayscale=False)
    img = image.img_to_array(img)
    img = img / 255
    test_image.append(img)


def main():
    train = pd.read_csv('train.csv')

    seed = 321
    np.random.seed(seed)
    train_image = []
    train['path'].apply(get_image_array, train_image = train_image)
    x = np.array(train_image)

    y = train['label'].values
    y = to_categorical(y)

    x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=42, test_size=0.2)
    print('Training data shape: ', x_train.shape, y_train.shape)
    print('Validation data shape: ', x_val.shape, y_val.shape)

    class_train = np.unique(y_train)
    nclass_train = len(class_train)
    print('Total no. of outputs : ', nclass_train)
    print('Output classes : ', class_train)

    cnn_model = Sequential()
    cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28,28,3), padding='same'))
    cnn_model.add(LeakyReLU(alpha=0.1))
    cnn_model.add(MaxPooling2D((2, 2), padding='same'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(BatchNormalization())

    cnn_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    cnn_model.add(LeakyReLU(alpha=0.1))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(BatchNormalization())

    cnn_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    cnn_model.add(LeakyReLU(alpha=0.1))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(BatchNormalization())

    cnn_model.add(Flatten())

    cnn_model.add(Dense(7, activation='linear'))
    cnn_model.add(LeakyReLU(alpha=0.1))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Dense(7, activation='softmax'))

    cnn_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy'])

    cnn_model.summary()

    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100)
    cnn_model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=64,
        epochs=300,
        verbose=1,
        callbacks=[early_stop])

    test = pd.read_csv('example.csv')
    df = pd.DataFrame()
    df['Id'] = test['Id']

    test_image = []
    test['Id'].apply(get_image_array_t, test_image=test_image)
    test = np.array(test_image)

    prediction = cnn_model.predict_classes(test)
    df['Prediction'] = prediction

    df.to_csv('kaggle_submission.csv', index=False)


if __name__=='__main__':
    main()