"""
This is the project file where the model for behavioral cloning is built
"""

import csv
from scipy import ndimage
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Cropping2D, Lambda, Convolution2D
from keras.callbacks import CSVLogger


def data_extraction():
    lines = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines[1:]:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = 'data/IMG/' + filename
            image = ndimage.imread(current_path)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)

    x_train = np.array(images)
    y_train = np.array(measurements)

    return x_train, y_train


def neural_network_model(x_train, y_train):
    # Create the Sequential model
    model = Sequential()

    """
    Following Nvidia's self driving car model
    https://devblogs.nvidia.com/deep-learning-self-driving-cars/
    """
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(3, 160, 320)))
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

    model.compile('adam', loss='mse')
    csv_logger = CSVLogger("model/model_history_log.csv", append=True, separator=';')
    model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[csv_logger])
    model.save('model/model.h5')


def main():
    x_train, y_train = data_extraction()
    neural_network_model(x_train, y_train)
    print('')

if __name__ == '__main__':
    main()
