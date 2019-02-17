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
        image, measurement = process_image(line)
        images.extend(image)
        measurements.extend(measurement)

    x_train = np.array(images)
    y_train = np.array(measurements)

    return x_train, y_train


def process_image(line):
    center_source_path = line[0]
    left_source_path = line[1]
    right_source_path = line[2]

    center_filename = center_source_path.split('/')[-1]
    left_filename = left_source_path.split('/')[-1]
    right_filename = right_source_path.split('/')[-1]

    center_current_path = 'data/IMG/' + center_filename
    left_current_path = 'data/IMG/' + left_filename
    right_current_path = 'data/IMG/' + right_filename

    center_image = ndimage.imread(center_current_path)
    left_image = ndimage.imread(left_current_path)
    right_image = ndimage.imread(right_current_path)

    flip_center_image = np.fliplr(center_image)
    flip_left_image = np.fliplr(left_image)
    flip_right_image = np.fliplr(right_image)

    correction = 0.2
    measurement = float(line[3])
    center_measurement = measurement
    left_measurement = center_measurement + correction
    right_measurement = center_measurement - correction

    flip_center_measurement = -1.0 * center_measurement
    flip_left_measurement = -1.0 * left_measurement
    flip_right_measurement = -1.0 * right_measurement

    images = [center_image, left_image, right_image, flip_center_image, flip_left_image, flip_right_image]
    measurements = [center_measurement, left_measurement, right_measurement, flip_center_measurement,
                    flip_left_measurement, flip_right_measurement]

    return images, measurements


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
