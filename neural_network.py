import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow.keras import layers

import cooler


def form_x_set(matrix):
    SQUARE_SIZE = 2
    square_h_w = SQUARE_SIZE * 2 + 1

    number_of_features = (square_h_w * square_h_w) - square_h_w

    x_set = np.array([]).reshape(number_of_features, 0)

    column_submatrices = [matrix[:, i:(i + square_h_w)] for i in
                          range(0, matrix.shape[1] - (square_h_w - 1))]

    for column_submatrix in column_submatrices:
        surrounding_square = np.delete(column_submatrix, (square_h_w // 2), axis=0)
        x_set = np.concatenate((x_set, surrounding_square.reshape((number_of_features, 1))), axis=1)

    return x_set


def mark_nans_as_zeros(matrix):
    # we think NaN == 0 for now
    matrix[np.isnan(matrix)] = 0


def invalid_hic(hic_row, defect_threshold):
    boolean_row = hic_row != 0
    return np.sum(boolean_row) < ((hic_row.size / 100) * defect_threshold)


# Neural network model prediction
def predict_nn(x, nn_model):
    return nn_model.predict(x.T, batch_size=32)


def restore_hic(hic_matrix, defect_threshold, nn_model):
    mark_nans_as_zeros(hic_matrix)

    # restore rows and columns
    for i in range(2, hic_matrix.shape[0] - 2):
        if invalid_hic(hic_matrix[i, :], defect_threshold):
            row_features = form_x_set(hic_matrix[(i - 2):(i + 3), :])
            row_prediction = predict_nn(row_features, nn_model).T
            hic_matrix[i, :] = np.pad(row_prediction, ((0, 0), (2, 2)), mode='mean')
            hic_matrix[:, i] = np.pad(row_prediction, ((0, 0), (2, 2)), mode='mean')


def create_folder():
    dirName = 'results_nn'

    if not os.path.exists(dirName):
        os.mkdir(dirName)


def save_hic(hic_matrix, chrom_name):
    create_folder()
    np.save('results_nn/' + chrom_name + '_restored', hic_matrix)


def get_nn_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(80, activation='relu', input_shape=(20,)))
    model.add(layers.Dense(80, activation='relu'))
    model.add(layers.Dense(80, activation='relu'))
    model.add(layers.Dense(80, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))

    model.load_weights('nn_model.h5')

    return model


def main():
    filepath = sys.argv[1]

    if len(sys.argv) == 3:
        defect_threshold = int(sys.argv[2])
    else:
        # by default row or column is considered to be defective if less than 5% of values are non-zero
        defect_threshold = 5

    c = cooler.Cooler(filepath)

    model = get_nn_model()

    for chrom in c.chromnames:
        arr = c.matrix(balance=True).fetch(chrom)
        restore_hic(arr, defect_threshold, model)
        save_hic(arr, chrom)
        print(chrom + " was processed")


main()
