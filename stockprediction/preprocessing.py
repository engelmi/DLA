# -*- coding: utf-8 -*-

from os import listdir, makedirs
from os.path import isfile, join, exists
import shutil

import numpy as np
import csv
import sys
import logging

# flag to check if python3 is used
PY3 = sys.version_info[0] == 3


def zero_center(filepath, shape):
    """
    Zero centers the stock and google trends data for a given csv file.
    :param filepath: The path to the csv file to read and zero center the data.
    :param shape: The shape of each set (30 rows, 16 columns).
    :return: A zero centered numpy array of shape (z, shape) containing the stock and google trends numbers. Where z is the
             number of sets. z=int(csvfilerows/shape[0])=int(csvfilerows/30)
    """
    data = read_merged_csv(filepath)
    # reshape and discard oldest values
    rows = data.shape[0]
    sets = int(data.shape[0] / shape[0])
    data = data[rows - sets * shape[0]:]
    data = data.reshape(sets, shape[0], shape[1])
    for s in range(0, sets):
        data[s, :, 2:] -= np.mean(data[s, :, 2:], axis=0)
    return data


def read_merged_csv(filepath):
    """
    Reads a csv file and returns a 2D array.
    :param filepath: The path to the csv file.
    :return: A numpy array.
    """
    data = []
    try:
        if isfile(filepath) and filepath[-3:] == "csv":
            with open(filepath) as f:
                reader = csv.reader(f, delimiter=',', quotechar='"')
                for row in reader:
                    data.append([float(row[i]) for i in range(len(row))])
    except Exception as ex:
        logging.error(ex)
    return np.array(data)


def preprocessing(folderPreprocessedData, folderMergedData, time_steps, values):
    """
    Preprocessing of the merged data. Reads the merged data of stocks and google trends,
    applies preprocessing routines to it and writes the result out.
    """
    delete_preprocessed_data(folderPreprocessedData)
    mergedCSVFiles = [f for f in listdir(folderMergedData) if
                      isfile(join(folderMergedData, f)) and f[-3:] == "csv"]
    for csvfile in mergedCSVFiles:
        centeredData = zeroCenter(join(folderMergedData, csvfile), (time_steps, values+1))
        # maybe further preprocessing...
        persist_preprocessed_data(centeredData, folderPreprocessedData, csvfile[:len(csvfile) - 4])


def persist_preprocessed_data(matrix, filepath, filename):
    """
    Persists a numpy matrix containing preprocessed data.
    :param matrix: The numpy matrix to persist.
    :param filepath: The relative path.
    :param filename: The name of the file.
    """
    if not exists(filepath):
        makedirs(filepath)
    np.save(join(filepath, filename), matrix)


def load_preprocessed_data(filepath):
    """
    Loads a persisted numpy matrix containing preprocessed data.
    :param filepath: The relative path including the filename and extension.
    """
    if exists(filepath):
        return np.array(np.load(filepath))
    return None


def delete_preprocessed_data(filepath):
    """
    Deletes the saved preprocessed data.
    :param filepath: The relative path.
    """
    if exists(filepath):
        shutil.rmtree(filepath)


if __name__ == "__main__":
    time_steps = 30
    values = 16
    d = zero_center("datasets/myDataset/MMM.csv", (time_steps, values))
    print(d.shape)
