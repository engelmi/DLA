# -*- coding: utf-8 -*-

from os.path import isfile, join

import numpy as np
import csv
import sys
import logging

# flag to check if python3 is used
PY3 = sys.version_info[0] == 3


def zeroCenter(filepath, shape):
    """
    Zero centers the stock and google trends data for a given csv file.
    :param filepath: The path to the csv file to read and zero center the data.
    :param shape: The shape of each set (30 rows, 16 columns).
    :return: A zero centered numpy array of shape (z, shape) containing the stock and google trends numbers. Where z is the
             number of sets. z=int(csvfilerows/shape[0])=int(csvfilerows/30)
    """
    data = readMergedCSV(filepath)
    # reshape and discard oldest values
    rows = data.shape[0]
    sets = int(data.shape[0] / shape[0])
    data = data[rows - sets * shape[0]:]
    data = data.reshape(sets, shape[0], shape[1])
    for s in range(0, sets):
        data[s, :, 2:] -= np.mean(data[s, :, 2:], axis=0)
    return data


def readMergedCSV(filepath):
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


if __name__ == "__main__":
    time_steps = 30
    values = 16
    d = zeroCenter("datasets/myDataset/MMM.csv", (time_steps, values))
    print(d[0])
