# -*- coding: utf-8 -*-

from stock import Stock

from os import listdir
from os.path import isfile, join

import numpy as np
import csv

"""
Zero centers the stock and google trends data for a given csv file. 
:param filepath: The path to the csv file to read and zero center the data. 
:return: A zero centered numpy matrix containing the stock and google trends numbers. 

TODO: exclude the labels (-1, 0, 1) from the mean calculation (column 1)
"""
def zeroCenter(filepath):
	data = readMergedCSV(filepath)
	data -= np.mean(data, axis=0)
	return data

"""
Reads a csv file and returns a 2D array. 
:param filepath: The path to the csv file. 
:return: A numpy matrix.  
"""
def readMergedCSV(filepath):
	data = []
	try:
		if isfile(filepath) and filepath[-3:] == "csv":
			with open(filepath) as f:
				reader = csv.reader(f, delimiter=',', quotechar='"')
				for row in reader:
					data.append([float(row[i]) for i in range(len(row))])
	except Exception as ex:
		# print ex.message
		pass
	return np.matrix(data)


"""
Iterates over all .csv files within the given folder. 

ToDo:   + error handling
        + extend function params -> file extension
"""
def loadAllStockData(relativeFolderPath):
    allStocks = {}
    csvfiles = [f for f in listdir(relativeFolderPath) if isfile(join(relativeFolderPath, f)) and f[-3:] == "csv"]
    for f in csvfiles:
        stock = loadSingleStock(join(relativeFolderPath, f))
        if stock is not None:
            allStocks[stock.getStockName()] = stock
    return allStocks

"""
ToDo: + Exception handling -> additional logging?
"""
def loadSingleStock(relativePath):
    stock = None
    try:
        csvfile = open(relativePath, 'rb')
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        csvHeader = reader.next()
        currDate, currOpen, currHigh, currLow, currClose, currVolume, currName = reader.next()
        stock = Stock(currName)
        stock.addDay(currDate, currOpen, currHigh, currLow, currClose, currVolume, currName)
        for rowCurrentDay in reader:
            currDate, currOpen, currHigh, currLow, currClose, currVolume, currName = rowCurrentDay
            stock.addDay(currDate, currOpen, currHigh, currLow, currClose, currVolume, currName)

    except Exception as ex:
        return stock

    return stock
