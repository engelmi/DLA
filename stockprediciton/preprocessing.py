from stock import Stock

from os import listdir
from os.path import isfile, join

import numpy as np
import csv


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
