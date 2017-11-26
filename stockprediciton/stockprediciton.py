import preprocessing as pp

import os
import tensorflow as tf
import numpy as np

"""
Main class for the prediction of stock course development. 
TODO:   + Documentation
"""
class StockPredictor(object):

    def __init__(self):
        pass

    """
    ToDo:   + Documentation
    """
    def preprocessing(self):
        stock = pp.loadSingleStock(os.path.join("datasets", "sandp500", "individual_stocks_5yr", "AAL_data.csv"))
        print stock.getStockName()
        for d in stock.getDates():
            print d, stock.getDay(d)

        allStocks = pp.loadAllStockData(os.path.join("datasets", "sandp500", "individual_stocks_5yr"))
        print list(allStocks.keys())



if __name__ == "__main__":
    sp = StockPredictor()
    sp.preprocessing()