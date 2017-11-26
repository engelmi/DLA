
"""
Contains all information for a single stock.
ToDo:   + Documentation
"""
class Stock(object):

    def __init__(self, stock_name):
        self.stockName = stock_name
        self.data = {}

    def addDay(self, date, on_open, highest, lowest, on_close, volume, name):
        if name == self.stockName and date not in self.data:
            self.data[date] = {
                "onOpen" : on_open,
                "onClose" : on_close,
                "highest" : highest,
                "lowest" : lowest,
                "volume" : volume
            }

    def getStockName(self):
        return self.stockName

    def getDates(self):
        return list(self.data.keys())

    def getDay(self, date):
        retData = None
        if date in self.data:
            retData = self.data[date]
        return retData

    def getDays(self):
        return self.data