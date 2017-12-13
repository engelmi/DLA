# -*- coding: utf-8 -*-

"""
merge-datasets.py: Merge the stock data and the data from google trend in one csv file.
Call it with the name of the stock.
The path where to store the data is an optional second parameter it defaults to "myDataset".
"""

import shutil
from datetime import datetime
import json
import logging
import csv
from os.path import join
import os
import sys

import numpy as np


__author__ = "Daniel Sikeler"


class Trend(object):
  """
  Collects all information for one google trend query.
  """

  def __init__(self, name):
    """
    Initialize.
    :param name: The explored keyword.
    """
    self.name = name
    self.data = {}

  def load(self, path):
    """
    Collect the trend data from a file.
    :param path: The folder where the file can be found.
    """
    try:
      with open(join(path, self.name + ".json"), "r") as fp:
        trend = json.load(fp)
        for day in trend["default"]["timelineData"]:
          self.data[day["formattedTime"]] = int(day["formattedValue"][0])
    except Exception as ex:
      logging.error("Could not get the googletrend data. " + ex)

  def getAllData(self):
    """
    Get all data for this trend sorted by day.
    :return: A list of tuples (day, value).
    """
    return sorted(self.data.items(), key=lambda d: datetime.strptime(d[0], "%d.%m.%Y"))

  def getData(self, day):
    """
    Get the data for one special day.
    :param day: The day we want the data for.
    :return: The value at this day or 0 if there was no data.
    """
    if day in self.data:
      return self.data[day]
    else:
      return "0"


class Stock(object):
  """
  Collects all information for a single stock.
  """

  def __init__(self, name):
    """
    Initialie.
    :param name: The name of the stock.
    """
    self.name = name
    self.data = {}

  def load(self, path):
    """
    Collect the stock data from a file.
    :param path: The folder where the stock file is located.
    """
    try:
      with open(join(path, self.name + "_data.csv"), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        csvHeader = next(reader)
        for row in reader:
          try:
            currOpen = float(row[1])
            currClose = float(row[4])
            day = datetime.strptime(row[0], "%Y-%m-%d").strftime("%d.%m.%Y")
            self.data[day] = (currOpen, int(np.sign(currClose-currOpen)))
          except ValueError:
            logging.warn("Could not interpret day " + row[0])
    except Exception as ex:
      logging.error("Could not get the stock data. " + ex)

  def getData(self, day):
    """
    Get the data for one special day.
    :param day: The day to get the data for.
    :return: The stock at this day or None. (value, label)
    """
    if day in self.data:
      return self.data[day]
    else:
      return None

  def getAllData(self):
    """
    Get all data for this stock sorted by day.
    :return: A list of tuples (day, stockdata)
    """
    return sorted(self.data.items(), key=lambda d: datetime.strptime(d[0], "%d.%m.%Y"))


class StockTrend(object):
  """
  A combination of the data from one stock and its according google trends.
  """

  def __init__(self, name, trendNames,
               stockPath="sandp500/individual_stocks_5yr", googlePath="googletrends"):
    """
    Initialize.
    :param name: The name of the stock.
    :param trendNames: A list of all google trend queries belonging to this stock.
    :param stockPath: Path to the folder where the stock is located.
    :param googlePath: Path to the folder where the google trend data are.
    """
    self.name = name
    self.trendNames = trendNames
    self.stockPath = stockPath
    self.googlePath = googlePath

  def load(self):
    """
    Load all stock and google trend data.
    """
    self.stock = Stock(self.name)
    self.stock.load(self.stockPath)

    self.trends = []
    for trendName in self.trendNames:
      trend = Trend(trendName)
      trend.load(join(self.googlePath, self.name))
      self.trends.append(trend)

  def getData(self):
    """
    Get the collected data as a list of tuples. One tuple for each day.
    """
    data = []
    for day, s in self.stock.getAllData():
      dataEntry = list(s)
      for trend in self.trends:
        s = trend.getData(day)
        dataEntry.append(s)
      data.append(dataEntry)
    return data

  def write(self, path):
    """
    Write the collected data to a file with the name of the stock
    :param path: Path were the file should be.
    """
    with open(join(path,self.name + ".csv"), "w") as fp:
      writer = csv.writer(fp, delimiter=",", quotechar='"')
      writer.writerows(self.getData())


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Not enough input parameters: stock name needed")
  stock = sys.argv[1]
  path = "myDataset"
  if len(sys.argv) > 2:
    path = sys.argv[2]
  trends = list(map(lambda file: file[:-5], os.listdir(join("googletrends", stock))))
  st = StockTrend(stock, trends)
  st.load()
  if os.path.exists(path):
    shutil.rmtree(path)
  os.makedirs(path)
  st.write(path)