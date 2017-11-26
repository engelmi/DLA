# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import shutil
import logging
from datetime import date

from trend import Trend


"""
TODO:
"""
def collectGoogleTrendsData(start, end):
    with open('nyse/securities.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        rows = list(reader)
        for i in range(start, end):
            stock = rows[i]
            trend = Trend(stock, date(2016, 1, 1), date.today())
            path = "googletrends/" + stock[0]

            try:
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.mkdir(path)
            except Exception as ex:
                print("\t error creating folder: " + ex.message)
                continue

            trend.collectData()
            index = 0
            s = ''
            for keyword, data in trend.getData():
              with open(os.path.join(path, keyword+str(index)+".json"), "w") as f:
                json.dump(data, f, indent=2, separators=(',', ': '))
              if s == keyword:
                index += 1
              else:
                s = keyword
                index = 0


if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Not enough input parameters: starting sock and ending stock needed")
  logging.basicConfig(format='%(message)s',level=logging.INFO)
  collectGoogleTrendsData(int(sys.argv[1]), int(sys.argv[2]))
