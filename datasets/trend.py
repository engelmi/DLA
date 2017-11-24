# coding=utf8

from datetime import date, timedelta
import urllib2
import json
import logging

class Trend:

  # used symbols for parameter within the urlTrendsExplore
  paramSymbols = {
    "keyword": "keyword",
    "geoLocation": "geo",
    "token": "token",
    "startTime": "startTime",
    "endTime": "endTime",
    "relatedType": "relatedType",
    "relatedStartTime": "relatedStartTime",
    "relatedEndTime": "relatedEndTime",
    "topic": "ENTITY",
    "search": "QUERY"
  }

  # used symbols to indicate a parameter within an url
  paramSymbolPre = "{{"
  paramSymbolPost = "}}"

  def createParam(param):
    paramSymbolPre = "{{"
    paramSymbolPost = "}}"
    return paramSymbolPre + param + paramSymbolPost

  urlTrendsExplore = (
    'https://trends.google.com/trends/api/explore?hl=de&tz=-60&req=%7B%22comparisonItem%22:%5B%7B%22keyword%22:%22' +
    createParam(paramSymbols["keyword"]) + '%22,%22geo%22:%22' +
    createParam(paramSymbols["geoLocation"]) + '%22,%22time%22:%22' +
    createParam(paramSymbols["startTime"]) + '+' + createParam(paramSymbols["endTime"]) +
    '%22%7D%5D,%22category%22:0,%22property%22:%22%22%7D&tz=-60')

  urlTrendsMultiline = (
    'https://trends.google.com/trends/api/widgetdata/multiline?hl=de&tz=-60&req=%7B%22time%22:%22' +
    createParam(paramSymbols["startTime"]) + '+' + createParam(paramSymbols["endTime"]) +
    '%22,%22resolution%22:%22DAY%22,%22locale%22:%22de%22,%22comparisonItem%22:%5B%7B%22geo%22:%7B%7D,' +
    '%22complexKeywordsRestriction%22:%7B%22keyword%22:%5B%7B%22type%22:%22BROAD%22,%22value%22:%22' +
    createParam(paramSymbols["keyword"]) +
    '%22%7D%5D%7D%7D%5D,%22requestOptions%22:%7B%22property%22:%22%22,%22backend%22:%22IZG%22,%22category%22:0%7D%7D&token=' +
    createParam(paramSymbols["token"]) + '&tz=-60')

  # topics and searches share the same url, the token distinguishes the requests
  urlRelated = (
    'https://trends.google.com/trends/api/widgetdata/relatedsearches?hl=de&tz=-60&req=%7B%22restriction%22:%7B%22geo%22:%7B%7D,%22time%22:%22' +
    createParam(paramSymbols["startTime"]) + '+' + createParam(paramSymbols["endTime"]) +
  	'%22,%22complexKeywordsRestriction%22:%7B%22keyword%22:%5B%7B%22type%22:%22BROAD%22,%22value%22:%22' +
    createParam(paramSymbols["keyword"]) + '%22%7D%5D%7D%7D,%22keywordType%22:%22' +
    createParam(paramSymbols["relatedType"]) +
  	'%22,%22metric%22:%5B%22TOP%22,%22RISING%22%5D,%22trendinessSettings%22:%7B%22compareTime%22:%22' +
    createParam(paramSymbols["relatedStartTime"]) + '+' + createParam(paramSymbols["relatedEndTime"]) +
    '%22%7D,%22requestOptions%22:%7B%22property%22:%22%22,%22backend%22:%22IZG%22,%22category%22:0%7D,%22language%22:%22de%22%7D&token=' +
    createParam(paramSymbols["token"]))

  def __init__(self, stock, startTime, endTime):
    self.timeBounderies = []
    self.data = []
    while startTime < endTime:
      stopTime = startTime + timedelta(days=90)
      if stopTime > date.today():
        stopTime = date.today()
      self.timeBounderies.append([startTime.strftime("%Y-%m-%d"), stopTime.strftime("%Y-%m-%d")])
      startTime = stopTime + timedelta(days=1)
    self.node = {
      "symbol": stock[0],
      "keyword": self.replaceInvalidKeywordchars(stock[1]),
      "sector": stock[3],
      "subindustry": stock[4]
    }

  def prepareParam(self, param):
    return self.paramSymbolPre + param + self.paramSymbolPost

  def replaceInvalidKeywordchars(self, keyword):
    return keyword.replace(" ", "%20").replace("&", "%26").replace(".", "%2E")

  def requestURLData(self, requestURL):
    urlData = None
    requestSuccessful = False
    maxNumberOfRetries = 3
    retry = 1
    while not requestSuccessful and retry <= maxNumberOfRetries:
      try:
        response = urllib2.urlopen(requestURL)
        if response.getcode() != 429:
          requestSuccessful = True
          urlData = response.read()
          break
      except Exception as ex:
        pass
      break
      # if something goes wrong (e.g. response code 429) wait a short amount of time
      timeToWait = retry * 2
      retry += 1
      time.sleep(timeToWait)

    return requestSuccessful, urlData


  def parseResponseToJSON(self, data):
    parsedData = None
    try:
        parsedData = json.loads(data)
    except Exception as ex:
        pass
    if parsedData is None:
        print("\t >> Parsing data to json failed...")
    return parsedData

  """
  The related topics and related searches keywords contain the actual keyword and, 
  separated by an '-' symbol, the category this keyword is classified. 
  """
  def preprocessingRelatedKeyword(self, keyword):
    words = keyword.split(' - ')  # [0] is keyword, [1] is category
    words.append("")  # ensure at least two entries in list
    validKeyword = self.replaceInvalidKeywordchars(words[0])
    validCategory = self.replaceInvalidKeywordchars(words[1])
    return validKeyword, validCategory

    # def collectRelated(keyword, relatedSearchToken, relatedFlag, relatedText):
    # 	print('\t collecting ' + relatedText + ' keywords...')
    # 	validKeyword, validCategory = preprocessingRelatedKeyword(keyword)
    # 	resourceUrl = urlRelated\
    # 					.replace(paramSymbolPre + paramSymbolsRelated["keyword"] + paramSymbolPost, validKeyword)\
    # 					.replace(paramSymbolPre + paramSymbolsRelated["relatedType"] + paramSymbolPost, relatedFlag)\
    # 					.replace(paramSymbolPre + paramSymbolsRelated["token"] + paramSymbolPost, relatedSearchToken)
    # 	requestSuccessful, data = requestURLData(resourceUrl)
    # 	if not requestSuccessful or data is None:
    # 		return None
    # 	return data

  def collectData(self):
    keyword = self.node["keyword"]

    logging.info("Processing '" + keyword + "'...")

    resourceUrlExplore = self.urlTrendsExplore \
      .replace(self.prepareParam(self.paramSymbols["keyword"]), keyword) \
      .replace(self.prepareParam(self.paramSymbols["geoLocation"]), "")

    for times in self.timeBounderies:
      resourceUrl = resourceUrlExplore \
        .replace(self.prepareParam(self.paramSymbols["startTime"]), times[0]) \
        .replace(self.prepareParam(self.paramSymbols["endTime"]), times[1])

      requestSuccessful, data = self.requestURLData(resourceUrl)
      if not requestSuccessful or data is None:
        logging.error("Request for explore data failed...")
        return

      parsedData = self.parseResponseToJSON(data[5:])  # cut off the first 5 chars ")]}'\n"
      if parsedData is None:
        return
      for widget in parsedData["widgets"]:
        token = widget["token"]
        if widget["title"] == "Interesse im zeitlichen Verlauf":
          self.node["token"] = token
        elif widget["title"] == u"Ähnliche Suchanfragen":
          self.node["relatedSearches"] = {}
          self.node["relatedSearches"]["token"] = token

      resourceUrl = self.urlTrendsMultiline \
        .replace(self.prepareParam(self.paramSymbols["keyword"]), keyword) \
        .replace(self.prepareParam(self.paramSymbols["token"]), self.node["token"]) \
        .replace(self.prepareParam(self.paramSymbols["startTime"]), times[0]) \
        .replace(self.prepareParam(self.paramSymbols["endTime"]), times[1])
      requestSuccessful, courseData = self.requestURLData(resourceUrl)
      if not requestSuccessful or courseData is None:
        logging.error("Request for course data failed...")
        return

      parsedCourseData = self.parseResponseToJSON(courseData[5:])  # cut off the first 5 chars ")]}'\n"
      if parsedCourseData is None:
        return

      self.data.append(parsedCourseData)

  def getData(self):
    return self.data
