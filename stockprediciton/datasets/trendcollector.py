# coding=utf8
from queue import Queue
from datetime import date, timedelta
from jsonmerge import Merger
from urllib import request
import json
import logging


class TrendCollector:
    """Class summarizing all googletrend requests for one stock."""

    # used symbols for parameter within the urls
    paramSymbols = {
        "keyword": "keyword",
        "geoLocation": "geo",
        "token": "token",
        "startTime": "startTime",
        "endTime": "endTime",
        "relatedType": "relatedType",
        "relatedStartTime": "relatedStartTime",
        "relatedEndTime": "relatedEndTime",
        "search": "QUERY"
    }

    # used symbols to indicate a parameter within an url
    paramSymbolPre = "{{"
    paramSymbolPost = "}}"

    def createParam(param):
        paramSymbolPre = "{{"
        paramSymbolPost = "}}"
        return paramSymbolPre + param + paramSymbolPost

    # url for retrieving the tokens for the following urls
    urlTrendsExplore = (
            'https://trends.google.com/trends/api/explore?hl=de&tz=-60&req=%7B%22comparisonItem%22:%5B%7B%22keyword%22:%22' +
            createParam(paramSymbols["keyword"]) + '%22,%22geo%22:%22' +
            createParam(paramSymbols["geoLocation"]) + '%22,%22time%22:%22' +
            createParam(paramSymbols["startTime"]) + '+' + createParam(paramSymbols["endTime"]) +
            '%22%7D%5D,%22category%22:0,%22property%22:%22%22%7D&tz=-60')

    # url for retrieving the course data
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
        """Initialize the trend for one stock.
        Keyword arguments:
        stock -- list, containing information about the stock (shortcut and name)
        startTime, endTime -- date, collecting data from startTime till endTime
        """
        self.timeBounderies = []
        self.data = []
        while startTime < endTime:
            stopTime = startTime + timedelta(days=90)
            if stopTime > date.today():
                stopTime = date.today()
            self.timeBounderies.append([startTime.strftime("%Y-%m-%d"), stopTime.strftime("%Y-%m-%d")])
            startTime = stopTime + timedelta(days=1)
        self.queue = Queue()
        self.queue.put(stock[0])
        self.queue.put(self.replaceInvalidKeywordchars(stock[1]))

    def prepareParam(self, param):
        """encapsulate param in {{<param>}}"""
        return self.paramSymbolPre + param + self.paramSymbolPost

    def replaceInvalidKeywordchars(self, keyword):
        """relplace chars not allowed in urls"""
        return keyword.replace(" ", "%20").replace("&", "%26").replace(".", "%2E")

    def requestURLData(self, requestURL, maxNumberOfRetries=3):
        """Do a HTTP request.

        Keyword arguements:
        requestURL -- string, the url to be requested.
        maxNumberOfRetries -- int, optional, defaults to 3
        """
        urlData = None
        requestSuccessful = False
        retry = 1
        while not requestSuccessful and retry <= maxNumberOfRetries:
            try:
                response = request.urlopen(requestURL)
                if response.getcode() != 429:
                    requestSuccessful = True
                    urlData = response.read()
                    break
            except Exception as ex:
                logging.error("Could not request url: " + ex)
            break
            # if something goes wrong (e.g. response code 429) wait a short amount of time
            timeToWait = retry * 2
            retry += 1
            time.sleep(timeToWait)
        return requestSuccessful, urlData

    def parseResponseToJSON(self, data):
        """strip of the first 5 chars as google's responses start with )]}'\n"""
        offset = ")]}'\n"
        return json.loads(data[len(offset):])

    def preprocessingRelatedKeyword(self, keyword):
        """
        The related topics and related searches keywords contain the actual keyword and,
        separated by an '-' symbol, the category this keyword is classified.
        """
        words = keyword.split(' - ')  # [0] is keyword, [1] is category
        words.append("")  # ensure at least two entries in list
        validKeyword = self.replaceInvalidKeywordchars(words[0])
        validCategory = self.replaceInvalidKeywordchars(words[1])
        return validKeyword, validCategory

    def collectRelated(self, keyword, relatedSearchToken, relatedFlag, time, trendiness):
        """Collect the related searches.

        Keyword arguments:
        keyword -- string, the keyword for which we are searching related words.
        relatedSearchToken -- string, the token from the explore
        relatedFlag -- string, indicating if we are searching related topics or searches
        time -- string[], the start and end time of the keyword we are processing
        trendiness -- string[], the time for which we compare the keyword with others
        """
        logging.info('collecting related keywords...')
        validKeyword, validCategory = self.preprocessingRelatedKeyword(keyword)
        resourceUrl = self.urlRelated \
            .replace(self.prepareParam(self.paramSymbols["keyword"]), validKeyword) \
            .replace(self.prepareParam(self.paramSymbols["relatedType"]), relatedFlag) \
            .replace(self.prepareParam(self.paramSymbols["token"]), relatedSearchToken) \
            .replace(self.prepareParam(self.paramSymbols["startTime"]), time[0]) \
            .replace(self.prepareParam(self.paramSymbols["endTime"]), time[1]) \
            .replace(self.prepareParam(self.paramSymbols["relatedStartTime"]), trendiness[0]) \
            .replace(self.prepareParam(self.paramSymbols["relatedEndTime"]), trendiness[1])
        requestSuccessful, data = self.requestURLData(resourceUrl)
        if not requestSuccessful or data is None:
            return None
        return data

    def requestData(self, keyword, related=False):
        """Request all data for one keyword.

        Keyword arguments:
        keyword -- string, the keyword we are processing.
        related -- boolean, should we look for related searches too? defaults to False.
        """
        logging.info("Processing '" + keyword + "'...")
        resourceUrlExplore = self.urlTrendsExplore \
            .replace(self.prepareParam(self.paramSymbols["keyword"]), keyword) \
            .replace(self.prepareParam(self.paramSymbols["geoLocation"]), "")

        for time in self.timeBounderies:
            resourceUrl = resourceUrlExplore \
                .replace(self.prepareParam(self.paramSymbols["startTime"]), time[0]) \
                .replace(self.prepareParam(self.paramSymbols["endTime"]), time[1])

            requestSuccessful, data = self.requestURLData(resourceUrl)
            if not requestSuccessful or data is None:
                logging.error("Request for explore data failed...")
                continue

            parsedData = self.parseResponseToJSON(data)
            if parsedData is None:
                continue

            timeLineToken = None
            relatedSearchToken = None
            trendiness = None
            for widget in parsedData["widgets"]:
                token = widget["token"]
                if widget["id"] == "TIMESERIES":
                    timeLineToken = token
                elif related and widget["id"] == "RELATED_QUERIES":
                    relatedSearchToken = token
                    trendiness = widget["request"]["trendinessSettings"]["compareTime"].split(' ')

            resourceUrl = self.urlTrendsMultiline \
                .replace(self.prepareParam(self.paramSymbols["keyword"]), keyword) \
                .replace(self.prepareParam(self.paramSymbols["token"]), timeLineToken) \
                .replace(self.prepareParam(self.paramSymbols["startTime"]), time[0]) \
                .replace(self.prepareParam(self.paramSymbols["endTime"]), time[1])
            requestSuccessful, courseData = self.requestURLData(resourceUrl)
            if not requestSuccessful or courseData is None:
                logging.error("Request for course data failed...")
                continue

            try:
                parsedCourseData = self.parseResponseToJSON(courseData)
                self.data.append((keyword, parsedCourseData))
            except Exception as ex:
                logging.error("Parsing data to json failed...")

        if related:
            relatedSearchData = self.collectRelated(keyword, relatedSearchToken, self.paramSymbols["search"], time,
                                                    trendiness)
            try:
                parsedSearchData = self.parseResponseToJSON(relatedSearchData)
                for keyword in parsedSearchData["default"]["rankedList"][0]["rankedKeyword"]:
                    self.queue.put(self.replaceInvalidKeywordchars(keyword["query"]))
            except Exception as ex:
                logging.error("Parsing data to json failed... " + ex.message)

    def collectData(self):
        """Collect all the data for this trend.
        The first to keywords are used to collect the related searches."""
        num_requests = 0
        while not self.queue.empty():
            keyword = self.queue.get()
            self.requestData(keyword, num_requests < 2)
            num_requests += 1

    def getData(self):
        """Return the data. All requests for each keyword get merged."""
        schema = {
            "oneOf": [
                {
                    "type": "array",
                    "mergeStrategy": "append"
                },
                {
                    "type": "object",
                    "additionalProperties": {
                        "$ref": "#"
                    }
                },
                {
                    "type": "string"
                },
            ]
        }
        merger = Merger(schema)
        mData = []
        k = self.data[0][0]
        j = {}
        for keyword, d in self.data:
            if k != keyword:
                mData.append((k, j))
                k = keyword
                j = {}
            j = merger.merge(j, d)
        mData.append((k, j))
        return mData
