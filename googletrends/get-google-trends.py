# -*- coding: utf-8 -*-

import os
import csv
import urllib2
import json
import time
from datetime import datetime, timedelta

# used symbols to indicate a parameter within an url
paramSymbolPre = "{{"
paramSymbolPost = "}}"

# used symbols for parameter within the urlTrendsExplore
paramSymbolsExplore = {
	"keyword" : "keyword", 
	"geolocation" : "geo", 
	"time" : {
		"5-years" : ["today+5-y", 365*5],
		"1-year" : ["today+12-m", 365],
		"3-months" : ["today+3-m", 1*30*3],
		"1-month" : ["today+1-m", 1*30],
		"1-week" : ["now+7-d", 1*7],
		"1-day" : ["now+1-d", 1]
	}
}

timeToUse = paramSymbolsExplore["time"]["1-year"]		# time for which the data is collected
timeUpperBound = datetime.now().strftime("%Y-%m-%d")	# lower bound of the timeframe
timeLowerBound = (datetime.now() - timedelta(days=timeToUse[1])).strftime("%Y-%m-%d")	# upper bound of the timeframe
relatedTimeLowerBound = (datetime.now() - timedelta(days=(timeToUse[1] * 2) + 1 )).strftime("%Y-%m-%d")	# lower bound of the timeframe for related compare time
relatedTimeUpperBound = (datetime.now() - timedelta(days=timeToUse[1] + 1)).strftime("%Y-%m-%d")	# upper bound of the timeframe for related compare time

urlTrendsExplore = ('https://trends.google.com/trends/api/explore?hl=de&tz=-60&req=%7B%22comparisonItem%22:%5B%7B%22keyword%22:%22' + 
					paramSymbolPre + paramSymbolsExplore["keyword"] + paramSymbolPost + 
					'%22,%22geo%22:%22' +
					paramSymbolPre + paramSymbolsExplore["geolocation"] + paramSymbolPost + 
					'%22,%22time%22:%22' + 
					timeToUse[0] + 
					'%22%7D%5D,%22category%22:0,%22property%22:%22%22%7D&tz=-60')

# used symbols for parameter within the urlTrendsMultiline (course data)
paramSymbolsMultiline = {
	"keyword" : "keyword", 
	"token" : "token"
}
urlTrendsMultiline = ('https://trends.google.com/trends/api/widgetdata/multiline?hl=de&tz=-60&req=%7B%22time%22:%22' +
						timeLowerBound + '+' + timeUpperBound +
						'%22,%22resolution%22:%22WEEK%22,%22locale%22:%22de%22,%22comparisonItem%22:%5B%7B%22geo%22:%7B%7D,%22complexKeywordsRestriction%22:%7B%22keyword%22:%5B%7B%22type%22:%22BROAD%22,%22value%22:%22' +
						paramSymbolPre + paramSymbolsMultiline["keyword"] + paramSymbolPost + 
						'%22%7D%5D%7D%7D%5D,%22requestOptions%22:%7B%22property%22:%22%22,%22backend%22:%22IZG%22,%22category%22:0%7D%7D&token=' + 
						paramSymbolPre + paramSymbolsMultiline["token"] + paramSymbolPost + 
						'&tz=-60')

# used symbols for parameter within the urlRelated (relatedTopics and relatedSearches keywords)
paramSymbolsRelated = {
	"keyword" : "keyword",
	"relatedType" : "relatedType",
	"token" : "token"
}
relatedTypes = {
	"topic" : "ENTITY",
	"search" : "QUERY"
}
# topics and searches share the same url, the token distinguishes the requests
urlRelated = ('https://trends.google.com/trends/api/widgetdata/relatedsearches?hl=de&tz=-60&req=%7B%22restriction%22:%7B%22geo%22:%7B%7D,%22time%22:%22' +
				timeLowerBound + '+' + timeUpperBound + 
				'%22,%22complexKeywordsRestriction%22:%7B%22keyword%22:%5B%7B%22type%22:%22BROAD%22,%22value%22:%22' +
				paramSymbolPre + paramSymbolsRelated["keyword"] + paramSymbolPost + 
				'%22%7D%5D%7D%7D,%22keywordType%22:%22' +
				paramSymbolPre + paramSymbolsRelated["relatedType"] + paramSymbolPost + 
				'%22,%22metric%22:%5B%22TOP%22,%22RISING%22%5D,%22trendinessSettings%22:%7B%22compareTime%22:%22' +
				relatedTimeLowerBound + '+' + relatedTimeUpperBound + 
				'%22%7D,%22requestOptions%22:%7B%22property%22:%22%22,%22backend%22:%22IZG%22,%22category%22:0%7D,%22language%22:%22de%22%7D&token=' + 
				paramSymbolPre + paramSymbolsRelated["token"] + paramSymbolPost)



"""
TODO:
"""
def replaceInvalidKeywordchars(keyword):
	return keyword.replace(" ", "%20").replace("&", "%26").replace(".", "%2E")

"""
TODO:
"""
def requestURLData(requestURL):
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

"""
TODO:
"""
def parseResponseToJSON(data):
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
def preprocessingRelatedKeyword(keyword):
	words = keyword.split(' - ') 	# [0] is keyword, [1] is category
	words.append("")					# ensure at least two entries in list
	validKeyword = replaceInvalidKeywordchars(words[0])
	validCategory = replaceInvalidKeywordchars(words[1])
	return validKeyword, validCategory

"""
TODO:
"""
def collectRelated(keyword, relatedSearchToken, relatedFlag, relatedText):
	print('\t collecting ' + relatedText + ' keywords...')
	validKeyword, validCategory = preprocessingRelatedKeyword(keyword)
	resourceUrl = urlRelated\
					.replace(paramSymbolPre + paramSymbolsRelated["keyword"] + paramSymbolPost, validKeyword)\
					.replace(paramSymbolPre + paramSymbolsRelated["relatedType"] + paramSymbolPost, relatedFlag)\
					.replace(paramSymbolPre + paramSymbolsRelated["token"] + paramSymbolPost, relatedSearchToken)
	requestSuccessful, data = requestURLData(resourceUrl)
	if not requestSuccessful or data is None:
		return None
	return data

"""
TODO:
"""
def collectGoogleTrendsData():

	with open('securities.csv', 'rb') as csvfile: 
		reader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
		for row in reader:

			node = {
				"symbol" : row[0],
				"keyword" : replaceInvalidKeywordchars(row[1]),
				"sector" : row[3],
				"subindustry" : row[4]
			}
			keyword = node["keyword"]
			
			print("Processing '" + keyword + "'...")

			resourceUrl = urlTrendsExplore\
							.replace(paramSymbolPre + paramSymbolsExplore["keyword"] + paramSymbolPost, keyword)\
							.replace(paramSymbolPre + paramSymbolsExplore["geolocation"] + paramSymbolPost, "")

			print("\t requesting explore data...")
			requestSuccessful, data = requestURLData(resourceUrl)
			if not requestSuccessful or data is None:
				print("\t >> Request for explore data failed...")
				continue

			print("\t parsing explore data...")
			parsedData = parseResponseToJSON(data[5:])	# cut off the first 5 chars ")]}'\n"
			if parsedData is None:
				continue

			print("\t extracting request tokens...")
			for widget in parsedData["widgets"]:
				token = widget["token"]
				if widget["title"] == "Interesse im zeitlichen Verlauf":
					node["token"] = token
				elif widget["title"] == "Verwandte Themen":
					node["relatedTopics"] = {}
					node["relatedTopics"]["token"] = token
				elif widget["title"] == u"Ähnliche Suchanfragen":
					node["relatedSearches"] = {}
					node["relatedSearches"]["token"] = token

			print("\t requesting course data...")
			resourceUrl = urlTrendsMultiline\
							.replace(paramSymbolPre + paramSymbolsMultiline["keyword"] + paramSymbolPost, keyword)\
							.replace(paramSymbolPre + paramSymbolsMultiline["token"] + paramSymbolPost, node["token"])
			requestSuccessful, courseData = requestURLData(resourceUrl)
			relatedSearchData = collectRelated(keyword, node["relatedSearches"]["token"], relatedTypes["search"], 'related search')
			relatedTopicData = collectRelated(keyword, node["relatedTopics"]["token"], relatedTypes["topic"], 'related topic')
			if not requestSuccessful or courseData is None:
				print("\t >> Request for course data failed...")
				continue
			elif relatedSearchData is None:
				print("\t >> Request for related search data failed...")
				continue
			elif relatedTopicData is None:
				print("\t >> Request for related topic data failed...")
				continue

			print("\t parsing course data...")
			parsedCourseData = parseResponseToJSON(courseData[5:])	# cut off the first 5 chars ")]}'\n"
			if parsedCourseData is None:
				continue
			print("\t parsing related search data...")
			parsedSearchData = parseResponseToJSON(relatedSearchData[5:])	# cut off the first 5 chars ")]}'\n"
			if parsedSearchData is None:
				continue
			print("\t parsing related topic data...")
			parsedTopicData = parseResponseToJSON(relatedTopicData[5:])	# cut off the first 5 chars ")]}'\n"
			if parsedTopicData is None:
				continue

			print("\t creating folder and saving data to files...")
			path = "keyword-" + keyword
			try:
				os.mkdir(path)
				with open(os.path.join(path,"course.csv"), "w") as f:
					f.write(json.dumps(parsedCourseData))
			except Exception as ex:
				print("\t error writing course data: " + ex.message)
			
			try:
				with open(os.path.join(path,"relatedSearches.csv"), "w") as f:
					f.write(json.dumps(parsedSearchData))
			except Exception as ex:
				print("\t error writing related search data: " + ex.message)

			try:
				with open(os.path.join(path,"relatedTopics.csv"), "w") as f:
					f.write(json.dumps(parsedTopicData))
			except Exception as ex:
				print("\t error writing related topic data: " + ex.message)


collectGoogleTrendsData()
