# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join

import preprocessing as pp

import os
import csv
import sys
import shutil
import tensorflow as tf
import numpy as np

PY3 = sys.version_info[0] == 3

"""
Main class for the prediction of stock course development. 
"""
class StockPredictor(object):

	"""
	Initialization. 
	"""
	def __init__(self):
		self.folderMergedData = "datasets/myDataset"
		self.folderPreprocessedData = "datasets/preprocessed"

	"""
	Creates a BasicLSTMCell with a given size of units. 
	:param lstm_size: The size of the cell.
	"""
	def createLstmCell(lstm_size):
  		return tf.contrib.rnn.BasicLSTMCell(lstm_size)
	
	"""
	Preprocessing of the merged data. Reads the merged data of stocks and google trends, 
	applies preprocessing routines to it and writes the result out. 
    """
	def preprocessing(self):
		self.deletePreprocessedData(self.folderPreprocessedData)
		mergedCSVFiles = [f for f in listdir(self.folderMergedData) if isfile(join(self.folderMergedData, f)) and f[-3:] == "csv"]
		for csvfile in mergedCSVFiles:
			centeredData = pp.zeroCenter(join(self.folderMergedData, csvfile))
			# maybe further preprocessing...
			self.persistPreprocessedData(centeredData, self.folderPreprocessedData, csvfile[:len(csvfile)-4])
			d = self.loadPreprocessedData(join(self.folderPreprocessedData, csvfile[:len(csvfile)-4] + ".npy"))
			
	"""
	Persists a numpy matrix containing preprocessed data. 
	:param matrix: The numpy matrix to persist. 
	:param filepath: The relative path. 
	:param filename: The name of the file. 
	"""
	def persistPreprocessedData(self, matrix, filepath, filename):
		if not os.path.exists(filepath):
			os.makedirs(filepath)
		np.save(join(filepath, filename), matrix)

	"""
	Loads a persisted numpy matrix containing preprocessed data. 
	:param filepath: The relative path including the filename and extension. 
	"""
	def loadPreprocessedData(self, filepath):
		if os.path.exists(filepath):
			return np.matrix(np.load(filepath))
		return None

	"""
	Deletes the saved preprocessed data. 
	:param filepath: The relative path. 
	"""
	def deletePreprocessedData(self, filepath):
		if os.path.exists(filepath):
			shutil.rmtree(filepath)

	"""
	Train the model. 
	:param dataIsPreprocessed: Boolean flag to indicate data is already preprocessed. 
	"""
	def train(self, doPreprocessing):
		if doPreprocessing:
			self.preprocessing()
		
		return

	"""
	Predicts the course of a given stock. 
	:param stock: The stock whose development shall be predicted. 
	"""
	def predict(self, stock):
		pass



if __name__ == "__main__":
	sp = StockPredictor()
	sp.train(True)
