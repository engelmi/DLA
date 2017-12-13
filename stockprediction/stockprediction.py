# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join

import preprocessing as pp
import stockpredictorconfig as config

import os
import csv
import sys
import shutil
import tensorflow as tf
import numpy as np

# flag to check if python3 is used
PY3 = sys.version_info[0] == 3

"""
Model class.
Used to predict the development of stock courses. 
"""
class Stockpredictor(object):

	"""
	Initialization. 
	"""
	def __init__(self, config):
		self.config = config
		self.folderMergedData = "datasets/myDataset"
		self.folderPreprocessedData = "datasets/preprocessed"

	"""
	Creates a BasicLSTMCell with a given size of units. 
	:param lstm_size: The size of the cell.
	"""
	def createLstmCell(self, lstm_size):
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
		
		

		lstm = self.createLstmCell(self.config.hidden_size)
		# Initial state of the lstm
		hidden_state = tf.zeros([self.config.batch_size, self.config.hidden_size], dtype=tf.float64)
		current_state = tf.zeros([self.config.batch_size, self.config.hidden_size], dtype=tf.float64)
		state = hidden_state, current_state
		probabilities = []
		loss = 0.0

		processedFiles = [f for f in listdir(self.folderPreprocessedData) if isfile(join(self.folderPreprocessedData, f)) and f[-3:] == "npy"]
		for pFile in processedFiles:
			loadedMatrix = self.loadPreprocessedData(join(self.folderPreprocessedData, pFile))
			num_of_datarows = loadedMatrix.shape[0]
			lower_index = 0
			while lower_index < num_of_datarows:
				upper_index = lower_index + self.config.batch_size
				# ensure dimension 0 is equal in both shapes (data_batch + state)
				if upper_index > num_of_datarows:
					break
				data_batch = tf.constant(loadedMatrix[lower_index : upper_index])
				# The value of state is updated after processing each batch of words.
				output, state = lstm(data_batch, state)	

				print(output)
				print(state)
				print("-------------------------------")
				# The LSTM output can be used to make next word predictions
				#logits = tf.matmul(output, softmax_w) + softmax_b
				#probabilities.append(tf.nn.softmax(logits))
				#loss += loss_function(probabilities, target_words)

				lower_index = upper_index
		
		return

	"""
	Predicts the course of a given stock. 
	:param stock: The stock whose development shall be predicted. 
	"""
	def predict(self, stock):
		pass



if __name__ == "__main__":
	config = config.StockpredictorConfig()
	sp = Stockpredictor(config)
	sp.train(True)
