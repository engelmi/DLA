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


class Stockpredictor(object):
    """
    Model class.
    Used to predict the development of stock courses.
    """

    def __init__(self, config):
        """
        Initialization.
        """
        self.config = config
        self.folderMergedData = "datasets/myDataset"
        self.folderPreprocessedData = "datasets/preprocessed"


    def createLstmCell(self, lstm_size):
        """
        Creates a BasicLSTMCell with a given size of units.
        :param lstm_size: The size of the cell.
        """
        return tf.contrib.rnn.BasicLSTMCell(lstm_size)


    def preprocessing(self):
        """
        Preprocessing of the merged data. Reads the merged data of stocks and google trends,
        applies preprocessing routines to it and writes the result out.
        """
        self.deletePreprocessedData(self.folderPreprocessedData)
        mergedCSVFiles = [f for f in listdir(self.folderMergedData) if
                          isfile(join(self.folderMergedData, f)) and f[-3:] == "csv"]
        for csvfile in mergedCSVFiles:
            centeredData = pp.zeroCenter(join(self.folderMergedData, csvfile), (self.config.time_steps, self.config.values+1))
            # maybe further preprocessing...
            self.persistPreprocessedData(centeredData, self.folderPreprocessedData, csvfile[:len(csvfile) - 4])

    def persistPreprocessedData(self, matrix, filepath, filename):
        """
        Persists a numpy matrix containing preprocessed data.
        :param matrix: The numpy matrix to persist.
        :param filepath: The relative path.
        :param filename: The name of the file.
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        np.save(join(filepath, filename), matrix)

    def loadPreprocessedData(self, filepath):
        """
        Loads a persisted numpy matrix containing preprocessed data.
        :param filepath: The relative path including the filename and extension.
        """
        if os.path.exists(filepath):
            return np.array(np.load(filepath))
        return None

    def deletePreprocessedData(self, filepath):
        """
        Deletes the saved preprocessed data.
        :param filepath: The relative path.
        """
        if os.path.exists(filepath):
            shutil.rmtree(filepath)

    def train_2(self, doPreprocessing):
        """
        Train the model.
        :param doPreprocessing: Boolean flag to indicate data is already preprocessed.
        """
        if doPreprocessing:
            self.preprocessing()

        num_classes = 30
        X = tf.placeholder("float", [None, self.config.time_steps, self.config.values])
        Y = tf.placeholder("float", [None, num_classes])
        weights = {
            'out': tf.Variable(tf.random_normal([self.config.hidden_size, num_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([num_classes]))
        }
        x = tf.unstack(X, self.config.time_steps, 1)
        lstm = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, forget_bias=1.0)
        outputs, states = tf.contrib.rnn.static_rnn(lstm, x, dtype=tf.float32)
        logits = tf.matmul(outputs[-1], weights['out'] + biases['out'])
        prediction = tf.nn.softmax(logits)
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        processedFiles = [f for f in listdir(self.folderPreprocessedData) if
                          isfile(join(self.folderPreprocessedData, f)) and f[-3:] == "npy"]
        with tf.Session() as sess:
            sess.run(init)
            for pFile in processedFiles:
                loadedMatrix = self.loadPreprocessedData(join(self.folderPreprocessedData, pFile))
                num_of_sets = loadedMatrix.shape[0]
                lower_index = 0
                for set in range(0, num_of_sets-self.config.batch_size, self.config.batch_size):
                    data_batch = loadedMatrix[set:set+self.config.batch_size]
                    batch_x = data_batch[:, :, 1:]
                    batch_x = batch_x.reshape((self.config.batch_size, self.config.time_steps, self.config.values))
                    batch_y = data_batch[:, :, 0]
                    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                    if set % 10 == 0 or set == 1:
                        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                        print("Step " + str(set))
                        print("Loss " + str(loss))
                        print("Accuracy " + str(acc))

        return

    def train(self, doPreprocessing):
        """
        Train the model.
        :param dataIsPreprocessed: Boolean flag to indicate data is already preprocessed.
        """
        if doPreprocessing:
            self.preprocessing()

        lstm = self.createLstmCell(self.config.hidden_size)
        # Initial state of the lstm
        hidden_state = tf.zeros([self.config.batch_size, self.config.hidden_size], dtype=tf.float64)
        current_state = tf.zeros([self.config.batch_size, self.config.hidden_size], dtype=tf.float64)
        state = hidden_state, current_state
        probabilities = []
        loss = 0.0

        processedFiles = [f for f in listdir(self.folderPreprocessedData) if
                          isfile(join(self.folderPreprocessedData, f)) and f[-3:] == "npy"]
        for pFile in processedFiles:
            loadedMatrix = self.loadPreprocessedData(join(self.folderPreprocessedData, pFile))
            num_of_sets = loadedMatrix.shape[0]
            lower_index = 0
            for set in range(0, num_of_sets):
                data_batch = tf.constant(loadedMatrix[set])
                # The value of state is updated after processing each batch of words.
                output, state = lstm(data_batch, state)

                print(output)
                print(state)
                print("-------------------------------")
                # The LSTM output can be used to make next word predictions
                # logits = tf.matmul(output, softmax_w) + softmax_b
                # probabilities.append(tf.nn.softmax(logits))
                # loss += loss_function(probabilities, target_words)


        return

    def predict(self, stock):
        """
        Predicts the course of a given stock.
        :param stock: The stock whose development shall be predicted.
        """
        pass


if __name__ == "__main__":
    config = config.StockpredictorConfig()
    sp = Stockpredictor(config)
    sp.train_2(True)
