# -*- coding: utf-8 -*-

import stockpredictorconfig as config
import preprocessing as pp
import simplelearningmodel as slm

import sys
import logging
import tensorflow as tf
import numpy as np
from os.path import isfile, join
from os import listdir

# flag to check if python3 is used
PY3 = sys.version_info[0] == 3


class Stockpredictor(object):
    """
    Predictor class.
    Used to predict the development of stock courses.
    """

    def __init__(self, spconfig, model):
        """
        Initialization.
        """
        self.config = spconfig
        self.learning_model = model
        self.folderMergedData = join("datasets", "myDataset")
        self.folderPreprocessedData = join("datasets", "preprocessed")
        self.load_preprocessed_data()

    def preprocess(self):
        """
        Preprocesses google trends and stock data.
        :return:
        """
        pp.preprocessing(self.folderPreprocessedData, self.folderMergedData, self.config.time_steps, self.config.values)

    def load_preprocessed_data(self):
        """
        Loads preprocessed data.
        :return:
        """
        processedFiles = [f for f in listdir(self.folderPreprocessedData) if
                          isfile(join(self.folderPreprocessedData, f)) and f[-3:] == "npy"]
        first = True
        for pFile in processedFiles:
            loadedMatrix = pp.load_preprocessed_data(join(self.folderPreprocessedData, pFile))
            if first:
                self.data = loadedMatrix
                first = False
            else:
                self.data = np.concatenate((self.data, loadedMatrix))

    def classes(self, value):
        """
        TODO
        :param value:
        :return:
        """
        if value == 1:
            return np.array([1, 0])
        elif value == -1 or value == 0:
            return np.array([0, 1])
        else:
            raise Exception("no valid classes")

    def next_batch(self):
        """
        TODO
        :return:
        """
        rands = np.random.randint(0, self.data.shape[0], self.config.batch_size)
        data_batch = self.data[rands]
        batch_x = data_batch[:, :, 1:]
        batch_x = batch_x.reshape((self.config.batch_size, self.config.time_steps, self.config.values))
        batch_y_tmp = data_batch[:, self.config.time_steps - 1, 0]
        first = True
        for i in range(0, self.config.batch_size):
            if first:
                batch_y = self.classes(batch_y_tmp[i])
                first = False
            else:
                batch_y = np.concatenate((batch_y, self.classes(batch_y_tmp[i])))

        batch_y = batch_y.reshape((self.config.batch_size, 2))
        return batch_x, batch_y

    def train(self, doPreprocessing):
        """
        Train the model.
        :param doPreprocessing: Boolean flag to indicate data is already preprocessed.
        """
        if doPreprocessing:
            self.preprocess()

        results = self.learning_model.create_model()
        if results is None:
            logging.error("Error creating learning model. Aborting...")
            return
        X, Y, outputs, states, prediction, loss_op, optimizer, train_op, correct_pred, accuracy, init = results

        with tf.Session() as sess:
            sess.run(init)
            for step in range(1, self.config.training_steps + 1):
                batch_x, batch_y = self.next_batch()
                sess.run([train_op], feed_dict={X: batch_x, Y: batch_y})
                if step % 100 == 0 or step == 1 or step == self.config.training_steps:
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                    print("Step " + str(step))
                    print("Loss " + str(loss))
                    print("Accuracy " + str(acc))
            self.learning_model.save_model(sess, "pretrained")

    def predict(self, create_model = False):
        """
        Predicts the course for a given stock.
        :param create_model: Flag to indicate if the model needs to be recreated.
                             Not necessary if train() was called during execution of the stock predictor.
        """

        if create_model:
            results = self.learning_model.create_model()
            if results is None:
                logging.error("Error creating learning model. Aborting...")
                return
            X, Y, outputs, states, prediction, loss_op, optimizer, train_op, correct_pred, accuracy, init = results

        with tf.Session() as sess:
            result = self.learning_model.restore_model(sess)
            if result:
                print("restore successful")
            else:
                print("restore NOT successful")



if __name__ == "__main__":
    config = config.StockpredictorConfig()
    learning_model = slm.SimpleLearningModel(tf, config)
    sp = Stockpredictor(config, learning_model)
    sp.train(False)
    sp.predict(False)
