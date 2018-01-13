# -*- coding: utf-8 -*-

import stockpredictorconfig as config
import preprocessing as pp
import simplelearningmodel as slm

import sys
import logging
import random
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

    def preprocess(self):
        """
        Preprocesses google trends and stock data.
        :return:
        """
        pp.preprocessing(self.folderPreprocessedData, self.folderMergedData, self.config.time_steps, self.config.values)

    def load_preprocessed_data(self, file_list):
        """
        Loads preprocessed data.
        :param file_list: List of files containing preprocessed data to be loaded.
        :return: The loaded data.
        """
        data = None
        for pFile in file_list:
            loadedMatrix = pp.load_preprocessed_data(join(self.folderPreprocessedData, pFile))
            if data is None:
                data = loadedMatrix
            else:
                data = np.concatenate((data, loadedMatrix))
        return data

    def train(self, doPreprocessing):
        """
        Train the model.
        :param doPreprocessing: Boolean flag to indicate data is already preprocessed.
        """
        if doPreprocessing:
            self.preprocess()

        build_succeeded = self.learning_model.build_graph()
        if not build_succeeded:
            logging.error("Error building learning model. Aborting...")
            return

        preProcessedFiles = [f for f in listdir(self.folderPreprocessedData) if
                          isfile(join(self.folderPreprocessedData, f)) and f[-3:] == "npy"]

        with tf.Session() as sess:
            sess.run(self.learning_model.get_init())

            fileIndices = [i for i in range(len(preProcessedFiles))]
            random.shuffle(fileIndices)
            validate_file_indices = fileIndices[0:int(len(preProcessedFiles) * self.config.validate_ratio)]
            train_and_test_indices = fileIndices[int(len(preProcessedFiles) * self.config.validate_ratio + 1) : len(preProcessedFiles)]

            validate_set = [preProcessedFiles[i] for i in validate_file_indices]
            train_and_test_set = [preProcessedFiles[i] for i in train_and_test_indices]

            for epoch in range(self.config.num_epochs):
                print("-------------------------" + str(epoch))
                fileIndices = [i for i in range(len(train_and_test_set))]
                random.shuffle(fileIndices)

                train_size = int(len(train_and_test_set) * self.config.train_test_ratio)

                training_files = [train_and_test_set[fileIndices[i]] for i in fileIndices[0:train_size]]
                test_files = [train_and_test_set[fileIndices[i]] for i in fileIndices[train_size + 1:]]

                training_file_data = self.load_preprocessed_data(training_files)
                test_file_data = self.load_preprocessed_data(test_files)

                self.learning_model.train(sess, training_file_data)
                self.learning_model.predict(sess, test_file_data)

            self.learning_model.save_model(sess, "trained")

    def predict(self, pathToFile, create_model = False, load_trained_model=None):
        """
        Predicts the course for a given stock.
        :param pathToFile: Path to the file which is used to predict the course development.
        :param create_model: Flag to indicate if the model needs to be recreated.
                             Not necessary if train() was called during execution of the stock predictor.
        :param load_trained_model: Flag to indicate if a trained model should be loaded.
        """
        if create_model:
            build_succeeded = self.learning_model.build_graph()
            if not build_succeeded:
                logging.error("Error building learning model. Aborting...")
                return

        with tf.Session() as sess:
            if load_trained_model:
                result = self.learning_model.restore_model(sess)
                if result:
                    print("restore successful")
                else:
                    print("restore NOT successful")

            data = self.load_preprocessed_data([pathToFile])
            self.learning_model.predict(sess, data)



if __name__ == "__main__":
    config = config.StockpredictorConfig()
    learning_model = slm.SimpleLearningModel(tf, config)
    sp = Stockpredictor(config, learning_model)
    sp.train(False)
    #sp.predict(False)
