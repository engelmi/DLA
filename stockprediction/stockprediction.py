 # -*- coding: utf-8 -*-

import stockpredictorconfig as config
import preprocessing as pp
import simplelearningmodel as slm
import onlystocklearningmodel as oslm

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
        self.folderTestData = join("datasets", "testData")

    def preprocess(self):
        """
        Preprocesses google trends and stock data.
        :return:
        """
        pp.preprocessing(self.folderPreprocessedData, self.folderMergedData, self.config.time_steps, self.config.values)

    def load_preprocessed_data(self, file_list, folder=None):
        """
        Loads preprocessed data.
        :param file_list: List of files containing preprocessed data to be loaded.
        :param folder: The folder to look for files.
        :return: The loaded data.
        """
        data = None
        if not folder:
            folder = self.folderPreprocessedData
        for pFile in file_list:
            loadedMatrix = pp.load_preprocessed_data(join(folder, pFile))
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
        # load the file data only once...
        file_data = self.load_preprocessed_data(preProcessedFiles)
        chunk_size = file_data.shape[0] // self.config.cross_validation_k

        with tf.Session() as sess:
            self.learning_model.setup_visualization(sess)
            sess.run(self.learning_model.get_init())

            for epoch in range(self.config.num_epochs):
                print("------------> Epoch: " + str(epoch))

                k_losses = []
                k_accuracies = []
                files = file_data
                for k in range(self.config.cross_validation_k):
                    print("------------>    k-Iteration: " + str(k))
                    training_file_data = files[chunk_size:]
                    test_file_data = files[:chunk_size]

                    self.learning_model.train(sess, training_file_data)
                    loss, acc = self.learning_model.evaluate_k_iteration(sess, test_file_data, epoch)
                    k_losses.append(loss)
                    k_accuracies.append(acc)

                    # rotate train/test files
                    files = np.concatenate((files[chunk_size:],files[:chunk_size]))
                mean_k_loss = sum(k_losses)/len(k_losses)
                mean_k_accuracy = sum(k_accuracies) / len(k_accuracies)
                self.learning_model.evaluate_k_mean(mean_k_loss, mean_k_accuracy, epoch)


            self.learning_model.save_model(sess, "trained")

    def predict(self, create_model = False, load_trained_model=None):
        """
        Predicts the course for a given stock.
        :param create_model: Flag to indicate if the model needs to be recreated.
                             Not necessary if train() was called during execution of the stock predictor.
        :param load_trained_model: Flag to indicate if a trained model should be loaded.
		:return: A tuple (corrected prediction, accuracy). 
        """
        pathToFile = [f for f in listdir(self.folderTestData) if
                   isfile(join(self.folderTestData, f)) and f[-3:] == "npy"]
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

            data = self.load_preprocessed_data(pathToFile, self.folderTestData)
            input, y = self.learning_model.split_X_Y(data)
            pred = self.learning_model.predict(input)
            cor_pred = np.equal(np.argmax(y, 1), np.argmax(pred, 1))
            print(cor_pred)
            acc = np.mean(cor_pred)
            print(acc)
            return cor_pred, acc


if __name__ == "__main__":
    config = config.StockpredictorConfig()
    learning_model = slm.SimpleLearningModel(tf, config, dropout_prob=0.3, save_name="model", save_folder="trainedModels/SLM/experiment4", visualization_folder="logs/SLM/experiment4")
    sp = Stockpredictor(config, learning_model)
    #sp.train(False)
    pred_acc_list = []
    for i in range(10):
        _, acc = sp.predict(True, True)
        pred_acc_list.append(acc)
    print(sum(pred_acc_list)/len(pred_acc_list))
