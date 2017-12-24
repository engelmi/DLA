# -*- coding: utf-8 -*-

import stockpredictorconfig as config
import preprocessing as pp

import sys
import tensorflow as tf
import numpy as np
from os.path import isfile, join
from os import listdir

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
        self.folderMergedData = join("datasets", "myDataset")
        self.folderPreprocessedData = join("datasets", "preprocessed")
        #self.modelSaver = tf.train.Saver()
        self.folderModelSafe = join("model")
        self.counter = 0
        self.loadPreprocessedData()

    def saveModel(self, session, filename):
        """
        Saves a trained model.
        :param session: The session to be saved.
        :param filename: The name of the file for the trained model.
        :return: The location of the saved model.
        """
        #return saver.save(session, join(self.folderModelSafe, filename))
        pass

    def restoreModel(self, session, filename):
        """
        Restores a previously trained model.
        :param session: The session where the model will be restored to.
        :param filename: The name of the file of the pre-trained model to restore.  
        """
        #saver.restore(session, join(self.folderModelSafe, filename))
        pass

    def preprocess(self):
        pp.preprocessing(self.folderPreprocessedData, self.folderMergedData, self.config.time_steps, self.config.values)

    def loadPreprocessedData(self):
        processedFiles = [f for f in listdir(self.folderPreprocessedData) if
                          isfile(join(self.folderPreprocessedData, f)) and f[-3:] == "npy"]
        first = True
        for pFile in processedFiles:
            loadedMatrix = pp.loadPreprocessedData(join(self.folderPreprocessedData, pFile))
            if first:
                self.data = loadedMatrix
                first = False
            else:
                self.data = np.concatenate((self.data, loadedMatrix))
        print(self.data.shape)

    def next_batch(self, batch_size):
        next_counter = self.counter + batch_size
        data_batch = self.data[self.counter:next_counter]
        self.counter = next_counter
        return data_batch


    def train_2(self, doPreprocessing):
        """
        Train the model.
        :param doPreprocessing: Boolean flag to indicate data is already preprocessed.
        """
        if doPreprocessing:
            self.preprocess()

        num_classes = 1
        X = tf.placeholder("float", [None, self.config.time_steps, self.config.values])
        Y = tf.placeholder("float", [None, num_classes])
        weights = tf.Variable(tf.random_normal([self.config.hidden_size, num_classes]))
        biases = tf.Variable(tf.random_normal([num_classes]))
        x = tf.unstack(X, self.config.time_steps, 1)
        lstm = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, forget_bias=1.0)
        outputs, states = tf.contrib.rnn.static_rnn(lstm, x, dtype=tf.float32)
        logits = tf.matmul(outputs[-1], weights + biases)
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

        with tf.Session() as sess:
            sess.run(init)
            lower_index = 0
            for step in range(1, self.config.training_steps + 1):
                data_batch = self.next_batch(self.config.batch_size)
                batch_x = data_batch[:, :, 1:]
                batch_x = batch_x.reshape((self.config.batch_size, self.config.time_steps, self.config.values))
                batch_y = data_batch[:, self.config.time_steps-1, 0]
                batch_y = batch_y.reshape((self.config.batch_size, 1))
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % 10 == 0 or step == 1:
                    p, loss, acc = sess.run([prediction, loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                    print("Step " + str(step))
                    print("Loss " + str(loss))
                    print("Accuracy " + str(acc))
                    print("Prediction " + str(p))

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
