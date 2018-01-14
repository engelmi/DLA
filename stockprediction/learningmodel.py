import logging
from os.path import join

from LearningGraph import LearningGraph

class LearningModel(object):
    """
    Abstract base class for learning models.
    Facilitates the use of creating and saving models as well as restoring persisted models.
    """

    def __init__(self, tf, save_folder, save_name, visualization_folder):
        """
        Initialization.
        :param tf: Tensorflow import. Used to create the model.
        :param save_name: The name of the file this model is being persisted to.
        :param save_folder: The folder that contains the saved model files.
        :param visualization_folder: The folder to store the visualized information.
        """
        self.tf = tf

        self.save_name = save_name
        self.save_folder = save_folder
        self.visualization_folder = visualization_folder

        self.graph = LearningGraph()
        self.graph_built = False

    def get_save_name(self):
        """
        Getter.
        :return: The file name of the persisted model.
        """
        return self.save_name

    def get_save_folder(self):
        """
        Getter.
        :return: The name of the folder which contains the saved model files.
        """
        return self.save_folder

    def get_save_path(self):
        """
        Getter.
        :return: The complete path to the saved model files.
        """
        return join(self.save_folder, self.save_name)

    def build_graph(self):
        """
        Abstract method.
        Derived class must implement this method to create a learning model.
        :return:
        """
        raise NotImplementedError("Subclass must override build_graph()!")

    def is_graph_built(self):
        """
        Method to check if the graph of the model has been built already.
        :return: Boolean-Flag which indicates if the graph of the model has been built already.
        """
        return self.graph_built

    def get_graph(self):
        """
        Getter for the Learning Graph.
        :return: The currently used Learning Graph.
        """
        return self.graph

    def train(self, session, data):
        """
        Abstract method.
        Derived class must implement this method to train the learning model.
        :param session: The session of the current training.
        :param data: The data of the current epoch.
        """
        raise NotImplementedError("Subclass must override train()!")

    def predict(self, session, data, epoch):
        """
        Abstract method.
        Derived class must implement this method to make predictions based on the trained data.
        :param session: The session of the current training.
        :param data: The data of the current epoch.
        :param epoch: The current epoch.
        """
        raise NotImplementedError("Subclass must override train()!")

    def next_batch(self, data):
        """
        Abstract method.
        Derived class must implement this method to process the next batch during training.
        :param data: The data of which the next batch is being extracted.
        :return: Implementing subclass should return a tuple (batch_x,batch_y).
        """
        raise NotImplementedError("Subclass must override next_batch()!")

    def setup_visualization(self, session):
        """
        Abstract method.
        Method to set up the visualization for the Learning Model.
        :param session: The session.
        """
        raise NotImplementedException("Subclass must override setup_visualization()!")

    def save_model(self, session, filename):
        """
        Saves a trained model.
        :param session: The session used for saving.
        :param filename: The name of the file for the trained model.
        :return: The location of the saved model.
        """
        try:
            saver = self.tf.train.Saver()
            saver.save(session, join(self.save_folder, self.save_name))
            return True
        except Exception as ex:
            logging.error(ex.message)
            return False

    def restore_model(self, session):
        """
        Restores a previously trained model.
        :param session: The session where the model will be restored to.
        """
        try:
            saver = self.tf.train.Saver()
            saver.restore(session, join(self.save_folder, self.save_name))
            return True
        except Exception as ex:
            logging.error(ex.message)
            return False
