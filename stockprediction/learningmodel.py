import logging
from os.path import join

class LearningModel(object):
    """
    Abstract base class for learning models.
    Facilitates the use of creating and saving models as well as restoring persisted models.
    """

    def __init__(self, tf, save_folder, save_name):
        """
        Initialization.
        :param tf: Tensorflow import. Used to create the model.
        :param save_name: The name of the file this model is being persisted to.
        :param save_folder: The folder that contains the saved model files.
        """
        self.tf = tf
        self.save_name = save_name
        self.save_folder = save_folder

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

    def create_model(self):
        """
        Abstract method.
        Derived class must implement this method to create a learning model.
        :return:
        """
        raise NotImplementedError("Subclass must override get_model()!")

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