import datetime
import logging
import numpy as np

from simplelearningmodel import SimpleLearningModel

class OnlyStockLearningModel(SimpleLearningModel):
    """
    Only Stock Learning Model class.
    """

    def __init__(self, tf, config, dropout_prob=None, save_name="model", save_folder="model", visualization_folder="logs"):
        """
        Initialization. The values property of the config are fixed to 1.
        :param tf: Tensorflow import. Used to create the model.
        :param config: The stock prediction config.
        :param save_name: The name of the file this model is being persisted to.
        :param save_folder: The folder that contains the saved model files.
        """
        super(OnlyStockLearningModel, self).__init__(tf, config, dropout_prob, save_name, save_folder, visualization_folder)
        self.config.values = 1