
class LearningGraph(object):
    """
    Simple class of a learning graph. Contains all parameters necessary for a Learning Model.
    """

    def __init__(self):
        """
        Initialization.
        """
        self.graph_parameters = {}

    def set_graph_parameter(self, key, value):
        """
        Method to add a new parameter to the learning graph.
        :param key: Key for the parameter.
        :param value: Value of the parameter.
        :return: True if the parameter was added, otherwise False
        """
        if key not in self.graph_parameters:
            self.graph_parameters[key] = value
            return True
        return False

    def get_graph_parameter(self, key):
        """
        Method to retrieve a parameter from the learning graph.
        :param key: The parameter key of the value to be retrieved.
        :return: The value of the parameter if the dictionary contains the key, otherwise None.
        """
        if key in self.graph_parameters:
            return self.graph_parameters[key]
        return None

    def get_graph(self):
        """
        Method to get the whole set of parameters of the graph.
        :return: A dictionary containing all parameters of the graph.
        """
        return self.graph_parameters
