import logging

from learningmodel import LearningModel

class SimpleLearningModel(LearningModel):
    """
    Simple learning model class.
    """

    def __init__(self, tf, config, save_name="model", save_folder="model"):
        """
        Initialization.
        :param tf: Tensorflow import. Used to create the model.
        :param config: The stock prediction config.
        :param save_name: The name of the file this model is being persisted to.
        :param save_folder: The folder that contains the saved model files.
        """
        super(SimpleLearningModel, self).__init__(tf, save_folder, save_name)
        self.config = config

    def create_model(self):
        """

        :return:
        """
        try:
            X = self.tf.placeholder("float", [None, self.config.time_steps, self.config.values])
            Y = self.tf.placeholder("float", [None, self.config.num_classes])
            weights = self.tf.Variable(self.tf.random_normal([self.config.hidden_size, self.config.num_classes]))
            biases = self.tf.Variable(self.tf.random_normal([self.config.num_classes]))
            x = self.tf.unstack(X, self.config.time_steps, 1)
            lstm = self.tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, forget_bias=1.0)
            outputs, states = self.tf.contrib.rnn.static_rnn(lstm, x, dtype=self.tf.float32)
            logits = self.tf.matmul(outputs[-1], weights + biases)
            prediction = self.tf.nn.softmax(logits)
            loss_op = self.tf.reduce_mean(self.tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=Y))
            optimizer = self.tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
            train_op = optimizer.minimize(loss_op)

            # Evaluate model (with test logits, for dropout to be disabled)
            correct_pred = self.tf.equal(self.tf.argmax(prediction, 1), self.tf.argmax(Y, 1))
            accuracy = self.tf.reduce_mean(self.tf.cast(correct_pred, self.tf.float32))

            # Initialize the variables (i.e. assign their default value)
            init = self.tf.global_variables_initializer()

            return X, Y, outputs, states, prediction, loss_op, optimizer, train_op, correct_pred, accuracy, init
        except Exception as ex:
            logging.error(ex.message)
            return None
