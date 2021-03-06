import datetime
import logging
import numpy as np

from learningmodel import LearningModel

class SimpleLearningModel(LearningModel):
    """
    Simple Learning Model class.
    """

    def __init__(self, tf, config, dropout_prob=None, save_name="model", save_folder="model", visualization_folder="logs"):
        """
        Initialization.
        :param tf: Tensorflow import. Used to create the model.
        :param config: The stock prediction config.
        :param dropout_prob: Optional. If defined dropout is used with the given probability.
        :param save_name: The name of the file this model is being persisted to.
        :param save_folder: The folder that contains the saved model files.
        :param visualization_folder: The folder that contains the tensorboard data.
        """
        super(SimpleLearningModel, self).__init__(tf, save_folder, save_name, visualization_folder)
        self.config = config
        self.dropout_prob = dropout_prob

        self.manual_summary = None
        self.test_summary = None
        self.file_writer = None

    def build_graph(self):
        """
        Method to build the Simple Learning Model Graph. Sets the graph_built-Flag.
        :return: The current graph_built-Flag.
        """
        if not self.is_graph_built():
            try:
                X = self.tf.placeholder("float", [None, self.config.time_steps, self.config.values])
                Y = self.tf.placeholder("float", [None, self.config.num_classes])
                weights = self.tf.Variable(self.tf.random_normal([self.config.hidden_size, self.config.num_classes]))
                biases = self.tf.Variable(self.tf.random_normal([self.config.num_classes]))
                x = self.tf.unstack(X, self.config.time_steps, 1)
                lstm = self.tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, forget_bias=1.0)
                if self.dropout_prob is not None:
                    lstm = self.tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.dropout_prob)
                outputs, states = self.tf.contrib.rnn.static_rnn(lstm, x, dtype=self.tf.float32)
                logits = self.tf.matmul(outputs[-1], weights) + biases
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

                self.set_graph_parameters(init, X, Y, outputs, states, train_op, prediction, loss_op, optimizer, correct_pred, accuracy)
                self.graph_built = True
            except Exception as ex:
                logging.error(ex)

        return self.is_graph_built()

    def train(self, session, data):
        """
        Train the Simple Learning Model.
        :param session: The session of the current training.
        :param data: The data of the current epoch.
        """
        train_op = self.graph.get_graph_parameter("train_op")
        X = self.graph.get_graph_parameter("X")
        Y = self.graph.get_graph_parameter("Y")

        for batch in self.next_batch(data):
            batch_x, batch_y = batch
            session.run([train_op], feed_dict={X: batch_x, Y: batch_y})

    def predict(self, data):
        """
        Predicts the class for a given stock.
        :param session: The session of the current training.
        :param data: The data to predict
        :return: The class of the stock
        """
        X = self.graph.get_graph_parameter("X")
        prediction = self.graph.get_graph_parameter("predict_op")
        label = prediction.eval(feed_dict={X: data})
        return label

    def evaluate_k_iteration(self, session, data, epoch):
        """
        Evaluates the current k-Iteration.
        :param session: The session of the current training.
        :param data: The data of the current epoch.
        :param epoch: The current epoch.
        :return: A tuple (loss, accuracy) of the current evaluation.
        """
        build_succeeded = self.build_graph()
        if not build_succeeded:
            logging.error("Error creating learning model. Aborting...")
            return

        loss_op = self.graph.get_graph_parameter("loss_op")
        accuracy = self.graph.get_graph_parameter("accuracy_op")
        X = self.graph.get_graph_parameter("X")
        Y = self.graph.get_graph_parameter("Y")

        losses = []
        accuracies = []
        for batch in self.next_batch(data):
            batch_x, batch_y = batch
            summary, loss, acc = session.run([self.test_summary, loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            losses.append(loss)
            accuracies.append(acc)
        self.file_writer.add_summary(summary, epoch)

        return sum(losses)/len(losses), sum(accuracies)/len(accuracies)

    def evaluate_k_mean(self, loss, accuracy, epoch):
        """
        Evaluates the summary of the current epoch (based on the mean of k iterations).
        :param session: The session of the current training.
        :param data: The data of the current epoch.
        :param epoch: The current epoch.
        """
        self.manual_summary.value[0].simple_value = loss
        self.manual_summary.value[1].simple_value = accuracy
        self.file_writer.add_summary(self.manual_summary, epoch)

    def next_batch(self, data):
        """
        Method to processes the data and yields the next batch for training the Simple Learning Model.
        :param data: The data of which the next batch is being extracted.
        :return: Yields a tuple (batch_x, batch_y).
        """
        low = 0
        for step in range(data.shape[0] // self.config.batch_size + 1):
            high = low + self.config.batch_size
            if high > data.shape[0]:
                # shape is not allowed to be changed
                break
            data_batch = data[low:high,:,:]
            batch_x, batch_y = self.split_X_Y(data_batch)
            low = high
            yield batch_x, batch_y

    def split_X_Y(self, data):
        """
        Split the dataset in input and label.
        :param data: The dataset to split
        :return: The input and label two variables.
        """
        batch_x = data[:, :, 1:self.config.values+1]
        batch_y_tmp = data[:, self.config.time_steps - 1, 0]
        batch_x = batch_x.reshape((data.shape[0], self.config.time_steps, self.config.values))
        first = True
        for i in range(0, data.shape[0]):
            if first:
                batch_y = self.map_classes(batch_y_tmp[i])
                first = False
            else:
                batch_y = np.concatenate((batch_y, self.map_classes(batch_y_tmp[i])))
        batch_y = batch_y.reshape((data.shape[0], 2))
        return batch_x, batch_y

    def map_classes(self, value):
        """
        Method to map a class-value to a class-array.
        :param value: An integer value indicating one of the supported classes.
        :return: A numpy array where the position of the class is indicated by 1.
        """
        if value == 1:
            return np.array([1, 0])
        elif value == -1 or value == 0:
            return np.array([0, 1])
        else:
            raise Exception("no valid classes")

    def set_graph_parameters(self, init, X, Y, outputs, states, train_op, predict_op, loss_op, optimizer_op, correct_pred_op, accuracy_op):
        """
        Method to set all graph parameters.
        :param init: Initialized global variables.
        :param X: Input array.
        :param Y: Label array.
        :param outputs: Output array.
        :param states: State array.
        :param train_op: Training operation.
        :param predict_op: Predict operation.
        :param loss_op: Loss operation.
        :param optimizer_op: Optimize operation.
        :param correct_pred_op: Corrected predict operation.
        :param accuracy_op: Accuracy operation.
        """
        self.graph.set_graph_parameter("init", init)
        self.graph.set_graph_parameter("X", X)
        self.graph.set_graph_parameter("Y", Y)
        self.graph.set_graph_parameter("outputs", outputs)
        self.graph.set_graph_parameter("states", states)
        self.graph.set_graph_parameter("train_op", train_op)
        self.graph.set_graph_parameter("predict_op", predict_op)
        self.graph.set_graph_parameter("loss_op", loss_op)
        self.graph.set_graph_parameter("optimizer_op", optimizer_op)
        self.graph.set_graph_parameter("correct_pred_op", correct_pred_op)
        self.graph.set_graph_parameter("accuracy_op", accuracy_op)

    def get_init(self):
        """
        Getter.
        :return: The initialized global variables of the graph.
        """
        return self.graph.get_graph_parameter("init")

    def setup_visualization(self, session):
        """
        Method to set up the visualization for the Simple Learning Model.
        :param session: The session.
        """
        # summary based on graph variables
        with self.tf.name_scope(datetime.datetime.now().strftime("%Y-%m-%d--test")):

            with self.tf.name_scope('accuracy'):
                accuracy = self.graph.get_graph_parameter("accuracy_op")
            self.tf.summary.scalar('accuracy', accuracy)

            with self.tf.name_scope('loss'):
                loss = self.graph.get_graph_parameter("loss_op")
            self.tf.summary.scalar('loss', loss)

        self.test_summary = self.tf.summary.merge_all()

        # manual summary to visualize the k-mean for validation in tensorboard
        loss = None
        accuracy = None
        self.manual_summary = self.tf.Summary()
        self.manual_summary.value.add(tag='k-loss', simple_value=loss)
        self.manual_summary.value.add(tag='k-accuracy', simple_value=accuracy)

        self.file_writer = self.tf.summary.FileWriter(self.visualization_folder, session.graph)