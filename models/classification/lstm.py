import tensorflow as tf
import tensorflow.keras as k
import numpy as np


class ClassifyLSTM(k.Model):
    def __init__(self, accent_classes):
        super(ClassifyLSTM, self).__init__()

        self.accent_classes = np.array(accent_classes)
        self.num_accent_classes = len(accent_classes)
        self.type = "classifier"

        self.optimizer = k.optimizers.Adam(3e-3)
        self.batch_size = 100
        self.embedding_size = 20

        # TODO: set up layers here
        self.rnn = tf.keras.layers.GRU(self.embedding_size, return_sequences=True, return_state=True)
        self.dense1 = tf.keras.layers.Dense(self.embedding_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.num_accent_classes, activation='softmax')

    @tf.function
    def call(self, inputs):
        """
        Run a forward pass of the LSTM classification model.
        :param inputs: Tensor or np array of size (batchSize, ..., ..., 1?)
        :return: Tensor of size (batchSize, numAccents)
        """
        rnn_output, state = self.rnn(inputs)
        return self.dense2(self.dense1(state))

    def get_class(self, inputs):
        """
        Given inputs, gets the class that is most likely (highest prob) for each.
        Useful for evaluation.
        :param inputs: A list of inputs to predict class for
        :return: The accent class that is most likely for the input
        """
        class_probs = self.call(inputs)
        return self.accent_classes[tf.argmax(class_probs, axis=1)]

    def accuracy(self, inputs, labels):
        """
        Given inputs and sparse labels (indices), calculates the accuracy of predictions.
        :param inputs: Batch of model inputs.
        :param labels: Batch of sparse label indices (1d tensor)
        :return: 0-1 accuracy value
        """
        predictions = tf.argmax(self.call(inputs), axis=1)
        return tf.reduce_mean(predictions == labels)

    def loss(self, inputs, labels):
        """
        Calculates the loss for some batch of input-label pairs.
        :param inputs: Tensor of size (batchSize, ..., ..., 1?)
        :param labels: Tensor of size (batchSize, 1) - sparse labels indicating index of accent
        :return: The average loss over this batch.
        """
        logits = self.call(inputs)
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))
