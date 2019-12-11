import tensorflow as tf
import tensorflow.keras as k
import numpy as np

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    BatchNormalization, MaxPooling2D, Dropout, GRU


class ClassifyGCNN(Model):
    def __init__(self, accent_classes):
        super(ClassifyGCNN, self).__init__()

        self.accent_classes = np.array(accent_classes)
        self.num_accent_classes = len(accent_classes)
        self.type = "classifier"

        self.optimizer = k.optimizers.Adam(1e-3)
        self.batch_size = 100
        self.embedding_size = 96

        self.CNN = Sequential()
        self.CNN.add(Conv2D(16, (3, 3), strides=(1, 1),
                            padding='same',
                            kernel_regularizer=k.regularizers.l1_l2(l1=0.01, l2=0.01),
                            activation='relu'))
        self.CNN.add(BatchNormalization())
        self.CNN.add(MaxPooling2D(pool_size=(2, 2),
                                  padding='valid'))
        self.CNN.add(Dropout(0.4))

        self.CNN.add(Conv2D(16, (3, 3), strides=(1, 1),
                            padding='same',
                            kernel_regularizer=k.regularizers.l1_l2(l1=0.01, l2=0.01),
                            activation='relu'))
        self.CNN.add(BatchNormalization())
        self.CNN.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        self.CNN.add(Dropout(0.4))

        self.RNN = GRU(self.embedding_size, dropout=0.4,
                       return_sequences=True, return_state=True)

        self.Dense = Sequential()
        self.Dense.add(Dense(256, activation='relu',
                             kernel_regularizer=k.regularizers.l1_l2(l1=0.01, l2=0.01)))
        self.Dense.add(Dropout(0.4))
        self.Dense.add(Dense(self.num_accent_classes, activation='softmax',
                             kernel_regularizer=k.regularizers.l1_l2(l1=0.01, l2=0.01)))

    @tf.function
    def call(self, inputs):
        """
        Run a forward pass of the CNN classification model.
        :param inputs: Tensor or np array of size (batchSize, ..., ..., 1?)
        :return: Tensor of size (batchSize, numAccents) containing probabilities
        """
        cnn_out = self.CNN(inputs)
        cnn_out = tf.reshape(cnn_out, [inputs.shape[0], cnn_out.shape[1], -1])
        _, state = self.RNN(cnn_out)

        return self.Dense(state)
    
    def get_class(self, inputs):
        """
        Given inputs, gets the class that is most likely (highest prob) for
        each. Useful for evaluation.
        :param inputs: A list of inputs to predict class for
        :return: The accent class that is most likely for the input
        """
        class_probs = self.call(inputs)
        return self.accent_classes[tf.argmax(class_probs, axis=1)]

    def accuracy(self, inputs, labels):
        """
        Given inputs and sparse labels (indices), calculates the accuracy of
        predictions.
        :param inputs: Batch of model inputs.
        :param labels: Batch of sparse label indices (1d tensor)
        :return: 0-1 accuracy value
        """
        predictions = tf.argmax(self.call(inputs), axis=1)
        return tf.reduce_mean(tf.cast(predictions == labels, tf.float32))

    def loss(self, inputs, labels):
        """
        Calculates the loss for some batch of input-label pairs.
        :param inputs: Tensor of size (batchSize, ..., ..., 1?)
        :param labels: Tensor of size (batchSize, 1) - sparse labels indicating
                        index of accent
        :return: The average loss over this batch.
        """
        probabilities = self.call(inputs)
        return tf.reduce_mean(
            k.losses.sparse_categorical_crossentropy(labels, probabilities))
        

class ClassifyCNN(Model):
    def __init__(self, accent_classes):
        super(ClassifyCNN, self).__init__()

        self.accent_classes = np.array(accent_classes)
        self.num_accent_classes = len(accent_classes)
        self.type = "classifier"

        self.optimizer = k.optimizers.Adam(3e-3)
        self.batch_size = 90

        self.model = Sequential()
        self.model.add(Conv2D(16, (3, 3), strides=(1, 1),
                              padding='same',
                              kernel_regularizer=k.regularizers.l1_l2(l1=0.02, l2=0.02),
                              activation='relu'))
        # self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2),
                                    padding='valid'))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(16, (3, 3), strides=(1, 1),
                              padding='same',
                              kernel_regularizer=k.regularizers.l1_l2(l1=0.02, l2=0.02),
                              activation='relu'))
        # self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())

        # self.model.add(Dense(256, activation='relu',
        #                      kernel_regularizer=k.regularizers.l1_l2(l1=0.02, l2=0.02)))
        # self.model.add(Dropout(0.7))
        self.model.add(Dense(self.num_accent_classes, activation='softmax',
                             kernel_regularizer=k.regularizers.l1_l2(l1=0.02, l2=0.02)))

    @tf.function
    def call(self, inputs):
        """
        Run a forward pass of the CNN classification model.
        :param inputs: Tensor or np array of size (batchSize, ..., ..., 1?)
        :return: Tensor of size (batchSize, numAccents) containing probabilities
        """
        return self.model(inputs)

    def get_class(self, inputs):
        """
        Given inputs, gets the class that is most likely (highest prob) for
        each. Useful for evaluation.
        :param inputs: A list of inputs to predict class for
        :return: The accent class that is most likely for the input
        """
        class_probs = self.call(inputs)
        return self.accent_classes[tf.argmax(class_probs, axis=1)]

    def accuracy(self, inputs, labels):
        """
        Given inputs and sparse labels (indices), calculates the accuracy of
        predictions.
        :param inputs: Batch of model inputs.
        :param labels: Batch of sparse label indices (1d tensor)
        :return: 0-1 accuracy value
        """
        predictions = tf.argmax(self.call(inputs), axis=1)
        return tf.reduce_mean(tf.cast(predictions == labels, tf.float32))

    def loss(self, inputs, labels):
        """
        Calculates the loss for some batch of input-label pairs.
        :param inputs: Tensor of size (batchSize, ..., ..., 1?)
        :param labels: Tensor of size (batchSize, 1) - sparse labels indicating
                        index of accent
        :return: The average loss over this batch.
        """
        probabilities = self.call(inputs)
        return tf.reduce_mean(
            k.losses.sparse_categorical_crossentropy(labels, probabilities))
