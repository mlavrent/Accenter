import tensorflow as tf
import tensorflow.keras as k
import numpy as np


class ClassifyCNN(k.Model):
    def __init__(self, accent_classes):
        super(ClassifyCNN, self).__init__()

        self.accent_classes = np.array(accent_classes)
        self.num_accent_classes = len(accent_classes)
        self.type = "classifier"

        self.optimizer = k.optimizers.Adam(3e-3)
        self.batch_size = 100

        self.conv1 = k.layers.Conv2D(filters=8,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding="SAME",
                                     activation=k.layers.LeakyReLU(0.2),
                                     use_bias=True,
                                     kernel_initializer=k.initializers.TruncatedNormal(stddev=0.1))
        self.conv2 = k.layers.Conv2D(filters=16,
                                     kernel_size=(4, 4),
                                     strides=(2, 2),
                                     padding="SAME",
                                     activation=k.layers.LeakyReLU(0.2),
                                     use_bias=True,
                                     kernel_initializer=k.initializers.TruncatedNormal(stddev=0.1))
        self.conv3 = k.layers.Conv2D(filters=32,
                                     kernel_size=(4, 4),
                                     strides=(2, 2),
                                     padding="SAME",
                                     activation=k.layers.LeakyReLU(0.2),
                                     use_bias=True,
                                     kernel_initializer=k.initializers.TruncatedNormal(stddev=0.1))
        self.flatten = k.layers.Flatten()

        self.dense1 = k.layers.Dense(units=256,
                                     activation=k.layers.LeakyReLU(0.2),
                                     use_bias=True,
                                     kernel_initializer=k.initializers.TruncatedNormal(stddev=0.1))
        self.dense2 = k.layers.Dense(units=self.num_accent_classes,
                                     use_bias=True,
                                     kernel_initializer=k.initializers.TruncatedNormal(stddev=0.1))

    @tf.function
    def call(self, inputs):
        """
        Run a forward pass of the CNN classification model.
        :param inputs: Tensor or np array of size (batchSize, ..., ..., 1?)
        :return: Tensor of size (batchSize, numAccents) containing logits
        """
        cur_calc = self.conv1(inputs)
        cur_calc = self.conv2(cur_calc)
        cur_calc = self.conv3(cur_calc)
        cur_calc = self.flatten(cur_calc)

        cur_calc = self.dense1(cur_calc)
        cur_calc = self.dense2(cur_calc)

        return cur_calc

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