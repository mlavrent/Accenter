import tensorflow as tf
import tensorflow.keras as k


class ClassifyCNN(k.Model):
    def __init__(self, accent_classes):
        super(ClassifyCNN, self).__init__()

        self.accent_classes = accent_classes
        self.num_accent_classes = len(self.accent_classes)

        # TODO: set up layers here
        self.conv1 = k.layers.Conv2D(filters=...,
                                     kernel_size=...,
                                     strides=...,
                                     padding="SAME",
                                     activation=...,
                                     use_bias=True,
                                     kernel_initializer=...)
        self.conv2 = k.layers.Conv2D(filters=...,
                                     kernel_size=...,
                                     strides=...,
                                     padding="SAME",
                                     activation=...,
                                     use_bias=True,
                                     kernel_initializer=...)
        self.conv3 = k.layers.Conv2D(filters=...,
                                     kernel_size=...,
                                     strides=...,
                                     padding="SAME",
                                     activation=...,
                                     use_bias=True,
                                     kernel_initializer=...)

        self.dense1 = k.layers.Dense(units=...,
                                     activation=...,
                                     use_bias=True,
                                     kernel_initializer=...)
        self.dense2 = k.layers.Dense(units=self.num_accent_classes,
                                     activation=...,
                                     use_bias=True,
                                     kernel_initializer=...)

    @tf.function
    def call(self, inputs):
        """
        Run a forward pass of the CNN classification model.
        :param inputs: Tensor or np array of size (batchSize, ..., ..., 1?)
        :return: Tensor of size (batchSize, numAccents) containing logits
        """
        ...

    def get_class(self, inp):
        """
        Given an input, gets the class that is most likely (highest prob). Useful for evaluation.
        :param inp: A single input example of size (..., ..., 1?)
        :return: The accent class that is most likely for the input
        """
        class_probs = self.call(tf.constant([inp]))
        return self.accent_classes[tf.argmax(class_probs)]

    def loss(self, inputs, labels):
        """
        Calculates the loss for some batch of input-label pairs.
        :param inputs: Tensor of size (batchSize, ..., ..., 1?)
        :param labels: Tensor of size (batchSize, 1) - sparse labels indicating index of accent
        :return: The average loss over this batch.
        """
        logits = self.call(inputs)
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))