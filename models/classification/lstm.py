import tensorflow as tf
import tensorflow.keras as k


class ClassifyLSTM(k.Model):
    def __init__(self, accent_classes):
        super(ClassifyLSTM, self).__init__()

        self.accent_classes = accent_classes
        self.num_accent_classes = len(self.accent_classes)

        # TODO: set up layers here

    @tf.function
    def call(self, inputs):
        """
        Run a forward pass of the CNN classification model.
        :param inputs: Tensor or np array of size (batchSize, ..., ..., 1?)
        :return: Tensor of size (batchSize, numAccents)
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

    def loss(self):
        """
        Calculates the loss for some batch of input-label pairs.
        :param inputs: Tensor of size (batchSize, ..., ..., 1?)
        :return: The average loss over this batch.
        """
        ...