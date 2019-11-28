from argparse import ArgumentParser, ArgumentTypeError
import os
import tensorflow as tf

from models.classification.cnn import ClassifyCNN
from models.classification.lstm import ClassifyLSTM


def read_args():
    """
    Reads and parses the command line arguments that dictate what to do with the model.
    Includes feature extraction, training, testing, and running.
    :return namespace object containing the parsed arguments
    """

    # Functions for validating that passed arguments have right format
    def valid_directory(directory):
        """
        Function for validating that passed directory exists and is valid.
        :param directory: directory to check for validity
        :return: the directory if valid, ArgumentTypeError if invalid
        """
        if os.path.isdir(directory):
            return directory
        else:
            raise ArgumentTypeError(f"Invalid directory given: {directory}")

    def valid_model_file(model_file):
        """
        Function to check that a model file name is valid (but may not exist)
        :param model_file: the file name to check for validity
        :return: the file name if valid, ArgumentTypeError if invalid
        """
        if True:  # TODO: figure out file ending
            return model_file
        else:
            raise ArgumentTypeError(f"Invalid model file given: {model_file}")

    def existing_model(model_file):
        """
        Function for validating that passed model file name exists and is valid.
        :param model_file: saved model file name to check for validity
        :return: the model file name if valid, ArgumentTypeError if invalid
        """
        if valid_model_file(model_file) and os.path.exists(model_file):
            return model_file
        else:
            raise ArgumentTypeError(f"Invalid or nonexistent model file given: {model_file}")

    def recording_file(rec_file):
        """
        Function to check that recording file is an existing valid .wav file.
        :param rec_file: name of recording to check for validity
        :return: recording file name if valid, ArgumentTypeError if invalid
        """
        if os.path.exists(rec_file) and rec_file.endswith(".wav"):
            return rec_file
        else:
            raise ArgumentTypeError(f"Invalid recording file (must be .wav): {rec_file}")

    parser = ArgumentParser(prog="accenter",
                            description="A deep learning program for classifying "
                                        "and converting accents in speech.")
    subparsers = parser.add_subparsers()

    # Command for process data - takes input raw data directory and output directory
    segment = subparsers.add_parser("segment", description="Segment clips from raw data and "
                                                           "save to a directory")
    segment.add_argument("raw_data_dir", nargs=1, default="data/raw", type=valid_directory)
    segment.add_argument("out_data_dir", nargs=1, default="data/processed", type=valid_directory)
    segment.add_argument("--sil_len", nargs=1, default=1000, type=int)
    segment.add_argument("--sil_thres", nargs=1, default=-62, type=int)

    # Command for feature extracting from npy segments file
    fextr = subparsers.add_parser("fextr", description="Extract features from a segment file")
    fextr.add_argument("processed_dir", nargs=1, default="data/processed", type=valid_directory)

    # Command for training the model - takes in model file and directory with the data
    train = subparsers.add_parser("train", description="Train a model on the given dataset")
    train.add_argument("model_file", nargs=1, type=valid_model_file)
    train.add_argument("data_dir", nargs=1, type=valid_directory)

    # Command for testing the model - takes in model file and directory of test data
    test = subparsers.add_parser("test", description="Evaluate the model on the given data")
    test.add_argument("saved_model", nargs=1, type=existing_model)
    test.add_argument("data_dir", nargs=1, type=valid_directory)

    # Command for running the model - takes in model file, optional output file, and recordings
    run = subparsers.add_parser("run", description="Run the model on the given data")
    run.add_argument("saved_model", nargs=1, type=existing_model)
    run.add_argument("--output-file", "-o", nargs="?", default=None, type=str)
    run.add_argument("recordings", nargs="+", type=recording_file)

    return parser.parse_args()


def process_audio():
    ...


def extract_features():
    ...


def train(model, epochs, train_data_dir, save_file=None):
    """
    Trains the model on the given training data, checkpointing the weights to the given file
    after every epoch.
    :param model: The model to train.
    :param epochs: Number of epochs to train for.
    :param train_data_dir: A directory of the training data to use
    :param save_file: The file to save the model weights to.
    :return: The trained model
    """

    # TODO: parse the train_data directory and get data and labels

    for e in range(epochs):

        for batch_start in range(0, dataset_size, model.batch_size):
            batch_inputs = train_inputs[batch_start:batch_start + model.batch_size]
            batch_labels = train_labels[batch_start:batch_start + model.batch_size]

            with tf.GradientTape() as tape:
                loss = model.loss(batch_inputs, batch_labels)

            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Save the model at the end of the epoch
        if save_file:
            model.save_weights(save_file, save_format="h5")


def test():
    ...


def evaluate(model, input_audio):
    if model.type == "classifier":
        # TODO: preprocess and feature extract the input audio
        model.get_class(...)
    elif model.type == "converter":
        ...
    else:
        print("Model type not recognized")


if __name__ == "__main__":
    args = read_args()

    accent_classes = ["british", "chinese", "american", "korean"]
    model = ClassifyCNN(accent_classes)

    # TODO: implement loading, etc.
