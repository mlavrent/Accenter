from argparse import ArgumentParser, ArgumentTypeError
import os
import tensorflow as tf
import numpy as np

from dataUtil import ioUtil as Io
from dataUtil import featureExtraction as fExtr
from dataUtil import processing
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
            return os.path.normpath(directory)
        else:
            raise ArgumentTypeError(f"Invalid directory given: {directory}")

    def valid_model_file(model_file):
        """
        Function to check that a model file name is valid (but may not exist)
        :param model_file: the file name to check for validity
        :return: the file name if valid, ArgumentTypeError if invalid
        """
        if True:  # TODO: figure out file ending
            return os.path.normpath(model_file)
        else:
            raise ArgumentTypeError(f"Invalid model file given: {model_file}")

    def existing_model(model_file):
        """
        Function for validating that passed model file name exists and is valid.
        :param model_file: saved model file name to check for validity
        :return: the model file name if valid, ArgumentTypeError if invalid
        """
        if valid_model_file(model_file) and os.path.exists(model_file):
            return os.path.normpath(model_file)
        else:
            raise ArgumentTypeError(f"Invalid or nonexistent model file given: {model_file}")

    def recording_file(rec_file):
        """
        Function to check that recording file is an existing valid .wav file.
        :param rec_file: name of recording to check for validity
        :return: recording file name if valid, ArgumentTypeError if invalid
        """
        if os.path.exists(rec_file) and rec_file.endswith(".wav"):
            return os.path.normpath(rec_file)
        else:
            raise ArgumentTypeError(f"Invalid recording file (must be .wav): {rec_file}")

    parser = ArgumentParser(prog="accenter",
                            description="A deep learning program for classifying "
                                        "and converting accents in speech.")
    subparsers = parser.add_subparsers()

    # Command for process data - takes input raw data directory and output directory
    segment = subparsers.add_parser("segment", description="Segment clips from raw data and "
                                                           "save to a directory")
    segment.set_defaults(command="segment")
    segment.add_argument("raw_data_dir", default="data/raw", type=valid_directory)
    segment.add_argument("out_data_dir", default="data/processed", type=valid_directory)
    segment.add_argument("--sil_len", default=1000, type=int)
    segment.add_argument("--sil_thresh", default=-62, type=int)

    # Command for feature extracting from npy segments file
    fextr = subparsers.add_parser("fextr", description="Extract features from a segment file")
    fextr.set_defaults(command="fextr")
    fextr.add_argument("processed_dir", default="data/processed", type=valid_directory)

    # Command for training the model - takes in model file and directory with the data
    train = subparsers.add_parser("train", description="Train a model on the given dataset")
    train.set_defaults(command="train")
    train.add_argument("epochs", type=int)
    train.add_argument("model_file", type=valid_model_file)
    train.add_argument("data_dir", type=valid_directory)

    # Command for testing the model - takes in model file and directory of test data
    test = subparsers.add_parser("test", description="Evaluate the model on the given data")
    test.set_defaults(command="test")
    test.add_argument("saved_model", type=existing_model)
    test.add_argument("data_dir", type=valid_directory)

    # Command for running the model - takes in model file, optional output file, and recordings
    run = subparsers.add_parser("run", description="Run the model on the given data")
    run.set_defaults(command="run")
    run.add_argument("saved_model", type=existing_model)
    run.add_argument("--output-file", "-o", nargs="?", default=None, type=str)
    run.add_argument("recording", nargs="+", type=recording_file)

    return parser.parse_args()


def get_data_from_dir(data_dir, preprocess_method):
    """
    Loads the data from a given data directory, giving back both inputs and labels
    :param data_dir: The directory to load data from
    :param preprocess_method: The method to use for preprocessing (mfcc | spectrogram)
    :return: The inputs and labels from the data directory
    """
    inputs = None
    labels = None

    accent_class_folders = [folder for folder in os.listdir(data_dir)
                            if os.path.isdir(os.path.join(data_dir, folder))]
    for folder in accent_class_folders:
        data_file = os.path.join(data_dir, folder, f"{folder}-{preprocess_method}.npy")
        class_data = Io.read_audio_data(data_file)
        class_labels = np.full((class_data.shape[0]), model.accent_classes.index(folder))

        if inputs and labels:
            inputs = np.concatenate(inputs, class_data)
            labels = np.concatenate(labels, class_labels)
        else:
            inputs = class_data
            labels = class_labels

    return inputs, labels


def train(model, epochs, train_data_dir, save_file=None, preprocess_method="mfcc"):
    """
    Trains the model on the given training data, checkpointing the weights to the given file
    after every epoch.
    :param model: The model to train.
    :param epochs: Number of epochs to train for.
    :param train_data_dir: A directory of the training data to use
    :param save_file: The file to save the model weights to.
    :param preprocess_method: The preprocess method to use for the inputs to the model (mfcc | spectrogram)
    :return: The trained model
    """

    train_inputs, train_labels = get_data_from_dir(train_data_dir, preprocess_method)

    assert train_inputs is not None
    assert train_labels is not None
    assert train_inputs.shape[0] == train_labels.shape[0]
    dataset_size = train_labels.shape[0]

    for e in range(epochs):

        # Shuffle the dataset before each epoch
        new_order = np.random.permutation(dataset_size)
        train_inputs = train_inputs[new_order]
        train_labels = train_labels[new_order]

        # Run training in batches
        for batch_start in range(0, dataset_size, model.batch_size):
            batch_inputs = train_inputs[batch_start:batch_start + model.batch_size]
            batch_labels = train_labels[batch_start:batch_start + model.batch_size]

            with tf.GradientTape() as tape:
                loss = model.loss(batch_inputs, batch_labels)

            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Print loss and accuracy
        epoch_loss = model.loss(train_inputs, train_labels)
        epoch_acc = model.accuracy(train_inputs, train_labels)
        print(f"Epoch {e}/{epochs} | Loss: {epoch_loss} | Accuracy: {epoch_acc}")

        # Save the model at the end of the epoch
        if save_file:
            model.save_weights(save_file, save_format="h5")


def test(model, test_data_dir, preprocess_method="mfcc"):
    """
    Runs the model on the test dataset and reports the accuracy on it.
    :param model: The model to run the test dataset on.
    :param test_data_dir: The data directory of the test data.
    :param preprocess_method: The method to use for preprocessing (mfcc | spectrogram)
    :return: The accuracy on the test dataset.
    """
    test_inputs, test_labels = get_data_from_dir(test_data_dir, preprocess_method)
    return model.accuracy(test_inputs, test_labels)


def classify_accent(model, input_audio, preprocess_method="mfcc"):
    """
    Gets the predicted class for a given audio segment.
    :param model: The model to use to predict the class.
    :param input_audio: The audio to predict the accent for.
    :param preprocess_method: The method to use for preprocessing (mfcc | spectrogram)
    :return: The predicted accent for the given audio.
    """
    if model.type == "classifier":
        spectrogram, mfccs = fExtr.segment_and_extract(input_audio)
        if preprocess_method == "mfcc":
            input_data = mfccs
        elif preprocess_method == "spectrogram":
            input_data = spectrogram
        else:
            raise Exception("Invalid preprocessing method specified")
        return model.get_class([input_data])
    elif model.type == "converter":
        raise Exception("Model not implemented")
    else:
        print("Model type not recognized")


if __name__ == "__main__":
    args = read_args()

    accent_classes = ["british", "chinese", "american", "korean"]
    model = ClassifyCNN(accent_classes)

    if args.command == "segment":
        # Create output directories if they don't exist
        for accent in accent_classes:
            out_path = os.path.join(args.out_data_dir, accent, "clips")
            if not os.path.exists(out_path):
                os.makedirs(out_path)

        # Segment the data
        processing.process_audio_directory(args.raw_data_dir, args.out_data_dir)

    elif args.command == "fextr":
        fExtr.extract_audio_directory(args.processed_dir, testing=False)

    elif args.command == "train":
        # Load the saved model or create directory if doesn't exist
        if os.path.exists(args.model_file):
            model.load_weights(args.model_file)
        else:
            os.makedirs(os.path.dirname(args.model_file))

        train(model, args.epochs, args.data_dir,
              save_file=args.model_file, preprocess_method="mfcc")

    elif args.command == "test":
        print(f"Testing {args.data_dir}")
        model.load_weights(args.saved_model)
        accuracy = test(model, args.data_dir)
        print(f"Accuracy: {accuracy*100:.1f}%")

    elif args.command == "run":
        print(f"Evaluating {args.recording}")
        model.load_weights(args.saved_model)
        accent = classify_accent(model, args.recording)
        print(f"Predicted class: {accent}")

    else:
        print("No command entered.")
