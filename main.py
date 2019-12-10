from argparse import ArgumentParser, ArgumentTypeError
import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


from dataUtil import ioUtil as Io
from dataUtil import featureExtraction as fExtr
from dataUtil import processing
from models.classification.cnn import ClassifyCNN, ClassifyGCNN
from models.classification.lstm import ClassifyLSTM


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
gpu_available = tf.test.is_gpu_available()


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


def get_data_from_dir(data_dir, preprocess_method, subset):
    """
    Loads the data from a given data directory, giving back both inputs and labels
    :param data_dir: The directory to load data from
    :param preprocess_method: The method to use for preprocessing (mfcc | spectrogram)
    :param subset: The subset of the data to get (train | test)
    :return: The inputs and labels from the data directory as tensors
    """

    def normalize_tensor(tensor):
        """
        Normalizes a tensor to the 0-1 range and casts it to floats
        :param tensor: A tensor of any shape
        :return: The same tensor, scaled such that all values are between 0 and 1
        """
        tensor = tf.cast(tensor, tf.float32)
        tens_stddev = tf.math.reduce_std(tensor)
        tens_mean = tf.reduce_mean(tensor)
        return tf.divide(tf.subtract(tensor, tens_mean), tens_stddev)

    inputs = []
    labels = []

    accent_class_folders = [folder for folder in os.listdir(data_dir)
                            if os.path.isdir(os.path.join(data_dir, folder))]
    for folder in accent_class_folders:
        assert folder in model.accent_classes

        data_file = os.path.join(data_dir, folder, f"{folder}-{preprocess_method}-{subset}.npy")
        class_data = Io.read_audio_data(data_file)
        class_labels = np.full((class_data.shape[0]), np.where(model.accent_classes == folder)[0])

        inputs.append(normalize_tensor(tf.expand_dims(class_data, -1)))
        labels.append(tf.convert_to_tensor(class_labels))

    return inputs, labels


def augment_random_noise(inputs):
    """
    Adds random noise from N(0,1) to the inputs (already normalized) to augment data.
    :param inputs: The input batch.
    :return: The input batch augmented with noise.
    """
    return tf.add(inputs, tf.random.normal(inputs.shape, mean=0.0, stddev=0.1))


def batch_generator(inputs, labels, batch_size):
    """
    Gets batches with equal numbers of examples from each accent class.
    :param inputs: All of the inputs
    :param labels: All of the labels
    :param batch_size: Batch size to return
    :return:
    """
    assert len(inputs) == len(labels)

    dataset_size = sum([inp.shape[0] for inp in inputs])
    num_batches = dataset_size // batch_size
    num_classes = len(inputs)

    for b in range(num_batches):
        # Get an equal number of samples from each accent class
        batch_inputs = None
        batch_labels = None
        for ac in range(num_classes):
            indeces = tf.random.shuffle(tf.range(len(inputs[ac])))[:batch_size // num_classes]

            if batch_inputs is not None and batch_labels is not None:
                batch_inputs = tf.concat((batch_inputs, tf.gather(inputs[ac], indeces)), 0)
                batch_labels = tf.concat((batch_labels, tf.gather(labels[ac], indeces)), 0)
            else:
                batch_inputs = tf.gather(inputs[ac], indeces)
                batch_labels = tf.gather(labels[ac], indeces)

        yield tf.convert_to_tensor(batch_inputs), tf.convert_to_tensor(batch_labels)


def augment_random_noise(inputs):
    """
    Adds random noise from N(0,1) to the inputs (already normalized) to augment data.
    :param inputs: The input batch.
    :return: The input batch augmented with noise.
    """
    return tf.add(inputs, tf.random.normal(inputs.shape, mean=0.0, stddev=1.0))


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

    c_train_inputs, c_train_labels = get_data_from_dir(train_data_dir, preprocess_method, "train")

    assert c_train_inputs is not None
    assert c_train_labels is not None
    assert len(c_train_inputs) > 0 and len(c_train_labels) > 0

    train_inputs = c_train_inputs[0]
    train_labels = c_train_labels[0]
    for c in range(1, len(c_train_inputs)):
        train_inputs = tf.concat([train_inputs, c_train_inputs[c]], 0)
        train_labels = tf.concat([train_labels, c_train_labels[c]], 0)

    dataset_size = sum([len(inp) for inp in c_train_inputs])
    print(f"Training on {dataset_size} examples")

    train_accs = []
    test_accs = []

    epoch_loss = model.loss(train_inputs, train_labels)
    epoch_acc = model.accuracy(train_inputs, train_labels)
    test_acc = test(model, train_data_dir, preprocess_method=preprocess_method)
    train_accs.append(epoch_acc)
    test_accs.append(test_acc)
    print(f"Baseline | Loss: {epoch_loss:.3f} | Train acc: {epoch_acc:.3f} | "
          f"Test Acc: {test_acc:.3f}")
    
    for e in range(epochs):
        # Shuffle the dataset before each epoch
        new_order = tf.random.shuffle(tf.range(dataset_size))
        train_inputs = tf.gather(train_inputs, new_order)
        train_labels = tf.gather(train_labels, new_order)

        # Run training in batches
        for batch_inputs, batch_labels in \
                batch_generator(c_train_inputs, c_train_labels, model.batch_size):

            with tf.GradientTape() as tape:
                loss = model.loss(batch_inputs, batch_labels)

            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Print loss and accuracy
        epoch_loss = model.loss(train_inputs, train_labels)
        epoch_acc = model.accuracy(train_inputs, train_labels)

        test_acc = test(model, train_data_dir, preprocess_method=preprocess_method)
        train_accs.append(epoch_acc)
        test_accs.append(test_acc)
        print(f"Epoch {e + 1}/{epochs} | Loss: {epoch_loss:.3f} | Accuracy: {epoch_acc:.3f} | "
              f"Test acc: {test_acc:.3f}")

        # Save the model at the end of the epoch
        if save_file:
            model.save_weights(save_file, save_format="h5")

    plot_feature(train_accs, test_accs, preprocess_method,
                 "Accuracy_CNN_MFCC", epochs, save=True)

    Io.export_audio_data("acc_train.npy", np.asarray(train_accs))
    Io.export_audio_data("acc_test.npy", np.asarray(test_accs))


def plot_feature(train_data, test_data, preprocess_method, title, epochs,
                 save=False):
    plt.plot(range(0, epochs + 1), train_data, label='train')
    plt.plot(range(0, epochs + 1), test_data, label='test')

    plt.xlabel("n epoch")
    plt.legend(loc='upper right')
    plt.title(title)
    if save:
        plt.savefig(f"acc_{preprocess_method}_{model.__class__.__name__}.png")
    plt.show()


def test(model, test_data_dir, preprocess_method="mfcc"):
    """
    Runs the model on the test dataset and reports the accuracy on it.
    :param model: The model to run the test dataset on.
    :param test_data_dir: The data directory of the test data.
    :param preprocess_method: The method to use for preprocessing (mfcc | spectrogram)
    :return: The accuracy on the test dataset.
    """
    inputs, labels = get_data_from_dir(test_data_dir, preprocess_method, "test")
    assert len(inputs) == len(labels)

    num_correct = []
    total_examples = 0
    for c in range(len(inputs)):
        c_inputs = inputs[c]
        c_labels = labels[c]
        num_correct.append(model.accuracy(c_inputs, c_labels) * c_inputs.shape[0])
        total_examples += c_inputs.shape[0]

    return sum(num_correct) / total_examples


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


def init_model(problem_type, model_type, accent_classes, preprocess_method="mfcc"):
    """
    Initializes a model and gets it ready for weight loading.
    :param problem_type: The problem the model solves (classify | convert)
    :param model_type: The type of model being used (cnn | lstm)
    :param accent_classes: List of accent classes that the model will deal with.
    :param preprocess_method: The preprocessing method to initialize with (mfcc | spectrogram)
    :return: The initialized model
    """

    if problem_type == "classify":
        if model_type == "cnn":
            model = ClassifyGCNN(accent_classes)
        elif model_type == "lstm":
            model = ClassifyLSTM(accent_classes)
        else:
            raise Exception("Invalid model type for classify. Must be either cnn or lstm")

        if preprocess_method == "mfcc":
            input_shape = (model.batch_size, 44, 49, 1)
        elif preprocess_method == "spectrogram":
            input_shape = (model.batch_size, 98, 70, 1)
        else:
            raise Exception("Invalid preprocessing method. Must be either mfcc or spectrogram")

        model.call(tf.random.uniform(input_shape))

    elif problem_type == "convert":
        raise Exception("Conversion model not implemented")
    else:
        raise Exception("Invalid problem type. Must be either classify or convert")

    return model


if __name__ == "__main__":
    args = read_args()
    
    accent_classes = ["american", "chinese", "korean"]
    preprocess_method = "mfcc"
    model = init_model("classify", "cnn", accent_classes, preprocess_method=preprocess_method)


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
        print(f"GPU available: {gpu_available}")
        # Load the saved model or create directory if doesn't exist
        if os.path.exists(args.model_file):
            model.load_weights(args.model_file)
        elif not os.path.exists(os.path.dirname(args.model_file)):
            os.makedirs(os.path.dirname(args.model_file))

        # Train on gpu if it's available
        with tf.device("/device:" + ("GPU:0" if gpu_available else "CPU:0")):
            train(model, args.epochs, args.data_dir,
                  save_file=args.model_file, preprocess_method=preprocess_method)

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
