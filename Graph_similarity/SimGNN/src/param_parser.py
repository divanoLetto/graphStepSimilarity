"""Getting params from the command line."""

import argparse


def parameter_parser_base(model_save_path, model_load_path, epochs):
    """
        A method to parse up command line parameters.
        The default hyperparameters give a high performance model without grid search.
        """
    parser = argparse.ArgumentParser(description="Run SimGNN.")

    parser.add_argument("--training-graphs",
                        nargs="?",
                        default="./dataset/train/",
                        help="Folder with training graph pair jsons.")

    parser.add_argument("--testing-graphs",
                        nargs="?",
                        default="./dataset/test/",
                        help="Folder with testing graph pair jsons.")

    parser.add_argument("--epochs",
                        type=int,
                        default=epochs,
                        help="Number of training epochs. Default is 5.")

    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
                        help="Neurons in tensor network layer. Default is 16.")

    parser.add_argument("--bottle-neck-neurons",
                        type=int,
                        default=16,
                        help="Bottle neck layer neurons. Default is 16.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=256,
                        help="Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--bins",
                        type=int,
                        default=16,
                        help="Similarity score bins. Default is 16.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="Dropout probability. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5 * 10 ** -4,
                        help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--histogram",
                        dest="histogram",
                        action="store_true")

    parser.set_defaults(histogram=True)

    parser.add_argument("--save-path",
                        type=str,
                        default=model_save_path,
                        help="Model saves path")

    parser.add_argument("--load-path",
                        type=str,
                        default=model_load_path,
                        help="Load a pretrained model")

    return parser


def parameter_parser_256(model_save_path, model_load_path, epochs):
    parser = parameter_parser_base(model_save_path, model_load_path, epochs)
    parser.add_argument("--filters-1",
                        type=int,
                        default=256,
                        help="Filters (neurons) in 1st convolution. Default is 1024.")

    parser.add_argument("--filters-2",
                        type=int,
                        default=256,
                        help="Filters (neurons) in 2nd convolution. Default is 512.")

    parser.add_argument("--filters-3",
                        type=int,
                        default=256,
                        help="Filters (neurons) in 3rd convolution. Default is 256.")

    parser.add_argument("--filters-4",
                        type=int,
                        default=128,
                        help="Filters (neurons) in 3rd convolution. Default is 128.")

    parser.add_argument("--filters-5",
                        type=int,
                        default=64,
                        help="Filters (neurons) in 3rd convolution. Default is 64.")

    parser.add_argument("--filters-6",
                        type=int,
                        default=32,
                        help="Filters (neurons) in 3rd convolution. Default is 32.")
    return parser.parse_args()


def parameter_parser_512(model_save_path, model_load_path, epochs):
    parser = parameter_parser_base(model_save_path, model_load_path, epochs)
    parser.add_argument("--filters-1",
                        type=int,
                        default=512,
                        help="Filters (neurons) in 1st convolution. Default is 1024.")

    parser.add_argument("--filters-2",
                        type=int,
                        default=512,
                        help="Filters (neurons) in 2nd convolution. Default is 512.")

    parser.add_argument("--filters-3",
                        type=int,
                        default=256,
                        help="Filters (neurons) in 3rd convolution. Default is 256.")

    parser.add_argument("--filters-4",
                        type=int,
                        default=128,
                        help="Filters (neurons) in 3rd convolution. Default is 128.")

    parser.add_argument("--filters-5",
                        type=int,
                        default=64,
                        help="Filters (neurons) in 3rd convolution. Default is 64.")

    parser.add_argument("--filters-6",
                        type=int,
                        default=32,
                        help="Filters (neurons) in 3rd convolution. Default is 32.")
    return parser.parse_args()


def parameter_parser_1024(model_save_path, model_load_path, epochs):
    parser = parameter_parser_base(model_save_path, model_load_path, epochs)
    parser.add_argument("--filters-1",
                        type=int,
                        default=1024,
                        help="Filters (neurons) in 1st convolution. Default is 1024.")

    parser.add_argument("--filters-2",
                        type=int,
                        default=512,
                        help="Filters (neurons) in 2nd convolution. Default is 512.")

    parser.add_argument("--filters-3",
                        type=int,
                        default=256,
                        help="Filters (neurons) in 3rd convolution. Default is 256.")

    parser.add_argument("--filters-4",
                        type=int,
                        default=128,
                        help="Filters (neurons) in 3rd convolution. Default is 128.")

    parser.add_argument("--filters-5",
                        type=int,
                        default=64,
                        help="Filters (neurons) in 3rd convolution. Default is 64.")

    parser.add_argument("--filters-6",
                        type=int,
                        default=32,
                        help="Filters (neurons) in 3rd convolution. Default is 32.")
    return parser.parse_args()


def parameter_parser_slim_256(model_save_path, model_load_path, epochs):
    parser = parameter_parser_base(model_save_path, model_load_path, epochs)
    parser.add_argument("--filters-3",
                        type=int,
                        default=256,
                        help="Filters (neurons) in 3rd convolution. Default is 256.")

    parser.add_argument("--filters-4",
                        type=int,
                        default=128,
                        help="Filters (neurons) in 3rd convolution. Default is 128.")

    parser.add_argument("--filters-5",
                        type=int,
                        default=64,
                        help="Filters (neurons) in 3rd convolution. Default is 64.")

    parser.add_argument("--filters-6",
                        type=int,
                        default=32,
                        help="Filters (neurons) in 3rd convolution. Default is 32.")
    return parser.parse_args()

