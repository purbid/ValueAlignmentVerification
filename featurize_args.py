import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-info",
        action="store_true",
        help="Boolean flag to enable info mode"
    )

    parser.add_argument(
        "-log",
        "--logFile",
        type=str,
        help="Path to file to print logging information",
        default=None
    )

    parser.add_argument(
        "-data",
        type=str,
        help="Path to file containing data"
    )

    parser.add_argument(
        "-model",
        type=str,
        help="Path to Longformer model on huggingface",
        default="allenai/longformer-base-4096",
    )

    parser.add_argument(
        '-seed',
        type=int,
        help='Random seed',
        default=11892
    )

    parser.add_argument(
        "-lm_output_column_prefix",
        type=str,
        help="The column name of text to input in the file (a csv or JSON file).",
        default="prediction"
    )

    parser.add_argument(
        "-preference_column",
        type=str,
        help="The column name of text to input in the file (a csv or JSON file).",
        default="preference"
    )

    parser.add_argument(
        "-n_lm_outputs",
        type=int,
        help="The number of LM outputs being generated for each input, included in the json files.",
        default=4
    )

    parser.add_argument(
        "-max_seq_length",
        type=int,
        help="The maximum total input sequence length after tokenization.",
        default=2048
    )

    parser.add_argument(
        "-pad_to_max_length",
        action="store_true",
        help="Whether to pad all samples to model maximum sentence length."
    )

    parser.add_argument(
        "-loadFeats",
        action="store_true",
        help="Boolean flag to load features from features.pkl",
    )

    parser.add_argument(
        "-factuality",
        action="store_true",
        help="Boolean flag to use factuality data"
    )

    parser.add_argument(
        "-grammaticality",
        action="store_true",
        help="Boolean flag to use grammaticality data"
    )

    parser.add_argument(
        "-fluency",
        action="store_true",
        help="Boolean flag to use fluency data"
    )

    parser.add_argument(
        "-addEpsilon",
        action="store_true",
        help="Boolean flag to add epsilon_i for every constraint"
    )

    parser.add_argument(
        "-cache_dir",
        help="Path to cache location for Huggingface",
        default="/scratch/general/vast/u1419542/huggingface_cache/"
    )

    parser.add_argument(
        "-start",
        type=int,
        help="Index to start from when checking for redundancies",
        default=0
    )

    parser.add_argument(
        "-shuffle",
        action="store_true",
        help="Boolean flag to shuffle features set",
    )

    return parser.parse_args()
