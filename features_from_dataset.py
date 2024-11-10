import json
import argparse
import logging
import argparse
from pathlib import Path
from os.path import exists
import os
import torch
from typing import Any, Dict, List, Union
import numpy as np
from my_longformer import LongformerForSequenceClassification, LongformerForTokenClassification
from transformers import AutoConfig, AutoTokenizer, set_seed, PretrainedConfig, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import pickle as pkl
from scipy.optimize import LinearConstraint, minimize
import random

from featurize_args import get_args


IGNORE_TAG = "Ignore"
NO_ERROR_TAG = "O"
ERROR_TAG = "ERR"

# MAX_PASSES = 2
MAX_PASSES = 50
EPSILON = 0.01
# EPSILON = 0.000001
# EPSILON = 0
MAX_EPSILON = 0.01
BIG_NUM = 225
MAX_TRIES = 5



if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

