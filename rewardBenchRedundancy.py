import json
import logging
import argparse
from pathlib import Path
from os.path import exists
import os
import torch
import numpy as np
from my_longformer import LongformerForSequenceClassification, LongformerForTokenClassification
from transformers import AutoConfig, AutoTokenizer, set_seed, PretrainedConfig, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import pickle as pkl
from scipy.optimize import LinearConstraint, minimize
import random

IGNORE_TAG = "Ignore"
NO_ERROR_TAG = "O"
ERROR_TAG = "ERR"

# MAX_PASSES = 2
MAX_PASSES = 25
EPSILON = 0.01
MAX_EPSILON = 0.01
BIG_NUM = 225
MAX_TRIES = 5


model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"


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
    help=(
        "The maximum total input sequence length after tokenization. If set, sequences longer "
        "than this will be truncated, sequences shorter will be padded."
    ),
    default=2048
)

parser.add_argument(
    "-pad_to_max_length",
    action="store_true",
    help=(
        "Whether to pad all samples to model maximum sentence length. "
        "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
        "efficient on GPU but very bad for TPU."
    )
)

parser.add_argument(
    "-loadFeats",
    action="store_true",
    help="Boolean flag to load features from features.pkl",
)

parser.add_argument(
    "-factuality",
    action="store_true",
    help="Boolean flag to use factuality data (instead of preference data)"
)



parser.add_argument(
    "-addEpsilon",
    action="store_true",
    help="Boolean flag to add epsilon_i for every constraint"
)

parser.add_argument(
    "-cache_dir",
    help="Path to cache location for Huggingface",
    default="/scratch/general/vast/u1472659/huggingface_cache/"
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


def checkIfExists(path, isDir=False, createIfNotExists=False):
    if isDir and not path.endswith("/"):
        raise ValueError("Directory path should end with '/'")
    pathExists = exists(path)
    if not pathExists:
        if createIfNotExists:
            os.makedirs(path)
        else:
            raise ValueError(f"{path} is an invalid path!")
    if not isDir:
        filePath = Path(path)
        if not filePath.is_file():
            raise ValueError(f"{path} is not a file!")


# ----------------------------------------------------------------------
def checkFile(fileName, fileExtension=None):
    if fileExtension:
        if not fileName.endswith(fileExtension):
            raise ValueError(f"[checkFile] {fileName} does not have expected file extension {fileExtension}!")
    file_exists = exists(fileName)
    if not file_exists:
        raise RuntimeError(f"[checkFile] {fileName} is an invalid file path!")
    path = Path(fileName)
    if not path.is_file():
        raise RuntimeError(f"[checkFile] {fileName} is not a file!")


# ----------------------------------------------------------------------
def readFile(filePath):
    data = []
    if filePath.endswith(".json"):
        with open(filePath, "r") as f:
            data = list(json.load(f))
    elif filePath.endswith(".jsonl"):
        with open(filePath, "r") as f:
            lines = f.readlines()
            for line in lines:
                data.append(json.loads(line))
    else:
        raise ValueError(f"[readFile] {filePath} does not have a supported file extension!")
    return data


# ----------------------------------------------------------------------
def writeFile(filePath, data):
    if filePath.endswith(".json"):
        with open(filePath, "w") as f:
            json.dump(data, f)
    elif filePath.endswith(".jsonl"):
        with open(filePath, "w") as f:
            for d in data:
                f.write(json.dumps(d))
                f.write("\n")
    else:
        raise ValueError(f"[readFile] {filePath} does not have a supported file extension!")


# ----------------------------------------------------------------------
def main():
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    if args.logFile:
        #checkFile(args.logFile)
        logging.basicConfig(filename=args.logFile, filemode='w', level=logging.INFO)
    elif args.info:
        logging.basicConfig(filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(filemode='w', level=logging.ERROR)

    if args.start < 0:
        logging.warning("[main] Cannot start from a negative index. Defaulting to zero...")
        args.start = 0

    logging.info("Args: {}".format(args))
    
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)



    features = []
    model_name_short = model_name.split('/')[-1]
    fileEnding = model_name_short

    if args.loadFeats:
        checkFile("features{}.pkl".format(model_name_short))
        with open("features{}.pkl".format(model_name_short), "rb") as f:
            features = pkl.load(f)
            features = features.to(torch.float32)

    else:
        logging.error("did not get a features files; exiting")
        exit()


    non_redundant_inds = np.arange(len(features))
    # if args.shuffle:
    #     logging.info("Shuffling features using seed: {}".format(args.seed))
    #     non_redundant_inds = np.random.choice(np.arange(len(features)), len(features), replace=False)
    #     with open("features_shuffleOrder{}.pkl".format(fileEnding), "wb") as f:
    #         pkl.dump(non_redundant_inds, f)
    redundant_inds = []
    failure_inds = []

    # Support for checkpointing
    if args.start > 0:
        checkIfExists(f"indsToTest{fileEnding}.pkl", isDir=False, createIfNotExists=False)
        checkIfExists(f"redundant{fileEnding}.pkl", isDir=False, createIfNotExists=False)
        checkIfExists(f"non_redundant{fileEnding}.pkl", isDir=False, createIfNotExists=False)
        checkIfExists(f"failure{fileEnding}.pkl", isDir=False, createIfNotExists=False)

        logging.info("Starting from {}".format(args.start))

        with open(f"indsToTest{fileEnding}.pkl", "rb") as f:
            indsToTest = pkl.load(f)

        with open(f"redundant{fileEnding}.pkl", "rb") as f:
            redundant_inds = pkl.load(f)

        with open(f"non_redundant{fileEnding}.pkl", "rb") as f:
            non_redundant_inds = pkl.load(f)

        with open(f"failure{fileEnding}.pkl", "rb") as f:
            failure_inds = pkl.load(f)

    def objFunc(x, *funcArgs):
        if args.addEpsilon:
            return x[:funcArgs[0].shape[0]].dot(funcArgs[0]) + sum(x[funcArgs[0].shape[0]:])
        else:
            return x.dot(funcArgs[0])

    numPass = 0

    results = {}
    while (numPass < MAX_PASSES):
        startInd = args.start if (numPass == 0) else 0

        if startInd == 0:
            indsToTest = non_redundant_inds.copy()

        if numPass == 0 and startInd > len(indsToTest):
            raise ValueError(
                "[main] Cannot start at an index ({}) greater than possible indices ([0,{}])!".format(startInd,
                                                                                                      len(indsToTest)))

        logging.info("Pass {}/{}".format(numPass + 1, MAX_PASSES))
        numPass += 1
        numRedundant = 0
        numFailure = 0
        if args.start and startInd != 0:
            numRedundant = len(redundant_inds)
            numFailure = len(failure_inds)
        else:
            failure_inds = []

        for i, ind in tqdm(enumerate(indsToTest), desc="Checking for redundancy"):
            if i < startInd:
                continue

            # Print debugging information
            if i % 25 == 0:
                logging.info("Stats:\nTested: {}\nRedundant: {}\nNon-redundant: {}\nFailure: {}".format(i, numRedundant,
                                                                                                        i - numRedundant - numFailure,
                                                                                                        numFailure))
                # Support for checkpointing
                with open(f"indsToTest{fileEnding}.pkl", "wb") as f:
                    pkl.dump(indsToTest, f)

                with open(f"redundant{fileEnding}.pkl", "wb") as f:
                    pkl.dump(redundant_inds, f)

                with open(f"non_redundant{fileEnding}.pkl", "wb") as f:
                    pkl.dump(non_redundant_inds, f)

                with open(f"failure{fileEnding}.pkl", "wb") as f:
                    pkl.dump(failure_inds, f)

            numTries = 0
            while 1:
                numTries += 1
                featsToConsider = features[np.delete(non_redundant_inds, np.where(non_redundant_inds == ind))]
                if args.addEpsilon:
                    constraints = torch.zeros(((featsToConsider.shape[0] + featsToConsider.shape[0]),
                                               (featsToConsider[0].shape[0] + featsToConsider.shape[0])))
                    constraints[:featsToConsider.shape[0], :featsToConsider.shape[1]] = featsToConsider
                    # w^T(phi(A) - phi(B)) + eps_i >= 0 <=> w^T(phi(A) - phi(B)) >= -eps_i
                    constraints[:featsToConsider.shape[0], featsToConsider.shape[1]:] = torch.diag(
                        torch.tensor([1] * featsToConsider.shape[0]))
                    # -eps_i + MAX_EPSILON >= 0 <=> eps_i -MAX_EPSILON <=0 <=> eps_i <= MAX_EPSILON
                    constraints[featsToConsider.shape[0]:, featsToConsider.shape[1]:] = torch.diag(
                        torch.tensor([-1] * featsToConsider.shape[0]))
                else:
                    constraints = featsToConsider

                if args.addEpsilon:
                    initialGuess = torch.rand((featsToConsider[0].shape[0] + featsToConsider.shape[0],))
                    initialGuess[featsToConsider[0].shape[0]:] = torch.zeros((featsToConsider.shape[0],))
                else:
                    initialGuess = torch.rand(featsToConsider[0].shape)

                if args.addEpsilon:
                    lowerBound = (-1) * torch.inf
                    upperBound = (0,) * featsToConsider.shape[0] + (MAX_EPSILON,) * featsToConsider.shape[0]
                else:
                    lowerBound = (-1) * torch.inf
                    upperBound = EPSILON

                m = minimize(
                    fun=objFunc,
                    x0=initialGuess,
                    constraints=LinearConstraint(
                        A=(-1) * constraints,
                        lb=lowerBound,
                        ub=upperBound,
                    ),
                    #options=dict(maxiter=50),
                    args=(features[ind].squeeze(),)
                )
                if not np.isclose(m.x[:features[0].shape[0]], torch.zeros(features[ind].shape)).all():
                    break
                else:
                    if numTries > MAX_TRIES:
                        logging.info("Zero solution obtained. Execeeded MAX_TRIES. Exiting...")
                        m.success = False
                        break
                    logging.info("Zero solution obtained. Rerunning...")
            if numTries > MAX_TRIES or not m.success:
                numFailure += 1
                logging.info("Inconsistent constraints @ {}".format(ind))
                failure_inds.append(ind)
                continue
            if m.x[:features[0].shape[0]].dot(features[ind]) >= 0:  # Redundant
                numRedundant += 1
                logging.info("Redundancy @ {}".format(ind))
                redundant_inds.append(ind)
                non_redundant_inds = np.delete(non_redundant_inds, np.where(non_redundant_inds == ind))
            else:  # Non-redundant
                pass
        logging.info(
            "End of Pass {}:\nStats:\nTested: {}\nRedundant: {}\nNon-redundant: {}\nFailure: {}".format(numPass,
                                                                                                        len(indsToTest),
                                                                                                        numRedundant,
                                                                                                        len(indsToTest) - numRedundant - numFailure,
                                                                                                        numFailure))

        results[numPass] = {
            "redundant": redundant_inds.copy(),
            "non_redundant": non_redundant_inds.copy(),
            "failure": failure_inds.copy()
        }

        with open(f"results{fileEnding}.pkl", "wb") as f:
            pkl.dump(results, f)

        if numRedundant == 0:
            logging.info("No new redundant pairs found. Exiting...")
            break

    with open(f"results{fileEnding}.pkl", "wb") as f:
        pkl.dump(results, f)

    with open(f"redundant{fileEnding}.pkl", "wb") as f:
        pkl.dump(redundant_inds, f)

    with open(f"non_redundant{fileEnding}.pkl", "wb") as f:
        pkl.dump(non_redundant_inds, f)

    with open(f"failure{fileEnding}.pkl", "wb") as f:
        pkl.dump(failure_inds, f)

    logging.info(
        "End of all passes:\nStats:\nTested: {}\nRedundant: {}\nNon-redundant: {}\nFailure: {}".format(len(features),
                                                                                                       len(redundant_inds),
                                                                                                       len(non_redundant_inds),
                                                                                                       len(failure_inds)))


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
