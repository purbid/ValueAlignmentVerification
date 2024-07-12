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
from transformers import AutoConfig, AutoTokenizer, set_seed, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from tqdm import tqdm
import pickle as pkl
import random
import pandas as pd

MAX_ITERS = 1000000000

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
    "-features",
    type=str,
    help="Path to pkl file containing feature vectors",
    required=True,
)

parser.add_argument(
    "-indsDir",
    type=str,
    help="Path to folder containing redundant/non-redundant/failure indices",
    required=True,
)

parser.add_argument(
    "-randomSeeds",
    nargs="+",
    help="List of random seeds used to obtain redundant/non-redundant sets",
    required=True,
)

parser.add_argument(
    "-fileAddendum",
    type=str,
    help="Addendum to pkl files containing indices in indsDir",
    default="",
)

parser.add_argument(
    '-seed', 
    type=int, 
    help='Random seed', 
    default=11892
)

parser.add_argument(
    "-cache_dir",
    help="Path to cache location for Huggingface",
    default="/scratch/general/vast/u1419542/huggingface_cache/"
)
#----------------------------------------------------------------------
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
#----------------------------------------------------------------------
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
#----------------------------------------------------------------------
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
    elif filePath.endswith(".pkl"):
        with open(filePath, "rb") as f:
            data = pkl.load(f)
    else: 
        raise ValueError(f"[readFile] {filePath} does not have a supported file extension!")
    return data
#----------------------------------------------------------------------
def writeFile(filePath, data):
    if filePath.endswith(".json"):
        with open(filePath, "w") as f:
            json.dump(data, f)
    elif filePath.endswith(".jsonl"):
        with open(filePath, "w") as f:
            for d in data: 
                f.write(json.dumps(d))
                f.write("\n")
    elif filePath.endswith(".pkl"):
        with open(filePath, "wb") as f:
            pkl.dump(data, f)
    else: 
        raise ValueError(f"[readFile] {filePath} does not have a supported file extension!")
#----------------------------------------------------------------------
def isPointInside(point, constraints):
    return ((np.dot(constraints, point) < 0).sum() <= 0)
#----------------------------------------------------------------------
def main():
    args = parser.parse_args()

    # Set seed before initializing model.
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.logFile:
        checkFile(args.logFile)
        logging.basicConfig(filename=args.logFile, filemode='w', level=logging.INFO)
    elif args.info:
        logging.basicConfig(filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(filemode='w', level=logging.ERROR)

    logging.info("Args: {}".format(args))

    checkIfExists(args.indsDir, isDir=True, createIfNotExists=False)
    for seed in args.randomSeeds:
        checkFile(args.indsDir + "non_redundant" + args.fileAddendum + "_" + str(seed) + ".pkl")

    all_non_redundant_inds = []
    for seed in args.randomSeeds:
        non_redundant_inds = readFile(args.indsDir + "non_redundant" + args.fileAddendum + "_" + str(seed) + ".pkl")
        all_non_redundant_inds.append(non_redundant_inds)
    
    checkFile(args.features, ".pkl")
    features = readFile(args.features)
    
    all_hits = []
    for _ in tqdm(range(MAX_ITERS), desc="Iteration"):
        w = np.random.rand(len(features[0]))
        hits = []
        for non_redundant_inds in tqdm(all_non_redundant_inds, desc="Non-Redundant indices"):
            nonRedundantFeatures = np.array(features)[non_redundant_inds]
            hits.append(int(isPointInside(w, nonRedundantFeatures)))
        all_hits.append(hits)
    
    with open("all_hits.pkl", "wb") as f:
        pkl.dump(all_hits, f)

#----------------------------------------------------------------------
if __name__ == "__main__":
    main()