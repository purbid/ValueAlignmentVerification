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

BASE_MODEL="microsoft/deberta-v3-large"

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
    "-grammaticality",
    action="store_true",
    help="Boolean flag to use grammaticality data (instead of preference/factuality/fluency data)"
)

parser.add_argument(
    "-fluency",
    action="store_true",
    help="Boolean flag to use fluency data (instead of preference/factuality/grammaticality data)"
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
#----------------------------------------------------------------------
class ClassificationHead(torch.nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, num_labels):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states, return_last_hidden_state=False):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        if return_last_hidden_state:
            return output, hidden_states
        return output
#----------------------------------------------------------------------
class PreferenceModel(torch.nn.Module):
    def __init__(self, modelPath="microsoft/deberta-v3-large", cache_dir=" ~/.cache/huggingface/", device="cpu"):
        super(PreferenceModel, self).__init__()
        self.modelPath = modelPath 
        self.model = AutoModel.from_pretrained(self.modelPath, cache_dir=cache_dir)
        self.classifier= ClassificationHead(
            hidden_size=self.model.config.hidden_size,
            hidden_dropout_prob=0.1,
            num_labels=1,
        )
        self.device=device

    def forward(self, inputs, return_last_hidden_state=False):
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
        outputs = self.model(**inputs)
        return self.classifier(outputs["last_hidden_state"], return_last_hidden_state)

    def to(self, device):
        self.device = device 
        self = super().to(device)
        return self 
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
    else: 
        raise ValueError(f"[readFile] {filePath} does not have a supported file extension!")
#----------------------------------------------------------------------
def main():
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if (args.factuality and args.grammaticality) or (args.factuality and args.fluency) or (args.fluency and args.grammaticality):
        raise ValueError("Cannot use factuality/grammaticality/fluency data simultaneously!")

    if args.logFile:
        checkFile(args.logFile)
        logging.basicConfig(filename=args.logFile, filemode='w', level=logging.INFO)
    elif args.info:
        logging.basicConfig(filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(filemode='w', level=logging.ERROR)

    if args.start < 0:
        logging.warning("[main] Cannot start from a negative index. Defaulting to zero...")
        args.start = 0

    logging.info("Args: {}".format(args))

    checkFile(args.data)

    # Set seed before initializing model.
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    fileAddendum = [""]
    if args.addEpsilon: 
        fileAddendum.append("epsilon")
    if args.grammaticality:
        fileAddendum.append("grammaticality")
    elif args.fluency:
        fileAddendum.append("fluency")
    elif args.factuality:
        fileAddendum.append("factuality")
    if args.shuffle:
        fileAddendum.append(str(args.seed))

    fileEnding = "_".join(fileAddendum)
    #Do not include seed in features file name
    featuresFileEnding = "_".join(fileAddendum[:-1])

    features = []
    if args.loadFeats:
        checkFile("features{}.pkl".format(featuresFileEnding))
        with open("features{}.pkl".format(featuresFileEnding),"rb") as f:
            features = pkl.load(f) 
    else:
        # Read the datasets
        data_files = {}
        data_files["train"] = args.data
        extension = args.data.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        if args.grammaticality or args.fluency:
            model = torch.load(args.model)
            model = model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=args.cache_dir)
        else:
            config = AutoConfig.from_pretrained(
                args.model,
                # num_labels=num_labels,
            )

            if args.factuality:
                model = LongformerForTokenClassification.from_pretrained(
                    args.model,
                    from_tf=bool(".ckpt" in args.model),
                    config=config,
                )
                model = model.to(device)
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model,
                    use_fast=True,
                    add_prefix_space=True,
                )
            else: 
                model = LongformerForSequenceClassification.from_pretrained(
                    args.model,
                    from_tf=bool(".ckpt" in args.model),
                    config=config,
                )
                model = model.to(device)
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model,
                    use_fast=True,
                )

        # Preprocessing the dataset
        # Padding strategy
        padding = "max_length" if args.pad_to_max_length else False
        #----------------------------------------------------------------------
        def process_annotations(annotations):
            examples = []

            prediction_keys = [f"{args.lm_output_column_prefix} {i+1}" for i in range(args.n_lm_outputs)]
            n_comparisons = (args.n_lm_outputs * (args.n_lm_outputs - 1)) // 2

            for ann in tqdm(annotations, desc="Processing annotations"):    
                predictions = [ann[key] for key in prediction_keys]

                # construct prompt
                tokens = ["question:"] + ann["question"].strip().split()
                for i, p in enumerate(ann['passages']):
                    if i == 0:
                        tokens += ["context:"]
                    title_string = f"wikipage: {p[0]}"
                    p_tokens = title_string.split()
                    p_tokens += ['text:'] + ' '.join(p[1:]).split()
                    tokens += p_tokens
                tokens += ['answer:']
                prompt = ' '.join(tokens)

                preferences = ann[args.preference_column]

                assert len(preferences) == n_comparisons

                pair_id = 0

                for i in range(len(predictions)):
                    for j in range(i+1, len(predictions)):
                        pref = preferences[pair_id] # 0 equal, 1 first better, 2 second better
                        pair_id += 1
                        example = {}
                        if pref == 0:
                            continue
                        elif pref == 1:
                            example["pred1"] = tokenizer.encode(
                                prompt + ' ' + predictions[i], 
                                padding=padding, 
                                max_length=args.max_seq_length,
                                truncation=True,
                                return_tensors="pt"
                            )
                            example["pred2"] = tokenizer.encode(
                                prompt + ' ' + predictions[j],
                                padding=padding, 
                                max_length=args.max_seq_length,
                                truncation=True,
                                return_tensors="pt"
                            )
                            example["label"] = 1
                        elif pref == 2:
                            example["pred1"] = tokenizer.encode(
                                prompt + ' ' + predictions[j],
                                padding=padding, 
                                max_length=args.max_seq_length,
                                truncation=True,
                                return_tensors="pt"
                            )
                            example["pred2"] = tokenizer.encode(
                                prompt + ' ' + predictions[i],
                                padding=padding, 
                                max_length=args.max_seq_length,
                                truncation=True,
                                return_tensors="pt"
                            )
                            example["label"] = 1
                        else:
                            raise("unknown preference")

                        examples.append(example)

            return examples
        #----------------------------------------------------------------------
        def tokenize_and_align_labels(examples, input_column="text"):
            proInp = {
                "text": {},
                "corrected_pred": {}
            }
            tokenized_inputs = tokenizer(
                examples[input_column],
                padding=padding,
                truncation=True,
                max_length=args.max_seq_length,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )

            tokenized_inputs_corrected_pred = tokenizer(
                examples[input_column+"_corrected_pred"],
                padding=padding,
                truncation=True,
                max_length=args.max_seq_length,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )

            proInp["text"] = tokenized_inputs
            proInp["corrected_pred"] = tokenized_inputs_corrected_pred

            return proInp
        #----------------------------------------------------------------------
        #Owishiboo/grammar-correction
        def tokenize_grammaticality(example):
            instance = {}
            instance["preferred"] = tokenizer.encode_plus(
                example["target"],
                padding="max_length",
                truncation=True,
                max_length=args.max_seq_length,
                return_tensors="pt"
            )

            instance["dispreferred"] = tokenizer.encode_plus(
                example["input"],
                padding="max_length",
                truncation=True,
                max_length=args.max_seq_length,
                return_tensors="pt"
            )

            return instance
        #----------------------------------------------------------------------
        #JFLEG Dataset
        def tokenize_fluency(example):
            instance = {}
            instance["preferred"] = tokenizer.encode_plus(
                example["corrections"][0],
                padding="max_length",
                truncation=True,
                max_length=args.max_seq_length,
                return_tensors="pt"
            )

            instance["dispreferred"] = tokenizer.encode_plus(
                example["sentence"],
                padding="max_length",
                truncation=True,
                max_length=args.max_seq_length,
                return_tensors="pt"
            )

            return instance
        #----------------------------------------------------------------------
    
        if args.grammaticality:
            train_dataset = raw_datasets["train"].map(
                tokenize_grammaticality,
                batched=False,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        elif args.fluency:
            train_dataset = raw_datasets["train"].map(
                tokenize_fluency,
                batched=False,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        elif args.factuality:
            train_dataset = raw_datasets["train"].map(
                tokenize_and_align_labels,
                batched=False,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        else:
            train_dataset = process_annotations(raw_datasets["train"])

        with open("train_dataset{}.pkl".format(fileEnding), "wb") as f:
            pkl.dump(train_dataset, f)

        model.eval()
        with torch.no_grad():
            if args.grammaticality or args.fluency:
                for d in tqdm(train_dataset, desc="Featurizing dataset"):
                    d["preferred"] = {k:torch.tensor(v) for (k,v) in d["preferred"].items()}
                    d["dispreferred"] = {k:torch.tensor(v) for (k,v) in d["dispreferred"].items()}
                    _, feat_1 = model(
                        d["preferred"],
                        return_last_hidden_state=True,
                    )
                    _, feat_2 = model(
                        d["dispreferred"],
                        return_last_hidden_state=True,
                    )
                    f1 = feat_1.cpu().squeeze()
                    f2 = feat_2.cpu().squeeze()
                    features.append((f1-f2).cpu())
                    logging.info("Shape: {}".format(features[-1].shape))
            elif args.factuality:
                for d in tqdm(train_dataset, desc="Featurizing dataset"):
                    feat_1 = model(
                        input_ids=torch.tensor(d["corrected_pred"]["input_ids"]).unsqueeze(0).to(device),
                        attention_mask=torch.tensor(d["corrected_pred"]["attention_mask"]).unsqueeze(0).to(device)
                    )
                    feat_2 = model(
                        input_ids=torch.tensor(d["text"]["input_ids"]).unsqueeze(0).to(device),
                        attention_mask=torch.tensor(d["text"]["attention_mask"]).unsqueeze(0).to(device)
                    )
                    f1 = feat_1["sequence_output"].cpu().squeeze(0).sum(0)
                    f2 = feat_2["sequence_output"].cpu().squeeze(0).sum(0)
                    features.append(f1-f2) 
                    logging.info("Shape: {}".format(features[-1].shape))
            else:
                for d in tqdm(train_dataset, desc="Featurizing dataset"):
                    feat_1 = model(d["pred1"].to(device))
                    feat_2 = model(d["pred2"].to(device))
                    features.append(feat_1["phi"].cpu().squeeze()-feat_2["phi"].cpu().squeeze()) 
                    logging.info("Shape: {}".format(features[-1].shape))
        features = torch.stack(features)
        logging.info("Shape: {}".format(features.shape))

        with open("features{}.pkl".format(featuresFileEnding), "wb") as f:
            pkl.dump(features, f)
    
    non_redundant_inds = np.arange(len(features))
    if args.shuffle: 
        logging.info("Shuffling features using seed: {}".format(args.seed))
        non_redundant_inds = np.random.choice(np.arange(len(features)), len(features), replace=False)
        with open("features_shuffleOrder{}.pkl".format(fileEnding), "wb") as f: 
            pkl.dump(non_redundant_inds, f)
    redundant_inds = []
    failure_inds = []

    #Support for checkpointing
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
    while(numPass < MAX_PASSES):
        startInd  = args.start if (numPass==0) else 0

        if startInd == 0:
            indsToTest = non_redundant_inds.copy()

        if numPass == 0 and startInd > len(indsToTest):
            raise ValueError("[main] Cannot start at an index ({}) greater than possible indices ([0,{}])!".format(startInd, len(indsToTest)))

        logging.info("Pass {}/{}".format(numPass+1, MAX_PASSES))
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

            #Print debugging information
            if i%25 == 0: 
                logging.info("Stats:\nTested: {}\nRedundant: {}\nNon-redundant: {}\nFailure: {}".format(i, numRedundant, i-numRedundant-numFailure, numFailure))
                #Support for checkpointing
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
                    constraints = torch.zeros(((featsToConsider.shape[0]+featsToConsider.shape[0]), (featsToConsider[0].shape[0]+featsToConsider.shape[0])))
                    constraints[:featsToConsider.shape[0], :featsToConsider.shape[1]] = featsToConsider
                    # w^T(phi(A) - phi(B)) + eps_i >= 0 <=> w^T(phi(A) - phi(B)) >= -eps_i
                    constraints[:featsToConsider.shape[0], featsToConsider.shape[1]:] = torch.diag(torch.tensor([1]*featsToConsider.shape[0]))
                    # -eps_i + MAX_EPSILON >= 0 <=> eps_i -MAX_EPSILON <=0 <=> eps_i <= MAX_EPSILON
                    constraints[featsToConsider.shape[0]:, featsToConsider.shape[1]:] = torch.diag(torch.tensor([-1]*featsToConsider.shape[0]))
                else: 
                    constraints = featsToConsider

                if args.addEpsilon:
                    # initialGuess = torch.zeros((featsToConsider[0].shape[0]+featsToConsider.shape[0],))
                    # initialGuess = BIG_NUM+torch.zeros((featsToConsider[0].shape[0]+featsToConsider.shape[0],))
                    initialGuess = torch.rand((featsToConsider[0].shape[0]+featsToConsider.shape[0],))
                    initialGuess[featsToConsider[0].shape[0]:] = torch.zeros((featsToConsider.shape[0],))
                else:
                    # initialGuess = torch.zeros(featsToConsider[0].shape)
                    # initialGuess = BIG_NUM+torch.zeros(featsToConsider[0].shape)
                    initialGuess = torch.rand(featsToConsider[0].shape)

                if args.addEpsilon:
                    lowerBound = (-1)*torch.inf
                    upperBound = (0, )*featsToConsider.shape[0]+(MAX_EPSILON, )*featsToConsider.shape[0]
                else: 
                    lowerBound = (-1)*torch.inf
                    upperBound = EPSILON

                m = minimize(
                    fun=objFunc,
                    x0=initialGuess,
                    constraints=LinearConstraint(
                        A=(-1)*constraints,
                        lb=lowerBound,
                        ub=upperBound,
                    ),
                    args=(features[ind].squeeze(), )
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
            if m.x[:features[0].shape[0]].dot(features[ind]) >= 0:#Redundant
                numRedundant += 1
                logging.info("Redundancy @ {}".format(ind))
                redundant_inds.append(ind)
                non_redundant_inds = np.delete(non_redundant_inds, np.where(non_redundant_inds == ind))
            else: #Non-redundant
                pass 
        logging.info("End of Pass {}:\nStats:\nTested: {}\nRedundant: {}\nNon-redundant: {}\nFailure: {}".format(numPass, len(indsToTest), numRedundant, len(indsToTest)-numRedundant-numFailure, numFailure))
        
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

    logging.info("End of all passes:\nStats:\nTested: {}\nRedundant: {}\nNon-redundant: {}\nFailure: {}".format(len(features), len(redundant_inds), len(non_redundant_inds), len(failure_inds)))
#----------------------------------------------------------------------
if __name__ == "__main__":
    main()