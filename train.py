import argparse
import transformers
from transformers import AutoModel, AutoTokenizer, set_seed
import numpy as np
import torch
import logging
from pathlib import Path
from os.path import exists
import csv
import json
import os
import glob
import re
import pandas as pd
import scipy
from tqdm import tqdm
from sklearn.utils import resample
import math
from datasets import load_dataset
import random

parser = argparse.ArgumentParser()

parser.add_argument(
    "-debug",
    action="store_true",
    help="Boolean flag to enable debug mode"
)

parser.add_argument(
    "-log",
    "--logFile",
    type=str,
    help="Path to file to print logging information",
    default=None
)

parser.add_argument(
    "-cacheDir",
    help="Path to cache location for Huggingface",
    default="/scratch/general/vast/u1419542/huggingface_cache/"
)

parser.add_argument(
    "-dataset",
    choices = [
        "Owishiboo/grammar-correction",
        "jhu-clsp/jfleg"
    ],
    required=True,
)

parser.add_argument(
    "-valSplit",
    type=float,
    help="Fraction of instances to use as validation data",
    default=0.3
)

parser.add_argument(
    "-maxSamples",
    type=int,
    help="[Deprecated] Maximum no. of samples to be used in train set",
    default=1000000
)

parser.add_argument(
    "-numEpochs",
    type=int,
    help="Number of epochs to train model for",
    default=5
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Batch size of dataloader",
    default=4
)

parser.add_argument(
    "-learningRate",
    type=float,
    help="Learning rate for optimizer",
    default=0.001
)

parser.add_argument(
    "-weightDecay",
    type=float,
    help="Weight Decay for optimizer",
    default=0.01
)

parser.add_argument(
    "-modelPath",
    help="Path to model to use",
    default="microsoft/deberta-v3-large"
)

parser.add_argument(
    "-maxSteps",
    type=int,
    help="Maximum number of optimization steps allowed",
    default=-1
)

parser.add_argument(
    "-saveModelPath",
    type=str,
    help="Path to save trained model",
    default="./models/"
)

parser.add_argument(
    "-saveModelName",
    type=str,
    help="Name to save trained model as",
    default="preference_model.pt"
)

parser.add_argument(
    '-seed', 
    type=int, 
    help='Random seed', 
    default=11892
)

parser.add_argument(
    "-maxLength",
    type=int,
    help="Max sequence length",
    default=1024
)

parser.add_argument(
    "-introspect",
    type=str,
    help="Path to model to introspect",
    default=None,
)

#---------------------------------------------------------------------------
class OwishibooGrammarCorrectionDataset:
    def __init__(self, input, target, tokenizer, maxLength=1024):
        assert len(input)==len(target), f"[OwishibooGrammarCorrectionDataset] Expected input ({len(input)}) and target ({len(target)}) to be of the same length!"
        self.input = input
        self.target = target
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.dataset = "Owishiboo/grammar-correction"

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, item):
        curInstance = {}
        curInstance["preferred"] = self.tokenizer.encode_plus(
            self.target[item],
            padding="max_length",
            truncation=True,
            max_length=self.maxLength,
            return_tensors="pt"
        )

        curInstance["dispreferred"] = self.tokenizer.encode_plus(
            self.input[item],
            padding="max_length",
            truncation=True,
            max_length=self.maxLength,
            return_tensors="pt"
        )
        return curInstance["preferred"]["input_ids"].squeeze(), curInstance["preferred"]["attention_mask"].squeeze(), curInstance["dispreferred"]["input_ids"].squeeze(), curInstance["dispreferred"]["attention_mask"].squeeze(), self.target[item], self.input[item]
#---------------------------------------------------------------------------
class JFLEGDataset:
    def __init__(self, input, target, tokenizer, maxLength=1024):
        assert len(input)==len(target), f"[JFLEGDataset] Expected input ({len(input)}) and target ({len(target)}) to be of the same length!"
        self.input = input
        self.target = target
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.dataset = "jhu-clsp/jfleg"

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, item):
        curInstance = {}
        curInstance["preferred"] = self.tokenizer.encode_plus(
            self.target[item],
            padding="max_length",
            truncation=True,
            max_length=self.maxLength,
            return_tensors="pt"
        )

        curInstance["dispreferred"] = self.tokenizer.encode_plus(
            self.input[item],
            padding="max_length",
            truncation=True,
            max_length=self.maxLength,
            return_tensors="pt"
        )
        return curInstance["preferred"]["input_ids"].squeeze(), curInstance["preferred"]["attention_mask"].squeeze(), curInstance["dispreferred"]["input_ids"].squeeze(), curInstance["dispreferred"]["attention_mask"].squeeze(), self.target[item], self.input[item]
#---------------------------------------------------------------------------
def collateBatch(batch):
    pref_input_ids, pref_attention_mask, dispref_input_ids, dispref_attention_mask, prefText, disprefText = zip(*batch)
    return {
        "preferred" : {
            "input_ids":torch.stack(pref_input_ids),
            "attention_mask":torch.stack(pref_attention_mask),
        },
        "dispreferred": {
            "input_ids":torch.stack(dispref_input_ids),
            "attention_mask":torch.stack(dispref_attention_mask),
        },
        "text": {
            "preferred": prefText,
            "dispreferred": disprefText,
        }
    }
#---------------------------------------------------------------------------
def createDataLoader(df, dataset, batchSize, tokenizer, maxLength=1024):
    if dataset == "Owishiboo/grammar-correction":
        ds = OwishibooGrammarCorrectionDataset(
            input = df["input"].to_numpy(), 
            target = df["target"].to_numpy(), 
            tokenizer = tokenizer,
            maxLength = maxLength   
        )
    elif dataset == "jhu-clsp/jfleg":
        ds = JFLEGDataset(
            input = df["sentence"].to_numpy(), 
            target = pd.DataFrame(df["corrections"].to_list(), columns=['ann1', 'ann2', 'ann3', 'ann4'])["ann1"].to_numpy(), 
            tokenizer = tokenizer,
            maxLength = maxLength   
        )
    else:
        raise ValueError("[createDataLoader] {} is not supported!".format(dataset))

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batchSize,
        num_workers=0,
        shuffle=True,
        collate_fn=collateBatch,
    )
#---------------------------------------------------------------------------
class ClassificationHead(torch.nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, num_labels):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output
#---------------------------------------------------------------------------
class PreferenceModel(torch.nn.Module):
    def __init__(self, modelPath, cache_dir=" ~/.cache/huggingface/", device="cpu"):
        super(PreferenceModel, self).__init__()
        self.modelPath = modelPath 
        self.model = AutoModel.from_pretrained(self.modelPath, cache_dir=cache_dir)
        self.classifier= ClassificationHead(
            hidden_size=self.model.config.hidden_size,
            hidden_dropout_prob=0.1,
            num_labels=1,
        )
        self.device=device

    def forward(self, inputs):
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
        outputs = self.model(**inputs)
        return self.classifier(outputs["last_hidden_state"])

    def to(self, device):
        self.device = device 
        self = super().to(device)
        return self 
#---------------------------------------------------------------------------
def trainModel(model, dataLoader, lossFunction, optimizer, device, scheduler, maxSteps=-1, debug=False):
    model.to(device)
    model.train()

    losses = []
    numExamples = 0
    numBatch = 0
    numSteps = 0
    for d in tqdm(dataLoader, desc="Train data"):
        numBatch += 1
        numExamples += len(d)
        outputsPref = model(d["preferred"])
        outputsDispref = model(d["dispreferred"])

        loss = lossFunction(outputsPref, outputsDispref)

        if debug:
            logging.info(f"Batch: {numBatch}/{len(dataLoader)}, Loss: {loss.item()}")

        losses.append(loss.item())
        #Zero out gradients from previous batches
        optimizer.zero_grad()
        #Backwardpropagate the losses
        loss.backward()
        #Avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #Perform a step of optimization
        optimizer.step()
        numSteps += 1
        if maxSteps and numSteps >= maxSteps:
            break
    scheduler.step()
    return np.mean(losses)
#---------------------------------------------------------------------------
def testModel(model, dataLoader, dataDesc="Test batch", writeToFile=None):
    model.eval()
    with torch.no_grad():
        numExamples = 0
        allPreds = []
        failure = [] 
        for d in tqdm(dataLoader, desc=dataDesc):
            numExamples += len(d)
            outputsPref = model(d["preferred"])
            outputsDispref = model(d["dispreferred"])
            allPreds.extend((outputsPref>=outputsDispref).view(-1).tolist())
            if writeToFile:
                for i in torch.where((outputsPref>=outputsDispref).view(-1)==False)[0].tolist():
                    failure.append(((d["text"]["preferred"][i], outputsPref[i][0].item()), (d["text"]["dispreferred"][i], outputsDispref[i][0].item()))) 
    if writeToFile:
        with open(writeToFile, "w") as f: 
            for fEx in failure:
                f.write("Difference: {}".format(fEx[0][1]-fEx[1][1]))
                f.write("\n")
                f.write("{}: {}".format(fEx[0][1], fEx[0][0]))
                f.write("\n")
                f.write("{}: {}".format(fEx[1][1], fEx[1][0]))
                f.write("\n")
                f.write("*"*20)
                f.write("\n")
    return (np.array(allPreds)).sum()/len(allPreds)
#---------------------------------------------------------------------------
def preferenceLoss(outputsPref, outputsDispref):
    outputs = (outputsPref[0] - outputsDispref[0]).view(-1)
    loss = -torch.mean(torch.log(torch.sigmoid(outputs)))
    return loss
#---------------------------------------------------------------------------
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
#---------------------------------------------------------------------------
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
#---------------------------------------------------------------------------
def readFile(fileName):
    data = []
    if fileName.endswith(".csv"):
        with open(fileName, "r") as f: 
            data = list(csv.DictReader(f))
    elif fileName.endswith(".json"):
        with open(fileName, "r") as f: 
            data = json.load(f)
    elif fileName.endswith(".jsonl"):
        with open(fileName, "r") as f: 
            for line in f:
                data.append(json.loads(line))
    else: 
        raise ValueError("[readFile] {} has unrecognized file extension!".format(fileName))
    return data
#---------------------------------------------------------------------------
def main():
    args = parser.parse_args()

    # Set seed before initializing model.
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.logFile:
        checkFile(args.logFile)
        logging.basicConfig(filename=args.logFile, filemode='w', level=logging.INFO)
    elif args.debug:
        logging.basicConfig(filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(filemode='w', level=logging.ERROR)

    if args.batchSize <= 0:
        raise ValueError("[main] Batch Size has to be a positive number!")
    if args.learningRate <= 0:
        raise ValueError("[main] Learning rate has to be a positive number!")
    if args.numEpochs <= 0:
        raise ValueError("[main] numEpochs has to be a positive number!")
    if args.maxSamples <= 0:
        raise ValueError("[main] maxSamples has to be a positive number!")
    if args.maxSteps <= 0:
        logging.warning("maxSteps cannot be non-positive!")
        args.maxSteps = -1
    if args.valSplit >= 1 or args.valSplit <= 0:
        raise ValueError("valSplit has to be between 0 and 1!")

    checkIfExists(args.saveModelPath, True, True)

    logging.info(args)

    if args.dataset == "Owishiboo/grammar-correction":
        ds = load_dataset(args.dataset, split="train")
        ds = ds.shuffle()
        ds = ds.train_test_split(test_size=args.valSplit)
        trainData = ds["train"]
        valData = ds["test"]
    elif args.dataset == "jhu-clsp/jfleg":
        trainData = load_dataset(args.dataset, split="validation")
        valData = load_dataset(args.dataset, split="test")
    else:
        raise ValueError("[main] {} is not supported!".format(args.dataset))

    trainDF = pd.DataFrame.from_records(trainData)
    valDF = pd.DataFrame.from_records(valData)

    trainDF.to_json("{}train.json".format(args.saveModelPath), orient="records")
    valDF.to_json("{}val.json".format(args.saveModelPath), orient="records")

    tokenizer = AutoTokenizer.from_pretrained(args.modelPath, cache_dir=args.cacheDir)

    trainDataLoader = createDataLoader(trainDF, args.dataset, args.batchSize, tokenizer, args.maxLength)
    valDataLoader = createDataLoader(valDF, args.dataset, args.batchSize, tokenizer, args.maxLength)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    if args.introspect: 
        logging.info("Introspecting model at {}".format(args.introspect))
        model = torch.load(args.introspect)
        valAcc = testModel(
            model, 
            valDataLoader, 
            dataDesc="Validation batch", 
            writeToFile="./introspect.txt"
        )
        logging.info("Validation Accuracy: {}".format(valAcc))
    else:
        model = PreferenceModel(args.modelPath, args.cacheDir, device)

        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.learningRate, 
            weight_decay=args.weightDecay,
        )
        totalSteps = args.numEpochs
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=totalSteps
        )
        # lossFunction = torch.nn.CrossEntropyLoss().to(device)
        lossFunction = preferenceLoss

        numTrainingSteps = args.numEpochs * len(trainDataLoader)
        if args.maxSteps == -1:
            args.maxSteps = numTrainingSteps
        elif args.maxSteps > 0:
            args.maxSteps = math.ceil(args.maxSteps/len(trainDataLoader))
        else: 
            raise ValueError(f"Maximum no. of steps (maxSteps) has to be positive!")

        logging.info("Dataset: {}".format(args.dataset))
        logging.info("Train Data: {}".format(len(trainDF)))
        logging.info("Validation Data: {}".format(len(valDF)))
        logging.info("No. of Epochs: {}".format(args.numEpochs))
        logging.info("Batch Size: {}".format(args.batchSize))
        logging.info("Learning Rate: {}".format(args.learningRate))
        logging.info("Trained model location: {}{}".format(args.saveModelPath, args.saveModelName))

        bestValAcc = 0
        for epoch in range(args.numEpochs):
            curLoss = trainModel(model, trainDataLoader, lossFunction, optimizer, device, scheduler, args.maxSteps, args.debug)
            args.maxSteps -= len(trainDataLoader)

            valAcc = testModel(
                model, 
                valDataLoader, 
                dataDesc="Validation batch", 
            )

            if bestValAcc <= valAcc:
                bestValAcc = valAcc
                torch.save(model, f"{args.saveModelPath}{args.saveModelName}")
            
            logging.info("Epoch {}/{}\nTraining Loss: {:0.2f}\nValidation Accuracy: {:0.2f}%".format(epoch+1, args.numEpochs, curLoss, valAcc*100))
            logging.info("*****")
#---------------------------------------------------------------------------
if __name__ == "__main__":
    main()


