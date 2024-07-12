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

NUM_RANDOM = 5

PROMPTS = {
    "llama2": {
        "jfleg": {
            "prompt": """### Question:
Which among the following pieces of text is more fluent?

### Options:
(a) {text1}
(b) {text2}

### Answer:
""",
            "choices": [
                "{prompt}(a)",
                "{prompt}(b)"
            ]
        } 
    },
    "gpt2": {
        "jfleg": {
            "prompt": """Question: Which among the following pieces of text is more fluent?

Options:
(a) {text1}
(b) {text2}

Answer: The most fluent piece of text among the options is """,
            "choices": [
                "{prompt}(a)",
                "{prompt}(b)"
            ]
        } 
    },
    "flan-t5": {
        "jfleg": {
            "prompt":"""Question: Which among the following pieces of text is more fluent?

Choices:
(1) {text1}
(2) {text2}

Answer: The most fluent piece of text among the choices is """,
            "choices": [
                "(1)",
                "(2)"
            ]
        }
    }
}

MODEL_2_KEY = {
    "google/flan-t5-small": "flan-t5",
    "google/flan-t5-base": "flan-t5",
    "google/flan-t5-large": "flan-t5",
    "google/flan-t5-xl": "flan-t5",
    "google/flan-t5-xxl": "flan-t5",
    "openai-community/gpt2": "gpt2",
    "meta-llama/Llama-2-7b-hf": "llama2",
}

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
    "-hf_token",
    type=str,
    help="Huggingface token",
    default=""
)

parser.add_argument(
    "-data",
    type=str,
    help="Path to pkl file containing data",
    required=True,
)

parser.add_argument(
    "-input_column",
    type=str, 
    help="The column name of input.",
    default="input"
)

parser.add_argument(
    "-target_column",
    type=str, 
    help="The column name of target.",
    default="target"
)

parser.add_argument(
    "-indsDir",
    type=str,
    help="Path to folder containing redundant/non-redundant/failure indices",
    required=True,
)

parser.add_argument(
    "-fileAddendum",
    type=str,
    help="Addendum to pkl files containing indices in indsDir",
    default="",
)

parser.add_argument(
    "-model",
    type=str,
    help="Path to HF model",
    choices=[
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
        "openai-community/gpt2",
        "meta-llama/Llama-2-7b-hf",
    ]
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Batch size of dataloader",
    default=1
)

parser.add_argument(
    "-maxLength",
    type=int,
    help="Max sequence length",
    default=1024
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
def loadModelTokenizer(modelPath, device="cpu", cache_dir="~", hf_token=""): 
    if "flan-t5" in modelPath:
        tokenizer = AutoTokenizer.from_pretrained(modelPath, cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(modelPath, cache_dir=cache_dir)
        model = model.to(device)
    elif "gpt2" in modelPath:
        model = AutoModelForCausalLM.from_pretrained(modelPath, cache_dir = cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(modelPath, cache_dir = cache_dir)
        model = model.to(device)
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings()
    elif "llama" in modelPath:
        model = AutoModelForCausalLM.from_pretrained(modelPath, cache_dir = cache_dir, token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(modelPath, cache_dir = cache_dir, token=hf_token)
        model = model.to(device)
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings()
    else: 
        raise ValueError(f"[loadModelTokenizer] Unrecognized model: {modelPath}")
    return model, tokenizer
#----------------------------------------------------------------------
class DataPreprocessor:
    def __init__(self, input_column, target_column, modelName, dataset="jfleg"):
        self.input_column = input_column 
        self.target_column = target_column
        self.modelName = modelName
        self.dataset = dataset

    def preprocess(self, instance):
        targetPos = 1+(np.random.rand() > 0.5)
        if targetPos == 1:
            prompt = PROMPTS[MODEL_2_KEY[self.modelName]][self.dataset]["prompt"].format(
                        text1=instance[self.target_column][0],
                        text2=instance[self.input_column],
                    )
        else: 
            prompt = PROMPTS[MODEL_2_KEY[self.modelName]][self.dataset]["prompt"].format(
                        text1=instance[self.input_column],
                        text2=instance[self.target_column][0],
                    )
        if "gpt2" in self.modelName or "llama" in self.modelName: 
            instance.update({
                "prompt": prompt,
                "choices": [choice.format(prompt=prompt) for choice in PROMPTS[MODEL_2_KEY[self.modelName]][self.dataset]["choices"]],
                "label": targetPos,
            })
        else:
            instance.update({
                "prompt": prompt,
                "choices": [choice for choice in PROMPTS[MODEL_2_KEY[self.modelName]][self.dataset]["choices"]],
                "label": targetPos,
            })
        return instance
#----------------------------------------------------------------------
class TestDataset:
    def __init__(self, prompt, choices, label, tokenizer, maxLength=1024):
        assert len(prompt)==len(label), f"[TestDataset] Expected input ({len(input)}) and target ({len(label)}) to be of the same length!"
        self.prompt = prompt
        self.choices = choices
        self.label = label
        self.tokenizer = tokenizer
        self.maxLength = maxLength

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, item):
        assert len(self.choices[item]) == 2, "Only two choices allowed!"

        curInstance = {
            "label": self.label[item],
        } 

        text = self.tokenizer.encode_plus(
            self.prompt[item],
            padding="max_length",
            truncation=True,
            max_length=self.maxLength,
            return_tensors="pt"
        )

        text_completion_1 = self.tokenizer.encode_plus(
            self.choices[item][0],
            padding="max_length",
            truncation=True,
            max_length=self.maxLength,
            return_tensors="pt"
        )

        text_completion_2 = self.tokenizer.encode_plus(
            self.choices[item][1],
            padding="max_length",
            truncation=True,
            max_length=self.maxLength,
            return_tensors="pt"
        )

        return text["input_ids"].squeeze(0), text_completion_1["input_ids"].squeeze(0), text_completion_2["input_ids"].squeeze(0), curInstance["label"]
#---------------------------------------------------------------------------
def createDataLoader(df, batchSize, tokenizer, maxLength=1024):
    ds = TestDataset(
        prompt = df["prompt"].to_numpy(), 
        choices = df["choices"].to_numpy(), 
        label = df["label"].to_numpy(), 
        tokenizer = tokenizer,
        maxLength = maxLength   
    )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batchSize,
        num_workers=0,
        shuffle=True,
    )
#----------------------------------------------------------------------
def testModel(modelName, model, tokenizer, dataLoader, device, dataDesc="Test batch"):
    model.eval()
    with torch.no_grad():
        numExamples = 0
        allPreds = []
        loss = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        for inp, text1, text2, label in tqdm(dataLoader, desc=dataDesc):
            inp = inp.to(device)
            text1 = text1.to(device)
            text2 = text2.to(device)
            numExamples += len(label)
            if "gpt2" in modelName or "llama" in modelName:
                outputs1 = model(text1)
                outputs2 = model(text2)
                assert outputs1["logits"].shape[0] == 1
                assert outputs2["logits"].shape[0] == 1
                # modText1 = text1.squeeze(0).clone()
                # modText2 = text2.squeeze(0).clone()
                # modText1[:min(torch.where(inp.squeeze(0) == tokenizer.pad_token_id)[0]).item()] = tokenizer.pad_token_id
                # modText2[:min(torch.where(inp.squeeze(0) == tokenizer.pad_token_id)[0]).item()] = tokenizer.pad_token_id
                # loss1 = loss(outputs1["logits"].cpu().squeeze(0), modText1.cpu().squeeze(0))
                # loss2 = loss(outputs2["logits"].cpu().squeeze(0), modText2.cpu().squeeze(0))

                # comp1 = tokenizer.encode("Choice (1)", return_tensors="pt").to(device)
                # comp2 = tokenizer.encode("Choice (2)", return_tensors="pt").to(device)
                # outComp1 = model(comp1)
                # outComp2 = model(comp2)
                # prob1 = torch.exp(-loss1)/torch.exp(-loss(outComp1["logits"].squeeze(0), comp1.squeeze(0)))
                # prob2 = torch.exp(-loss2)/torch.exp(-loss(outComp2["logits"].squeeze(0), comp2.squeeze(0)))
                # pred = 1+(prob1.cpu() < prob2.cpu())
                # logging.info("\nLabels: {}\nPreds: {}\n*******".format(label[0], pred))
                loss1 = loss(outputs1["logits"].cpu().squeeze(0), text1.cpu().squeeze(0))
                loss2 = loss(outputs2["logits"].cpu().squeeze(0), text2.cpu().squeeze(0))
                pred = 1+(loss1 > loss2)
            else:
                outputs1 = model(input_ids=inp, labels=text1)
                outputs2 = model(input_ids=inp, labels=text2)
                pred = 1+(outputs1["loss"].cpu() > outputs2["loss"].cpu())
            allPreds.extend((label==pred).view(-1).tolist())
    return (np.array(allPreds)).sum()/len(allPreds)
#---------------------------------------------------------------------------
def main():
    args = parser.parse_args()

    assert args.batchSize == 1, "Loss calculation woulf fail if batch size != 1"

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

    checkFile(args.data, ".pkl")
    checkIfExists(args.indsDir, isDir=True, createIfNotExists=False)
    checkFile(args.indsDir + "redundant" + args.fileAddendum + ".pkl")
    checkFile(args.indsDir + "non_redundant" + args.fileAddendum + ".pkl")

    data = readFile(args.data)
    redundant_inds = readFile(args.indsDir + "redundant" + args.fileAddendum + ".pkl")
    non_redundant_inds = readFile(args.indsDir + "non_redundant" + args.fileAddendum + ".pkl")

    redundantData = np.array(data)[redundant_inds]
    nonRedundantData = np.array(data)[non_redundant_inds]
    randomData = []
    for _ in range(NUM_RANDOM):
        randomData.append(np.random.choice(data, len(nonRedundantData), replace=False))

    dp = DataPreprocessor(args.input_column, args.target_column, args.model)
    redundantData = list(map(dp.preprocess, redundantData))
    nonRedundantData = list(map(dp.preprocess, nonRedundantData))
    proRandomData = []
    for iRan in range(NUM_RANDOM):
        proRandomData.append(list(map(dp.preprocess, randomData[iRan])))

    #Print samples
    randInd = np.random.choice(len(nonRedundantData), 1)[0]
    randInst = nonRedundantData[randInd]
    logging.info("Sample input:\nInput:\n{}\n\nText completion 1:\n{}\n\nText completion 1:\n{}\n\nLabel:\n{}\n".format(randInst["prompt"], randInst["choices"][0], randInst["choices"][1], randInst["label"]))
    #Print samples

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model, tokenizer = loadModelTokenizer(args.model, device, args.cache_dir, args.hf_token)

    redDF = pd.DataFrame.from_records(redundantData)
    nonDF = pd.DataFrame.from_records(nonRedundantData)
    randomDF = []
    for iRan in range(NUM_RANDOM):
        randomDF.append(pd.DataFrame.from_records(proRandomData[iRan]))

    redDataLoader = createDataLoader(redDF, args.batchSize, tokenizer, args.maxLength)
    nonDataLoader = createDataLoader(nonDF, args.batchSize, tokenizer, args.maxLength)
    randomDataLoader = []
    for iRan in range(NUM_RANDOM):
        randomDataLoader.append(createDataLoader(randomDF[iRan], args.batchSize, tokenizer, args.maxLength))

    logging.info("\nAccuracy:") 
    redAccuracy = testModel(args.model, model, tokenizer, redDataLoader, device, "Redundant")
    logging.info("\n\tRedundant ({:0.2f}%): {:0.2f}%".format((len(redundantData)/len(data))*100, redAccuracy*100))
    
    nonAccuracy = testModel(args.model, model, tokenizer, nonDataLoader, device, "Non-redundant")
    logging.info("\n\tNon-Redundant ({:0.2f}%): {:0.2f}%".format((len(nonRedundantData)/len(data))*100, nonAccuracy*100))

    randomAccuracy = []
    for iRan in range(NUM_RANDOM):
        randomAccuracy.append(testModel(args.model, model, tokenizer, randomDataLoader[iRan], device, "Random"))
        logging.info("\n\t\t({}) Random ({:0.2f}%): {:0.2f}%".format(iRan+1, (len(randomData[0])/len(data))*100, randomAccuracy[-1]*100))
    logging.info("\n\tRandom ({:0.2f}%): {:0.2f}%".format((len(randomData[0])/len(data))*100, np.mean(randomAccuracy)*100))


    logging.info("\nAccuracy:\n\tRedundant ({:0.2f}%): {:0.2f}%\n\tNon-Redundant ({:0.2f}%): {:0.2f}%\n\tRandom ({:0.2f}%): {:0.2f}%".format((len(redundantData)/len(data))*100, redAccuracy*100, (len(nonRedundantData)/len(data))*100, nonAccuracy*100, (len(randomData[0])/len(data))*100, np.mean(randomAccuracy)*100))
#----------------------------------------------------------------------
if __name__ == "__main__":
    main()