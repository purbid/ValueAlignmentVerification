import os
import torch
import logging
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import AutoPeftModelForSequenceClassification
from peft import PeftModel
from  safetensors import safe_open
from peft import PeftConfig


import glob
import torch.nn as nn


filter_subsets_dict = {'chat': ['alpacaeval-easy', 'alpacaeval-length', 'alpacaeval-hard', 'mt-bench-easy', 'mt-bench-medium'],
                        'chat_hard': [ 'mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor', 'llmbar-adver-GPTInst', 'llmbar-adver-GPTOut', 'llmbar-adver-manual'],
                        'safety': ['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse', 'xstest-should-respond', 'do not answer'],
                        'reasoning': ['math-prm', 'hep-cpp', 'hep-go', 'hep-java', 'hep-js', 'hep-python', 'hep-rust']}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process reward bench dataset with optional hard subset filtering.")
    parser.add_argument("--filter_by_subset", default="chat_hard", help="Filter for hard subsets in the reward bench dataset.")
    #parser.add_argument("--chat", action="store_true", help="Filter for regular/easy subsets in the reward bench dataset.")
    parser.add_argument("--shorten_size", default="768", help="Take full feature length or only the first 768 dimensions")  
    parser.add_argument("--using_peft", action='store_true', help='If set, use a fine-tuned PEFT model, otherwise use the base model')
    
    return parser.parse_args()


def filter_dataset(dataset, filter_by_subset):

    """Filter the dataset based on the subset field if chat_hard is enabled."""
    if filter_by_subset!= '':
        include_only = filter_subsets_dict[filter_by_subset]
        '''
        [
            'mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor',
            'llmbar-adver-GPTInst', 'llmbar-adver-GPTOut', 'llmbar-adver-manual'
        ]''' 

        # Filter dataset based on the 'subset' field
        filtered_dataset = [example for example in dataset if example['subset'] in include_only]
        logging.info("the length of chat hard is {}".format(len(filtered_dataset)))
        
        return filtered_dataset

    return dataset

def process_examples(model, tokenizer, dataset, device, args):
    """Process dataset examples and compute features."""
    
    features_chosen = []
    features_chosen_full_length = []

    features_rejected = []
    features_rejected_full_length = []


    features_diff = []
    
    chosen_scores = []
    rejected_scores = []

    for example in tqdm(dataset, desc="Processing examples"):
        prompt = example['prompt']
        chosen_completion = example['chosen']
        rejected_completion = example['rejected']

        conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen_completion}]
        conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected_completion}]

        conv1_formatted = tokenizer.apply_chat_template(conv1, tokenize=False)
        conv2_formatted = tokenizer.apply_chat_template(conv2, tokenize=False)

        conv1_tokenized = tokenizer(conv1_formatted, return_tensors="pt", truncation=True).to(device).to(torch.bfloat16)
        conv2_tokenized = tokenizer(conv2_formatted, return_tensors="pt", truncation=True).to(device).to(torch.bfloat16)


        # conv1_tokenized = conv1_tokenized.to(dtype=torch.float16, device=device)
        # conv2_tokenized = conv2_tokenized.to(dtype=torch.float16, device=device)

        # with torch.no_grad():
        with torch.amp.autocast('cuda'):
            output_1 = model(**conv1_tokenized, output_hidden_states=True)
            output_2 = model(**conv2_tokenized, output_hidden_states=True)

            score_chosen = output_1.logits[0][0].item()
            score_rejected = output_2.logits[0][0].item()

            chosen_scores.append(score_chosen)
            rejected_scores.append(score_rejected)




            if args.using_peft:
                hidden_states1 = output_1.hidden_states[-1][:, -1, :]
                hidden_states2 = output_2.hidden_states[-1][:, -1, :]

                cls_embedding1 = model.score[0](hidden_states1).cpu().squeeze()
                cls_embedding2 = model.score[0](hidden_states2).cpu().squeeze()

            else:

                hidden_states1 = output_1.hidden_states
                hidden_states2 = output_2.hidden_states

                # Get the last token embedding that isn't a PAD

                cls_embedding1 = hidden_states1[-1][:, -1, :].cpu().squeeze()
                cls_embedding2 = hidden_states2[-1][:, -1, :].cpu().squeeze()

            
            if args.shorten_size:
                features_chosen.append(cls_embedding1[:int(args.shorten_size)].cpu())
                features_rejected.append(cls_embedding2[:int(args.shorten_size)].cpu())

                features_chosen_full_length.append(cls_embedding1.cpu())
                features_rejected_full_length.append(cls_embedding2.cpu())  
                features_diff.append((cls_embedding1 - cls_embedding2).cpu())

                # features.append((cls_embedding1 - cls_embedding2)[:int(args.shorten_size)].cpu())
                # features_full_length.append((cls_embedding1 - cls_embedding2).cpu())
            else:
                features_chosen.append(cls_embedding1.cpu())
                features_rejected.append(cls_embedding2.cpu())

    return torch.stack(features_chosen), torch.stack(features_rejected), \
            torch.stack(features_chosen_full_length),  torch.stack(features_rejected_full_length), \
            torch.stack(features_diff), chosen_scores, rejected_scores


def main():
    args = parse_args()
    # base_model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"
    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    # model_name = "/scratch/general/vast/u1472659/lora_llama_ft/merged_model/"
    # tokenizer_name = 'Skywork/Skywork-Reward-Llama-3.1-8B'
    peft_name = '/scratch/general/vast/u1472659/lora_llama_ft/Skywork-Reward-Llama-3.1-8B-v0.2_BT_RM_len512_lora32_1e-05_dataSkywork-Reward-Preference-80K-v0.2/'
    # logging_directory_path = "~/logging/"
    logging_directory_path = "/uufs/chpc.utah.edu/common/home/u1472659/ValueAlignmentVerification/logging/"
    os.makedirs(logging_directory_path, exist_ok=True)
    model_name_short = model_name.split('/')[-1]

    logging.basicConfig(filename=os.path.join(logging_directory_path, f"{model_name_short}_for_{args.filter_by_subset}_{args.shorten_size}.txt"), filemode='w',
                        level=logging.INFO)

    cache_directory = "/scratch/general/vast/u1472659/huggingface_cache/"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = load_dataset('allenai/reward-bench', split='raw')

    # Filter dataset if chat_hard is enabled
    filtered_dataset = filter_dataset(dataset, args.filter_by_subset)

    # Only load the model and tokenizer after filtering the dataset
    print("Loading model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cache_directory,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )



    get_modified_model = False
    print(args)
    if args.using_peft:
    # if 'freeze' in script_args.peft_name or script_args.freeze_pretrained:
        print('loading freeze nonlinear parameters')
        tensors = {}
        path_list = glob.glob(os.path.join(peft_name, "adapter-*.safetensors"))
        
        for path in path_list:
            with safe_open(path, framework="pt", device=0) as f:
                for k in f.keys():
                    if 'score' in k:
                        tensors[k] = f.get_tensor(k)

        # use the same structure as the training
        mlp_layer = nn.Sequential(
            nn.Linear(model.config.hidden_size, 512, dtype=torch.float16),  
            nn.ReLU(),
            nn.Linear(512, 1, dtype=torch.float16)  
        )

        mlp_layer.to(device)
        # Replace the classifier with the MLP
        model.score = mlp_layer
        model.load_state_dict(tensors, strict=False)
        

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # If there is lora for loading
    if os.path.exists(peft_name):
        print("yes this is peft model")
        model = PeftModel.from_pretrained(model, peft_name)
    if hasattr(model, 'merge_and_unload'):
        print("gonf fo rmerge")
        model = model.merge_and_unload()


    print("we have peft merged model")

    

    # peft_config = PeftConfig.from_pretrained(peft_name)

    # peft_config.init_lora_weights = True

    # model.add_adapter(peft_config)
    # model.enable_adapters()


    # model.resize_token_embeddings(len(tokenizer))
    # model.config.pad_token_id = tokenizer.pad_token_id

    # # If there is lora for loading
    # if len(peft_name) and os.path.exists(peft_name):
    #     model = PeftModel.from_pretrained(model, PeftModel)
    # if hasattr(model, 'merge_and_unload'):

    #     model = model.merge_and_unload()

    
    model.to(device)
    logging.info("Model downloaded and cached")

    # Process the examples
    features_chosen, features_rejected, features_chosen_full_length, features_rejected_full_length, \
    features_diff, chosen_scores, rejected_scores = process_examples(model, tokenizer, filtered_dataset, device, args)
    
    ###check the 87.4% accuracy for skyworks model.
    chosen_np_arr = np.array(chosen_scores)
    rejected_np_arr = np.array(rejected_scores)

    comparison = chosen_np_arr > rejected_np_arr

    # Calculate the percentage where list A's elements are greater
    percentage = np.mean(comparison) * 100

    logging.info("the percentage where score of chosen is greater than rej is {}".format(percentage))
    print("the percentage where score of chosen is greater than rej is {}".format(percentage))


    if args.using_peft:
        base_directory_for_features = "features"
        model_directory = os.path.join(base_directory_for_features, model_name_short)
        os.makedirs(model_directory, exist_ok=True)

        with open(os.path.join(model_directory, 'scores_chosen.pkl'), 'wb') as f:
            pkl.dump(chosen_np_arr, f)

        with open(os.path.join(model_directory, 'scores_rejected.pkl'), 'wb') as f:
            pkl.dump(rejected_np_arr, f)
        
        # Save features to a file
        logging.info(f"Shape of features chosen is : {features_chosen.shape}")
        logging.info(f"Shape of features rejection is : {features_rejected.shape}")

        for save_name, features_var in zip(['features_chosen', 'features_rejected', 'features_chosen_full_length', 'features_rejected_full_length'], [features_chosen, features_rejected, features_chosen_full_length, features_rejected_full_length]):

            if len(features_var) > 0:
                if 'full_length' in save_name:
                    features_size = '' #no suffix neaded, default is 4096
                else:
                    features_size = str(args.shorten_size)

                with open(os.path.join(model_directory, '{}_{}{}.pkl'.format(save_name, args.filter_by_subset, features_size)), 'wb') as f:
                    pkl.dump(features_var, f)

        if len(features_diff) > 0:

            print("writing difference in features")
            with open(os.path.join(model_directory, 'features_diff_full_length_{}.pkl'.format(args.filter_by_subset)), 'wb') as f:
                pkl.dump(features_diff, f)

if __name__ == "__main__":


    main()

