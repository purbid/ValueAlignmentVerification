# import os
# import torch
# import logging
# import pickle as pkl
# from tqdm import tqdm
# from datasets import load_dataset
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
#
#
# model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"
# logging_directory_path = "/uufs/chpc.utah.edu/common/home/u1472659/ValueAlignmentVerificationPurbid/logging/"
# os.makedirs(logging_directory_path, exist_ok=True)
# model_name_short = model_name.split('/')[-1]
#
# logging.basicConfig(filename=os.path.join(logging_directory_path, f"{model_name_short}.txt"), filemode='w', level=logging.INFO)
#
# cache_directory = "/scratch/general/vast/u1472659/huggingface_cache/"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map=device,
#     cache_dir=cache_directory,
#     attn_implementation="flash_attention_2",
#     num_labels=1,
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model.to(device)
# print("model downloaded and cached")
#
# logging.info(model)
# dataset = load_dataset('allenai/reward-bench', split='filtered')
#
# features = []
# #for example in dataset:
# for example in tqdm(dataset, desc="Processing examples"):
#
#     prompt = example['prompt']
#     chosen_completion = example['chosen']
#     rejected_completion = example['rejected']
#
#     conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen_completion}]
#     conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected_completion}]
#     conv1_formatted = tokenizer.apply_chat_template(conv1, tokenize=False)
#     conv2_formatted = tokenizer.apply_chat_template(conv2, tokenize=False)
#
#     conv1_tokenized = tokenizer(conv1_formatted, return_tensors="pt", truncation=True).to(device)
#     conv2_tokenized = tokenizer(conv2_formatted, return_tensors="pt", truncation=True).to(device)
#
#     with torch.no_grad():
#
#
#         logging.info(conv1_tokenized)
#         logging.info("\n\n\n now the rejected prompt")
#         logging.info(conv2_tokenized)
#
#         with open("tokenized_chosen.pkl", 'wb') as f:
#             pkl.dump(conv1_tokenized, f)
#
#         with open("tokenized_rejected.pkl", 'wb') as f:
#             pkl.dump(conv2_tokenized, f)
#
#
#         output_1 = model(**conv1_tokenized, output_hidden_states=True)
#         hidden_states1 = output_1.hidden_states
#
#         output_2 = model(**conv2_tokenized, output_hidden_states=True)
#         hidden_states2 = output_2.hidden_states
#
#
#         #### now we get the last token that isn't a PAD
#
#         cls_embedding1 = hidden_states1[-1][:, -1, :].cpu().squeeze()
#         cls_embedding2 = hidden_states2[-1][:, -1, :].cpu().squeeze()
#
#         features.append((cls_embedding1-cls_embedding2).cpu())
#
#         with open("output_1_example_{i}.pkl", 'wb') as f_out1, \
#              open("output_2_example_{i}.pkl", 'wb') as f_out2:
#             pkl.dump(output_1, f_out1)
#             pkl.dump(output_2, f_out2)
#
#
# features = torch.stack(features)
# logging.info("Shape: {}".format(features.shape))
#
# with open("features{}.pkl".format(model_name_short), "wb") as f:
#     pkl.dump(features, f)

import os
import torch
import logging
import argparse
import pickle as pkl
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
    features = []
    for example in tqdm(dataset, desc="Processing examples"):
        prompt = example['prompt']
        chosen_completion = example['chosen']
        rejected_completion = example['rejected']

        conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen_completion}]
        conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected_completion}]

        conv1_formatted = tokenizer.apply_chat_template(conv1, tokenize=False)
        conv2_formatted = tokenizer.apply_chat_template(conv2, tokenize=False)

        conv1_tokenized = tokenizer(conv1_formatted, return_tensors="pt", truncation=True).to(device)
        conv2_tokenized = tokenizer(conv2_formatted, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            output_1 = model(**conv1_tokenized, output_hidden_states=True)
            output_2 = model(**conv2_tokenized, output_hidden_states=True)

            hidden_states1 = output_1.hidden_states
            hidden_states2 = output_2.hidden_states

            # Get the last token embedding that isn't a PAD
            cls_embedding1 = hidden_states1[-1][:, -1, :].cpu().squeeze()
            cls_embedding2 = hidden_states2[-1][:, -1, :].cpu().squeeze()
            

            if args.shorten_size:
                features.append((cls_embedding1 - cls_embedding2)[:args.shorten_size].cpu())
            else:
                features.append((cls_embedding1 - cls_embedding2).cpu())
    return torch.stack(features)


def main():
    args = parse_args()

    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"


    logging_directory_path = "~/logging/"
    logging_directory_path = "/uufs/chpc.utah.edu/common/home/u1472659/ValueAlignmentVerification/logging/"
    os.makedirs(logging_directory_path, exist_ok=True)
    model_name_short = model_name.split('/')[-1]

    print(args.shorten_size)
    print(type(args.shorten_size))
    exit()        

    logging.basicConfig(filename=os.path.join(logging_directory_path, f"{model_name_short}_for_{args.filter_by_subset}.txt"), filemode='w',
                        level=logging.INFO)

    cache_directory = "/scratch/general/vast/u1472659/huggingface_cache/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = load_dataset('allenai/reward-bench', split='raw')

    # Filter dataset if chat_hard is enabled
    filtered_dataset = filter_dataset(dataset, args.filter_by_subset)

    # Only load the model and tokenizer after filtering the dataset
    print("Loading model and tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cache_directory,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    logging.info("Model downloaded and cached")

    # Process the examples
    features = process_examples(model, tokenizer, filtered_dataset, device, args)

    # Save features to a file
    logging.info(f"Shape: {features.shape}")
    with open(f"features_{model_name_short}_for_{args.filter_by_subset}.pkl", "wb") as f:
        pkl.dump(features, f)


if __name__ == "__main__":
    main()

