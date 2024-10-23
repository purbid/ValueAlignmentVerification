import os
import torch
import logging
import pickle as pkl
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"
logging_directory_path = "/uufs/chpc.utah.edu/common/home/u1472659/ValueAlignmentVerificationPurbid/logging/"
os.makedirs(logging_directory_path, exist_ok=True)
model_name_short = model_name.split('/')[-1]

logging.basicConfig(filename=os.path.join(logging_directory_path, f"{model_name_short}.txt"), filemode='w', level=logging.INFO)

cache_directory = "/scratch/general/vast/u1472659/huggingface_cache/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
print("model downloaded and cached")

logging.info(model)
dataset = load_dataset('allenai/reward-bench', split='filtered')

features = []
#for example in dataset:
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


        logging.info(conv1_tokenized)
        logging.info("\n\n\n now the rejected prompt")
        logging.info(conv2_tokenized)
        
        with open("tokenized_chosen.pkl", 'wb') as f:
            pkl.dump(conv1_tokenized, f)

        with open("tokenized_rejected.pkl", 'wb') as f:
            pkl.dump(conv2_tokenized, f)

        



        output_1 = model(**conv1_tokenized, output_hidden_states=True)
        hidden_states1 = output_1.hidden_states

        output_2 = model(**conv2_tokenized, output_hidden_states=True)
        hidden_states2 = output_2.hidden_states

        
        #### now we get the last token that isn't a PAD
        
        cls_embedding1 = hidden_states1[-1][:, -1, :].cpu().squeeze()
        cls_embedding2 = hidden_states2[-1][:, -1, :].cpu().squeeze()
        
        features.append((cls_embedding1-cls_embedding2).cpu())
        
        with open("output_1_example_{i}.pkl", 'wb') as f_out1, \
             open("output_2_example_{i}.pkl", 'wb') as f_out2:
            pkl.dump(output_1, f_out1)
            pkl.dump(output_2, f_out2)

print(features)
features = torch.stack(features)
logging.info("Shape: {}".format(features.shape))

with open("features{}.pkl".format(model_name_short), "wb") as f:
    pkl.dump(features, f)

        #actual scores not needed
        # score1 = output_1.logits[0][0].item()
        # score2 = model(**conv2_tokenized).logits[0][0].item()
        # print(score1)
        # exit()

    # feature_difference = score1 - score2
    # print(f"Feature Difference: {feature_difference}")

