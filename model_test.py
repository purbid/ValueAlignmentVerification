import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load Skywork reward model and tokenizer
model_name = "Skywork/Skywork-Reward-Llama-3.1-8B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    # attn_implementation="flash_attention_2",
    num_labels=1,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)


from datasets import load_dataset

dataset = load_dataset('allenai/reward-bench', split='filtered')

for example in dataset:
    prompt = example['prompt']
    chosen_completion = example['chosen']
    rejected_completion = example['rejected']

    conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen_completion}]
    conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected_completion}]

    # Tokenize and score using the Skywork model
    conv1_tokenized = tokenizer(conv1, return_tensors="pt", truncation=True, padding="max_length", max_length=2048).to(device)
    conv2_tokenized = tokenizer(conv2, return_tensors="pt", truncation=True, padding="max_length", max_length=2048).to(device)

    with torch.no_grad():
        output_1 = model(**conv1_tokenized)
        print(type(output_1))
        score1 = output_1.logits[0][0].item()
        score2 = model(**conv2_tokenized).logits[0][0].item()
        print(score1.shape)
        exit()

    feature_difference = score1 - score2
    print(f"Feature Difference: {feature_difference}")

