#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --time=2:00:00
#SBATCH --mem=80GB
#SBATCH --mail-user=bambroopurbid@gmail.com
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_1-%j
#SBATCH --export=ALL

#source ~/ekfac/EKFAC-Influence-Benchmarks/venv/bin/activate
source ~/ValueAlignmentVerificationPurbid/vavenv/bin/activate

# python3 model_test.py --filter_by_subset "chat_hard" --shorten_size  512


# python3 rewardBenchRedundancy.py -loadFeats -log curr_file.log -featuresPath features/Skywork-Reward-Llama-3.1-8B-v0.2/features_diff_full_length_chat_hard.pkl
# python3 rewardBenchRedundancy.py -log curr_file.log  -grammaticality

python3 removeRedundancy.py -log curr_file.log  -grammaticality


# python3 reduce_dimensions.py --filter_by_subset "chat_hard" --target_dim_svd 1024 --reduction_technique "svd"
# python3 sanity_check_on_features.py --filter_by_subset "chat_hard" --shorten_size  768


# ### Fine tune Llama 3.1 using PEFT and LORA
# python peft_wrapped_llama.py \
#     --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
#     --dataset_name Skywork/Skywork-Reward-Preference-80K-v0.2 \
#     --output_dir  /scratch/general/vast/u1472659/ \
#     --per_device_train_batch_size 2 \
#     --num_train_epochs 2 \
#     --learning_rate 2e-6 \
#     --logging_steps 25 \
#     --eval_strategy steps \
#     --eval_steps 50 \
#     --max_length 2048 \
#     --use_peft \
#     --lora_r 32 \
#     --lora_task_type SEQ_CLS \
#     --lora_alpha 16 \
#     --gradient_checkpointing True \




### eval on reward bench
# rewardbench --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2
# rewardbench --model meta-llama/Llama-3.1-8B-Instruct
# rewardbench  --model /scratch/general/vast/u1472659/lora_llama_ft/merged_model
# python reward-bench/scripts/run_rm.py --model /scratch/general/vast/u1472659/lora_llama_ft/Skywork-Reward-Llama-3.1-8B-v0.2_BT_RM_len512_lora32_1e-05_dataSkywork-Reward-Preference-80K-v0.2/
# rewardbench --model Skywork/Skywork-Reward-Llama-3.1-8B-v0.2

####when training the models
# python3 ~/ValueAlignmentVerification/train.py -dataset Owishiboo/grammar-correction -noise 0.5


#####when testing the model
#python3 train_copy.py -dataset Owishiboo/grammar-correction -introspect ./models/preference_model.pt -customData ./models/val.json -customRedundant redundant_grammaticality.pkl -customNonRedundant non_redundant_grammaticality.pkl

#testing the model on inflated sets
#python3 test_on_inflated.py -dataset Owishiboo/grammar-correction -introspect ./models/preference_model.pt -customData ./models/val.json
#python3  test_on_inflated.py -dataset Owishiboo/grammar-correction -introspect ./models/preference_model_50_noise.pt  -customData ./create_data/zero_shot_real_redundant/non_redundant_1.csv




#when the model is trained, and we have the features, just need redunddant pairs now
# python3 rewardBenchRedundancy.py -loadFeats -log curr_file.log -featuresPath features/Skywork-Reward-Llama-3.1-8B/pca_features_diff_400_chat_hard.pkl


#when we need features for the first time
#python3 removeRedundancy.py -grammaticality -model ./models_from_local/preference_model.pt -data ./models_from_local/val.json



#python3 ~/ValueAlignmentVerification/removeRedJaxOpt.py -grammaticality -log curr_file.log 
