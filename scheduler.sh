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

# python3 model_test.py --filter_by_subset "chat_hard" --shorten_size  768
python3 sanity_check_on_features.py --filter_by_subset "chat_hard" --shorten_size  768

####when training the models
#python3 ~/ValueAlignmentVerification/train.py -dataset Owishiboo/grammar-correction -noise 0.5


#####when testing the model
#python3 train_copy.py -dataset Owishiboo/grammar-correction -introspect ./models/preference_model.pt -customData ./models/val.json -customRedundant redundant_grammaticality.pkl -customNonRedundant non_redundant_grammaticality.pkl

#testing the model on inflated sets
#python3 test_on_inflated.py -dataset Owishiboo/grammar-correction -introspect ./models/preference_model.pt -customData ./models/val.json
#python3  test_on_inflated.py -dataset Owishiboo/grammar-correction -introspect ./models/preference_model_50_noise.pt  -customData ./create_data/zero_shot_real_redundant/non_redundant_1.csv




#when the model is trained, and we have the features, just need redunddant pairs now
#python3 rewardBenchRedundancy.py -loadFeats -log curr_file.log  




#when we need features for the first time
#python3 removeRedundancy.py -grammaticality -model ./models_from_local/preference_model.pt -data ./models_from_local/val.json



#python3 ~/ValueAlignmentVerification/removeRedJaxOpt.py -grammaticality -log curr_file.log 
