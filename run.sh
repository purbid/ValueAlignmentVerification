#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks=64
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH -o outputs-%j

export PYTHONPATH=/scratch/general/vast/u1419542/miniconda3/envs/valueAlignmentVerification/bin/python
source /scratch/general/vast/u1419542/miniconda3/etc/profile.d/conda.sh
conda activate valueAlignmentVerification

# wandb disabled 
# mkdir /scratch/general/vast/u1419542/huggingface_cache
export HF_HOME="/uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/huggingface_cache"

DATASET="Owishiboo/grammar-correction"
NOISE=0.0
HF_TOKEN=""
while getopts "d:n:t:" opt
do 
    case "$opt" in 
        d ) DATASET="$OPTARG" ;;
        n ) NOISE="$OPTARG" ;;
        t ) HF_TOKEN="$OPTARG" ;;
    esac 
done
# python3 removeRedundancy.py \
#     -info \
#     -pad_to_max_length \
#     -data ./data/dev_feedback.json \
#     -loadFeats

# python3 removeRedundancy.py \
#     -info \
#     -pad_to_max_length \
#     -data ./data/dev_feedback.json \
#     -loadFeats \
#     -addEpsilon

# python3 removeRedundancy.py \
#     -info \
#     -pad_to_max_length \
#     -data ./proData/F-ERR_sentence/dev.json \
#     -loadFeats \
#     -factuality

# python3 removeRedundancy.py \
#     -info \
#     -pad_to_max_length \
#     -data ./proData/F-ERR_sentence/dev.json \
#     -loadFeats \
#     -factuality \
#     -addEpsilon

# python3 train.py -debug -dataset Owishiboo/grammar-correction

# python3 removeRedundancy.py \
#     -info \
#     -pad_to_max_length \
#     -max_seq_length 1024 \
#     -data ./models/val.json \
#     -model /uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/ValueAlignmentVerification/models/preference_model.pt \
#     -grammaticality \
#     -addEpsilon \
#     -loadFeats 

# python3 removeRedundancy.py \
#     -info \
#     -pad_to_max_length \
#     -max_seq_length 1024 \
#     -data ./models/val.json \
#     -model /uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/ValueAlignmentVerification/models/preference_model.pt \
#     -grammaticality \
#     -loadFeats

# python3 removeRedundancy.py \
#     -info \
#     -pad_to_max_length \
#     -max_seq_length 1024 \
#     -data ./models/JFLEG/val.json \
#     -model /uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/ValueAlignmentVerification/models/preference_model.pt \
#     -fluency \
#     -loadFeats \
#     -seed 3053283 \
#     -shuffle

# python3 train.py -debug -introspect ./models/preference_model.pt

# python3 train.py -debug -dataset "jhu-clsp/jfleg" -saveModelPath ./models/JFLEG/

# python3 test.py \
#     -info \
#     -data /uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/ValueAlignmentVerification/jfleg/train_dataset_grammaticality.pkl \
#     -indsDir /uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/ValueAlignmentVerification/jfleg/ \
#     -fileAddendum _grammaticality \
#     -model "openai-community/gpt2" \
#     -input_column sentence \
#     -target_column corrections \
#     -maxLength 512

# python3 test.py \
#     -info \
#     -data /uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/ValueAlignmentVerification/train_dataset_fluency.pkl \
#     -indsDir /uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/ValueAlignmentVerification/ \
#     -fileAddendum _fluency \
#     -model "google/flan-t5-base" \
#     -input_column sentence \
#     -target_column corrections \

# python3 test.py \
#     -info \
#     -data /uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/ValueAlignmentVerification/train_dataset_fluency.pkl \
#     -indsDir /uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/ValueAlignmentVerification/ \
#     -fileAddendum _fluency \
#     -model "meta-llama/Llama-2-7b-hf" \
#     -input_column sentence \
#     -target_column corrections \
#     -hf_token $HF_TOKEN \

# python3 monteCarloTest.py \
#     -info \
#     -features /uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/ValueAlignmentVerification/features_fluency.pkl \
#     -indsDir /uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/ValueAlignmentVerification/ \
#     -fileAddendum _fluency \
#     -seed 3253432 \
#     -randomSeeds 7 26 43613 82483 3053283

# python3 train.py -debug -dataset ${DATASET} -saveModelPath ./models_${NOISE}/ -noise ${NOISE}

# python3 train.py -debug -introspect ./models_${NOISE}/preference_model.pt -dataset ${DATASET}

python3 train.py \
    -debug \
    -introspect ./models_${NOISE}/preference_model.pt \
    -dataset ${DATASET} \
    -customData ./train_dataset_fluency.pkl \
    -customRedundant ./redundant_fluency_3053283.pkl \
    -customNonRedundant ./non_redundant_fluency_3053283.pkl 