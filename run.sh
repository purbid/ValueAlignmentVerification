#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --qos=marasovic-gpulong-np
#SBATCH --ntasks=64
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=36:00:00
#SBATCH --mem=128GB
#SBATCH -o outputs-%j

export PYTHONPATH=/scratch/general/vast/u1419542/miniconda3/envs/valueAlignmentVerification/bin/python
source /scratch/general/vast/u1419542/miniconda3/etc/profile.d/conda.sh
conda activate valueAlignmentVerification

# wandb disabled 
# mkdir /scratch/general/vast/u1419542/huggingface_cache
export HF_HOME="/uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/huggingface_cache"

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

# python3 train.py -debug

python3 removeRedundancy.py \
    -info \
    -pad_to_max_length \
    -max_seq_length 1024 \
    -data ./models/val.json \
    -model /uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/ValueAlignmentVerification/models/preference_model.pt \
    -grammaticality \
    -addEpsilon \
    -loadFeats 

# python3 removeRedundancy.py \
#     -info \
#     -pad_to_max_length \
#     -max_seq_length 1024 \
#     -data ./models/val.json \
#     -model /uufs/chpc.utah.edu/common/home/u1419542/scratch/DONT_DELETE/ValueAlignmentVerification/models/preference_model.pt \
#     -grammaticality \
#     -loadFeats

# python3 train.py -debug -introspect ./models/preference_model.pt

# python3 train.py -debug -dataset "jhu-clsp/jfleg"