#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=20:00:00
#$ -j y
#$ -o logs/
#$ -cwd 


cd ${HOME}/retrieva/work/LLM-Research-Benchmark/
proj_data_dir=/groups/gcd50709/LLM/
export HF_DATASETS_CACHE=${proj_data_dir}/datasets
export TRANSFORMERS_CACHE=${proj_data_dir}/models
source env_module_abci.sh
source .venv/bin/activate

cd lm-evaluation-harness

model_name_or_path=$1
peft_model_path=$2
TASK=$3
result_path_name=$4

python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$model_name_or_path,peft=$peft_model_path \
    --tasks $TASK \
    --num_fewshot "0" \
    --device "cuda" \
    --output_path $result_path_name
