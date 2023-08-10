#!/bin/bash

models=(cyberagent/open-calm-1b)
tasks=(jnli-1.1-0.2 jnli-1.1-0.3 jnli-1.1-0.4)
result_dir="/groups/gcd50709/LLM/eval/"

fewshots=(2 3 4)
for model in ${models[@]};
do
  for task in ${tasks[@]};
  do
    for fewshot in ${fewshots[@]};
    do
      result_path=${result_dir}/${model}/${fewshot}  
      # fill in few-shot evaluation
     done
  done
done

peft_algos=(PromptTuning Lora)
peft_model_root="/groups/gcd50709/LLM/peft/"

for model in ${models[@]};
do
  for peft_algo in ${peft_algos[@]};
  do
    for task in ${tasks[@]};
    do
      task_info=(${task//"-"/ })
      dataset=${task_info[0]}
      target=${dataset}/${task}/${model}/${peft_algo}
      peft_model_algo=${peft_model_root}/${target}
      params=`ls ${peft_model_algo}` 
      for param in ${params[@]};
      do
	peft_model_path=${peft_model_algo}/${param}
	result_peft_dir=${result_dir}/peft/${target}/${param}
	mkdir -p $result_peft_dir
	result_peft_path=${result_peft_dir}/result.json
	qsub -g gcd50709 peft_eval.sh $model $peft_model_path $task $result_peft_path
      done
    done
  done
done
