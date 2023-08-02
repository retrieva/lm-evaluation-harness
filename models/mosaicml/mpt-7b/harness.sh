export MODEL_ARGS="pretrained=mosaicml/mpt-7b,trust_remote_code=True,batch_size=1,load_in_8bit=True"
export TASK="jsquad-1.1-0.4,jcommonsenseqa-1.1-0.4,jnli-1.1-0.4,marc_ja-1.1-0.4"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot "2,3,3,3" \
    --device "cuda" \
    --output_path "results/mpt-result.json"
