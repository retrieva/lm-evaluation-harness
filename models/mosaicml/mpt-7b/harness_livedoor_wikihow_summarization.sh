export MODEL_ARGS="pretrained=mosaicml/mpt-7b,trust_remote_code=True,batch_size=1,load_in_8bit=True"
export TASK="livedoor_summarization-1.0-0.0,wikihow_summarization-1.0-0.0"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot "0,0" \
    --device "cuda" \
    --output_path "results/mpt-result-sum.json"
