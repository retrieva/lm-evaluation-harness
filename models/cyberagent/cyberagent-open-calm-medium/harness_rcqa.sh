export MODEL_ARGS="pretrained=cyberagent/open-calm-medium,trust_remote_code=True,batch_size=4"
export TASK="rcqa-1.0-0.2"
export CUDA_VISIBLE_DEVICES=1
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot "1" \
    --device "cuda" \
    --output_path "results/cyberagent-result-rcqa.json"
