export MODEL_ARGS="pretrained=cyberagent/open-calm-medium,trust_remote_code=True,batch_size=4"
export TASK="jsnli-1.0-0.2,ebe_identification-1.0-0.2,livedoor_classification-1.0-0.2,pc_customer_demo-1.0-0.2,tmup-1.0-0.0"
export TASK="jsnli-1.1-0.2"
export CUDA_VISIBLE_DEVICES=1
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot "1" \
    --device "cuda" \
    --output_path "results/cyberagent-result-jsnli.json"
