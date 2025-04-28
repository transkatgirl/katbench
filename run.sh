lighteval accelerate \
	"pretrained=meta-llama/Llama-3.2-1B,dtype=bfloat16,model_parallel=True" \
	"community|pile_10k|0|1" \
	--custom-tasks katbench.py \
	--override-batch-size 1
