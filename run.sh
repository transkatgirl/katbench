lighteval accelerate \
	"pretrained=$1,dtype=bfloat16,model_parallel=True" \
	./katbench.txt \
	--custom-tasks katbench.py \
	--override-batch-size 1 \
	--output-dir output
