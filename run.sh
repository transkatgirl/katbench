lighteval tasks list --custom-tasks katbench.py > /dev/null
HF_HOME=cache lighteval accelerate \
	"pretrained=$1,dtype=bfloat16,model_parallel=True" \
	./tasks.txt \
	--custom-tasks katbench.py \
	--override-batch-size 1 \
	--output-dir output
