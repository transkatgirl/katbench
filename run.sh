if ! [ -e "model.yaml" ] ; then
	echo "model:
    inference_server_address: "http://your-inference-server.example.com"
    inference_server_auth: null
    model_name: null" > model.yaml
fi
lighteval tasks list --custom-tasks katbench.py > /dev/null
HF_HOME=cache lighteval endpoint tgi \
	model.yaml \
	./tasks.txt \
	--custom-tasks katbench.py \
	--output-dir output
