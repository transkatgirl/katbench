# katbench
what i use to compare LLM base models using dataset perplexities

## setup

you will need:

- a python3 venv with the dependencies specified in `requirements.txt`
- a working [TGI server](https://huggingface.co/docs/text-generation-inference/en/index) run with the `--trust-remote-code --enable-prefill-logprobs --payload-limit 100000000 --max-client-batch-size 16` flags

## usage

### bench.py

performs dataset tokenization & logprob calculation using remote LLM server, saving results to disk

```bench.py [-h] [--api_key API_KEY] [--model MODEL] [--task_file TASK_FILE] [--output_file OUTPUT_FILE] [--context_len CONTEXT_LEN] base_url```

### analyze.py

analyzes data saved to disk by `bench.py` and calculates metrics

TODO