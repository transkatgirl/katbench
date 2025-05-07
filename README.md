# katbench
what i use to compare LLM base models using dataset perplexities

## setup

you will need:

- a python3 venv with the dependencies specified in `requirements.txt`
- a working [Text Generation Inference (TGI) server](https://huggingface.co/docs/text-generation-inference/en/index) run with the `--enable-prefill-logprobs --max-client-batch-size 32` flags

## usage

### bench.py

performs dataset tokenization & logprob calculation using TGI server, saving tokens + logprobs to disk

```
usage: bench.py [-h] [--api_key API_KEY] [--model MODEL] [--task_file TASK_FILE] [--output_file OUTPUT_FILE] [--context_len CONTEXT_LEN]
                [--payload_limit PAYLOAD_LIMIT] [--depth_limit DEPTH_LIMIT] [--request_timeout REQUEST_TIMEOUT] [--retry_timeout RETRY_TIMEOUT]
                base_url
```

note: if you plan on testing very long context lengths (>100k tokens), you will need to use the `--payload-limit 100000000` flag on the TGI server, and the `--payload_limit 100000000 --request_timeout 7200` flag on the `bench.py` script

### collate.py

combines multiple incomplete runs saved by `bench.py` into one output file, removing unfinished tasks, duplicate tasks, and trimming corrupted JSON

```
usage: collate.py [-h] [--output_file OUTPUT_FILE] input_files [input_files ...]
```

### analyze.py (WIP)

analyzes data saved to disk by `bench.py` and generates data analyses

```
usage: analyze.py [-h] [--output_dir OUTPUT_DIR] [--skip_slow_analyses] input_files [input_files ...]
```