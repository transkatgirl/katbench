# katbench
what i use to compare LLM base models using dataset perplexities

## setup

you will need:

- a python3 venv with the dependencies specified in `requirements.txt`
- a working [TGI server](https://huggingface.co/docs/text-generation-inference/en/index) run with the `--enable-prefill-logprobs --max-client-batch-size 16` flags

## usage

### bench.py

performs dataset tokenization & logprob calculation using remote LLM server, saving results to disk

```bash
python bench.py [-h] [--api_key API_KEY] [--model MODEL] [--task_file TASK_FILE] [--output_file OUTPUT_FILE] [--context_len CONTEXT_LEN] [--payload_limit PAYLOAD_LIMIT] base_url
```

note: if you plan on testing very long context lengths (>100k tokens), you will need to use the `--payload-limit 100000000` flag on the TGI server, and the `--payload_limit 100000000` flag on the `bench.py` script

### collate.py

combines multiple incomplete runs saved by `bench.py` into one output file, removing unfinished tasks and trimming corrupted JSON

```bash
python collate.py [-h] [--output_file OUTPUT_FILE] input_files [input_files ...]
```

### analyze.py

analyzes data saved to disk by `bench.py` and calculates metrics

WIP

### compare.py

analyzes multiple outputs from `analyze.py` and creates comparisons

WIP