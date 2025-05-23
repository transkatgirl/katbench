import os
import json
import argparse
import asyncio
import tqdm
import time
import datetime
import sys
from datasets import load_dataset
from huggingface_hub import AsyncInferenceClient
from tenacity import retry, wait_random_exponential, stop_after_attempt

# TODO: switch inference backend from TGI to vLLM, implement saving vLLM prompt_logprobs

parser = argparse.ArgumentParser()
parser.add_argument('base_url', type=str)
parser.add_argument('--api_key', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--task_file', type=str, default="tasks.json")
parser.add_argument('--output_file', type=str, default="output-" + str(round(time.time())) + ".jsonl")
parser.add_argument('--context_len', type=int)
parser.add_argument('--payload_limit', type=int, default=2000000)
parser.add_argument('--depth_limit', type=int, default=500)
parser.add_argument('--request_timeout', type=int, default=60*30)
parser.add_argument('--max_retries', type=int, default=12)

args = parser.parse_args()

sys.setrecursionlimit(max((args.depth_limit+500), 1000))

def load_raw_tasks(filename):
	tasks = {}
	with open(filename) as file:
		data = json.load(file)
		for key, value in data.items():
			field = value.get("field") or "text"
			for subset in value["subsets"]:
				tasks[key+":"+subset] = {"repo": value["hf_repo"], "subset": subset, "split": value["split"], "field": field}
			if len(value["subsets"]) == 0:
				tasks[key] = {"repo": value["hf_repo"], "subset": "default", "split": value["split"], "field": field}
	return tasks

def hydrate_tasks(tasks):
	hydrated_tasks = {}
	for key, value in tqdm.tqdm(tasks.items(), desc="load_datasets"):
		hydrated_tasks[key] = {"dataset": load_dataset(value["repo"], value["subset"], split=value["split"]), "field": value["field"]}
		assert len(hydrated_tasks[key]["dataset"][1][hydrated_tasks[key]["field"]]) > 0
	return hydrated_tasks

async def run_task(semaphore, payload_limit, depth_limit, client, prompt, truncate):
	async with semaphore:
		return await _run_task_loop(payload_limit, depth_limit, client, prompt, truncate)

async def _run_task_loop(payload_limit, depth_limit, client, prompt, truncate, depth = 0):
	output = await _run_task_request(client, prompt[:payload_limit], truncate)
	characters = 0
	for token in output.details.prefill[1:]:
		characters += len(token.text)
	if len(prompt) > characters and depth < depth_limit:
		split_output = await _run_task_loop(payload_limit, depth_limit, client, prompt[characters:], truncate, depth + 1)
		return [*output.details.prefill, *split_output]
	else:
		return output.details.prefill

@retry(wait=wait_random_exponential(multiplier=1,max=60), stop=stop_after_attempt(args.max_retries))
async def _run_task_request(client, prompt, truncate):
	return await client.text_generation(prompt=prompt, stream=False, details=True, decoder_input_details=True, do_sample=False, watermark=False, truncate=truncate, max_new_tokens=1)

def convert_output_format(output):
	tokens = []
	for token in output:
		tokens.append({token.text: token.logprob})
	return tokens

async def main():
	print("load_tasks", args.task_file)
	raw_tasks = load_raw_tasks(args.task_file)
	tasks = hydrate_tasks(raw_tasks)

	client = AsyncInferenceClient(base_url=args.base_url, api_key=args.api_key, model=args.model, timeout=args.request_timeout)

	print("get_endpoint_info", args.base_url)
	info = await client.get_endpoint_info()
	max_input = min(int(args.context_len or info["max_input_tokens"]), int(info["max_input_tokens"]))
	batch_size = int(info["max_client_batch_size"])
	payload_limit = args.payload_limit - 100000

	semaphore = asyncio.Semaphore(batch_size)
	print("model="+info["model_id"]+", context_len="+str(max_input)+", batch_size="+str(batch_size)+", payload_limit="+str(payload_limit)+", depth_limit="+str(args.depth_limit))

	print("open_output_file", args.output_file)
	output_file = open(args.output_file, "x")

	output_file.write(json.dumps({"tasks": raw_tasks, "endpoint_info": info, "effective_context_len": max_input, "payload_limit": payload_limit, "depth_limit": args.depth_limit}, separators=(',', ':')))
	output_file.write("\n")
	output_file.flush()

	for name, dataset in tqdm.tqdm(tasks.items(), desc="run_tasks", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"):
		field = dataset["field"]
		output_file.write(json.dumps({"start_task": name, "wallclock": datetime.datetime.now().astimezone().isoformat()}, separators=(',', ':')))
		output_file.write("\n")
		start = time.perf_counter_ns()
		for result in tqdm.asyncio.tqdm.as_completed([run_task(semaphore, payload_limit, args.depth_limit, client, item[field], max_input) for item in dataset["dataset"] if item[field]], desc=name):
			output_file.write(json.dumps({name: convert_output_format(await result)}, separators=(',', ':')))
			output_file.write("\n")
		output_file.write(json.dumps({"completed_task": name, "monotonic_ns": time.perf_counter_ns() - start}, separators=(',', ':')))
		output_file.write("\n")
		output_file.flush()
		os.fsync(output_file)

	print("close_output_file", args.output_file)
	output_file.close()

if __name__ == '__main__':
	asyncio.run(main())
