import os
import json
import argparse
import asyncio
import tqdm
import time
import datetime
from datasets import load_dataset
from huggingface_hub import AsyncInferenceClient

parser = argparse.ArgumentParser()
parser.add_argument('base_url')
parser.add_argument('--api_key')
parser.add_argument('--model')
parser.add_argument('--task_file')
parser.add_argument('--output_file')
parser.add_argument('--context_len')

args = parser.parse_args()

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
	return hydrated_tasks

async def run_task(semaphore, client, prompt, truncate):
	async with semaphore:
		return await _run_task_loop(client, prompt, truncate)

async def _run_task_loop(client, prompt, truncate):
	output = await client.text_generation(prompt=prompt, stream=False, details=True, decoder_input_details=True, do_sample=False, watermark=False, truncate=truncate, max_new_tokens=1)
	characters = 0
	for token in output.details.prefill[1:]:
		characters += len(token.text)
	if len(prompt) > characters:
		split_output = await _run_task_loop(client, prompt[characters:], truncate)
		return [*output.details.prefill, *split_output]
	else:
		return output.details.prefill

def convert_output_format(output):
	tokens = []
	for token in output:
		tokens.append({token.text: token.logprob})
	return tokens

async def main():
	taskfilename = args.task_file or "tasks.json"
	outputfilename = args.output_file or "output-" + str(round(time.time())) + ".jsonl"

	print("load_tasks", taskfilename)
	raw_tasks = load_raw_tasks(taskfilename)
	tasks = hydrate_tasks(raw_tasks)

	client = AsyncInferenceClient(base_url=args.base_url, api_key=args.api_key, model=args.model)

	print("get_endpoint_info", args.base_url)
	info = await client.get_endpoint_info()
	max_input = min(int(args.context_len or info["max_input_tokens"]), int(info["max_input_tokens"]))
	batch_size = int(info["max_client_batch_size"])

	semaphore = asyncio.Semaphore(batch_size)
	print("model="+info["model_id"]+", context_len="+str(max_input)+", batch_size="+str(batch_size))

	print("open_output_file", outputfilename)
	outputfile = open(outputfilename, "x")

	outputfile.write(json.dumps({"tasks": raw_tasks, "endpoint_info": info, "effective_context_len": max_input}, separators=(',', ':')))
	outputfile.write("\n")
	outputfile.flush()

	for name, dataset in tqdm.tqdm(tasks.items(), desc="run_tasks", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"):
		field = dataset["field"]
		outputfile.write(json.dumps({"start_task": name, "wallclock": datetime.datetime.now().astimezone().isoformat()}, separators=(',', ':')))
		outputfile.write("\n")
		start = time.perf_counter_ns()
		for result in tqdm.asyncio.tqdm.as_completed([run_task(semaphore, client, item[field], max_input) for item in dataset["dataset"]], desc=name):
			outputfile.write(json.dumps({name: convert_output_format(await result)}, separators=(',', ':')))
			outputfile.write("\n")
		outputfile.write(json.dumps({"completed_task": name, "monotonic_ns": time.perf_counter_ns() - start}, separators=(',', ':')))
		outputfile.write("\n")

	print("flush_output_file", outputfilename)
	outputfile.flush()
	os.fsync(outputfile)
	outputfile.close()

if __name__ == '__main__':
	asyncio.run(main())
