import os
import json
import argparse
import asyncio
import tqdm
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
			for subset in value["hf_subsets"]:
				tasks[key+":"+subset] = {"repo": value["hf_repo"], "subset": subset, "split": value["evaluation_split"]}
			if len(value["hf_subsets"]) == 0:
				tasks[key] = {"repo": value["hf_repo"], "subset": "default", "split": value["evaluation_split"]}
	return tasks

def hydrate_tasks(tasks):
	hydrated_tasks = {}
	for key, value in tqdm.tqdm(tasks.items(), desc="load_datasets"):
		hydrated_tasks[key] = load_dataset(value["repo"], value["subset"], split=value["split"], num_proc=8)
	return hydrated_tasks

async def run_task(semaphore, client, input, truncate):
	async with semaphore:
		output = await client.text_generation(prompt=input["text"], stream=False, details=True, decoder_input_details=True, do_sample=False, watermark=False, truncate=truncate, max_new_tokens=1)
		return output.details.prefill

async def main():
	taskfilename = args.task_file or "tasks.json"
	outputfilename = args.output_file or "output.jsonl"

	print("open_output_file", outputfilename)
	if os.path.exists(outputfilename):
		os.remove(outputfilename)
	outputfile = open(outputfilename, "x")

	print("load_tasks", taskfilename)
	raw_tasks = load_raw_tasks(taskfilename)
	tasks = hydrate_tasks(raw_tasks)

	client = AsyncInferenceClient(base_url=args.base_url, api_key=args.api_key, model=args.model)

	print("get_endpoint_info", args.base_url)
	info = await client.get_endpoint_info()
	max_input = args.context_len or info["max_input_tokens"]

	print(info)

	outputfile.write(json.dumps({"tasks": raw_tasks, "endpoint_info": info, "max_input": max_input}))
	outputfile.write("\n")
	outputfile.flush()

	semaphore = asyncio.Semaphore(info["max_concurrent_requests"]/4)

	for name, dataset in tasks.items():
		print("run_task", name, info["model_id"], "context_len="+str(max_input))
		#task_results = []

		for result in tqdm.asyncio.tqdm.as_completed([run_task(semaphore, client, item, max_input) for item in dataset]):
			#task_results.append(await result)
			outputfile.write(json.dumps(await result))
			outputfile.write("\n")

		#outputfile.write(json.dumps({name: task_results}))
		#outputfile.write("\n")
		outputfile.flush()


if __name__ == '__main__':
	asyncio.run(main())
