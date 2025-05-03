import os
import json
import argparse
import tqdm
import time
import datetime
import math
import nltk
import numpy as np

nltk.download('punkt_tab')

parser = argparse.ArgumentParser()
parser.add_argument('input_data', type=str)
parser.add_argument('--output_dir', type=str, default="analysis-" + str(round(time.time())) + "/")

args = parser.parse_args()

def calculate_item_metrics(token_logprobs):
	text = ""
	logprobs = []
	for token in token_logprobs:
		for key, value in token.items():
			if value:
				logprobs.append(value)
				text += key

	byte_count = len(text.encode("utf-8"))
	word_count = len(nltk.tokenize.word_tokenize(text))
	token_count = len(logprobs)

	logprob_sum = np.sum(logprobs)

	return (
		{
			"byte_count": byte_count,
			"word_count": word_count,
			"token_count": token_count,
			"word_perplexity": math.exp(-logprob_sum / max(word_count, 1)),
			"byte_perplexity": math.exp(-logprob_sum / max(byte_count, 1)),
			"token_perplexity": math.exp(-np.mean(logprobs)),
			"bits_per_byte": -logprob_sum / max(byte_count, 1) * 1 / math.log(2),
		},
		logprobs
    )

def get_input_line_count(filename):
	with open(filename) as file:
		return sum(1 for line in file)

def process_input_data(filename):
	line_count = get_input_line_count(filename)
	input_file = open(filename)

	metadata = {}
	task_metrics = {}
	tasks = {}

	for line in tqdm.tqdm(input_file, desc="analyze_lines", total=line_count):
		line_data = json.loads(line)
		if len(metadata) == 0:
			metadata = line_data
			continue

		task_name = None
		for key, value in line_data.items():
			if key == "start_task":
				task_name = value
				task_metrics[value] = {"completed": False}
			elif key == "completed_task":
				task_name = value
				task_metrics[value]["completed"] = True
			elif task_name:
				task_metrics[task_name][key] = value
			elif isinstance(value, list):
				task_name = key
				if task_name not in tasks:
					tasks[task_name] = []
				#tasks[task_name].append(calculate_metrics(value)[0])
				#print(calculate_item_metrics(value, args.stride)[0])
				calculate_item_metrics(value) # FIXME

	input_file.close()

process_input_data(args.input_data)


# measurements to include in rewrite:
# - perplexity by position
# - perplexity / bits per byte by input length
# - perplexity / bits per byte by task
# - tokens/bytes per second by input length
# - tokens/bytes per second by task
# - bytes/token by task
# - words/token by task

# - perplexity / bits per byte by model
# - perplexity / bits per byte by task + model
# - perplexity by position + model
# - bytes/token by model
# - bytes/token by task + model
# - words/token by model
# - words/token by task + model

# create graphical plots, don't just output a handful of measurements
# - make data distributions visible
# - make statistical significance visible

"""
parser = argparse.ArgumentParser()
parser.add_argument('bench_data', nargs="+", type=str)
parser.add_argument('--data_output_file', type=str)
parser.add_argument('--stats_output_file', type=str)
parser.add_argument('--tokenizer', type=str, default="allenai/OLMo-2-0425-1B")

args = parser.parse_args()

tokenizer = Tokenizer.from_pretrained(args.tokenizer)

nltk.download('punkt_tab')

def calculate_metrics(token_logprobs):
	token_count = 0
	text = ""
	logprob_sum = 0
	for token in token_logprobs:
		for key, value in token.items():
			if value:
				token_count += 1
				text += key
				logprob_sum += value
	byte_count = len(text.encode("utf-8"))
	word_count = len(nltk.tokenize.word_tokenize(text))

	return {
		"byte_count": byte_count,
		"word_count": word_count,
		"token_count": token_count,
		"word_perplexity": math.exp(-logprob_sum / max(word_count, 1)),
		"byte_perplexity": math.exp(-logprob_sum / max(byte_count, 1)),
		"token_perplexity": math.exp(-logprob_sum / max(token_count, 1)),
		"normalized_token_perplexity": math.exp(-logprob_sum / max(len(tokenizer.encode(text)), 1)),
		"bits_per_byte": -logprob_sum / max(byte_count, 1) * 1 / math.log(2),
	}

def calculate_task_metrics(items):
	task_items = {}

	for item in items:
		for key, value in item.items():
			if key not in task_items:
				task_items[key] = []
			task_items[key].append(value)

	metrics = {}

	for key, value in task_items.items():
		quartiles = numpy.percentile(value, [1, 5, 10, 25, 50, 75, 90, 95, 99])
		histogram = numpy.histogram(value)

		metrics[key] = {
			"n": len(items),
			"mean": float(numpy.mean(value)),
			"std": float(numpy.std(value)),
			"percentiles": {
				"1": float(quartiles[0]),
				"5": float(quartiles[1]),
				"10": float(quartiles[2]),
				"25": float(quartiles[3]),
				"50": float(quartiles[4]),
				"75": float(quartiles[5]),
				"90": float(quartiles[6]),
				"95": float(quartiles[7]),
				"99": float(quartiles[8]),
			}
		}

	return metrics

for filename in args.bench_data:
	file = open(filename)
	metadata = {}
	tasks = {}
	task_data = {}

	for line in file:
		linedata = json.loads(line)
		if len(metadata) == 0:
			for key, value in linedata.items():
				metadata[key] = value
		task = None
		is_task_end = False
		for key, value in linedata.items():
			if key == "start_task":
				task = value
				task_data[value] = {"total_tokens": 0, "total_bytes": 0, "completed": False}
			elif key == "completed_task":
				task = value
				task_data[value]["completed"] = True
			elif task:
				task_data[task][key] = value
			if isinstance(value, list):
				task = key
				if task not in tasks:
					tasks[task] = []
				metrics = calculate_metrics(value)
				tasks[task].append(metrics)
				task_data[task]["total_bytes"] += metrics["byte_count"]
				task_data[task]["total_tokens"] += metrics["token_count"]

	file.close()

	if args.data_output_file:
		with open(args.data_output_file, "x") as file:
			json.dump({"metadata": metadata, "task_data": task_data, "tasks": tasks}, file, separators=(',', ':'))

	summary = {}

	for key, value in tasks.items():
		summary[key] = calculate_task_metrics(value)

	print(json.dumps(summary, indent="\t"))
"""