import os
import json
import argparse
import tqdm
import time
import datetime
import math
import nltk
import numpy as np
import matplotlib.pyplot as plt

# TODO: Multithreading

nltk_downloader = nltk.downloader.Downloader()

if not nltk_downloader.is_installed('punkt_tab'):
	nltk_downloader.download('punkt_tab')

parser = argparse.ArgumentParser()
parser.add_argument('input_data', type=str)
parser.add_argument('--output_dir', type=str, default="analysis-" + str(round(time.time())) + "/")
parser.add_argument('--confidence_interval', type=float, default=95.0)

args = parser.parse_args()

confidence_bounds = [(100.0 - args.confidence_interval)/2, args.confidence_interval+((100.0 - args.confidence_interval)/2)]

def calculate_item_metrics(token_logprobs):
	text = ""
	probs = []
	logprob_sum = 0
	token_count = 0
	wrapped = False
	for token in token_logprobs:
		for key, value in token.items():
			if value:
				text += key
				logprob_sum += value
				token_count += 1
				if not wrapped:
					probs.append(math.exp(-value))
			elif len(probs) > 0:
				wrapped = True

	byte_count = max(len(text.encode("utf-8")), 1)
	word_count = max(len(nltk.tokenize.word_tokenize(text)), 1)
	token_count = max(token_count, 1)

	return (
		{
			"byte_count": byte_count,
			"word_count": word_count,
			"token_count": token_count,
			"byte_perplexity": math.exp(-logprob_sum / byte_count),
			"word_perplexity": math.exp(-logprob_sum / word_count),
			"token_perplexity": math.exp(-logprob_sum / token_count),
			"bits_per_byte": -logprob_sum / byte_count * 1 / math.log(2),
		},
		probs
	)

def calculate_task_data_metrics(items):
	byte_counts = []
	word_counts = []
	token_counts = []
	byte_perplexities = []
	word_perplexities = []
	token_perplexities = []
	bpbs = []
	bytes_per_token = []
	words_per_token = []
	bytes_per_word = []
	for item in items:
		byte_counts.append(item["byte_count"])
		word_counts.append(item["word_count"])
		token_counts.append(item["token_count"])
		byte_perplexities.append(item["byte_perplexity"])
		word_perplexities.append(item["word_perplexity"])
		token_perplexities.append(item["token_perplexity"])
		bpbs.append(item["bits_per_byte"])
		bytes_per_token.append(item["byte_count"] / item["token_count"])
		if item["word_count"] > 0:
			words_per_token.append(item["word_count"] / item["token_count"])
			bytes_per_word.append(item["byte_count"] / item["word_count"])

	return {
		"size": {
			"items": len(items),
			"bytes": int(np.sum(byte_counts)),
			"words": int(np.sum(word_counts)),
			"tokens": int(np.sum(token_counts)),
		},
		"item_statistics": {
			"sizes": {
				"bytes": calculate_task_element_metrics(byte_counts),
				"words": calculate_task_element_metrics(word_counts),
				"tokens": calculate_task_element_metrics(token_counts),
			},
			"tokenization": {
				"bytes_per_token": calculate_task_element_metrics(bytes_per_token),
				"words_per_token": calculate_task_element_metrics(words_per_token),
				"bytes_per_word": calculate_task_element_metrics(bytes_per_word),
			},
			"perplexities": {
				"byte_perplexity": calculate_task_element_metrics(byte_perplexities),
				"word_perplexity": calculate_task_element_metrics(word_perplexities),
				"token_perplexity": calculate_task_element_metrics(token_perplexities),
				"bits_per_byte": calculate_task_element_metrics(bpbs),
			}
		},
	}

def calculate_task_element_metrics(items):
	# TODO: Calculate histograms

	# TODO: Optimize performance

	percentiles = np.percentile(items, [confidence_bounds[0], 25, 50, 75, confidence_bounds[1]])

	#plt.figure()
	#plt.violinplot(items, showmedians=True)
	#plt.show()
	#plt.savefig('foo.png')

	return {
		"min": float(np.min(items)),
		"max": float(np.max(items)),
		"mean": float(np.mean(items)),
		"stdev": float(np.std(items)),
		"percentiles": {
			str(confidence_bounds[0]): percentiles[0],
			"25": percentiles[1],
			"50": percentiles[2],
			"75": percentiles[3],
			str(confidence_bounds[1]): percentiles[4],
		},
	}

def calculate_task_throughput_metrics(task_metrics):
	return {
		"throughput_statistics": {
			"seconds_per_task": task_metrics["duration"] / task_metrics["size"]["items"],
			"bytes_per_second": task_metrics["size"]["bytes"] / task_metrics["duration"],
			"words_per_second": task_metrics["size"]["words"] / task_metrics["duration"],
			"tokens_per_second": task_metrics["size"]["tokens"] / task_metrics["duration"],
		}
	}


def graph_task(task_name, items, prob_items):
	graph_task_tokenization(items, task_name, 32, "output/"+task_name+"-tokenization.png")
	graph_task_perplexity(items, task_name, 32, "output/"+task_name+"-perplexity.png")
	graph_task_length_perplexity(items, task_name, "output/"+task_name+"-length-perplexity.png")
	graph_task_positional_perplexity(prob_items, task_name, 50, "output/"+task_name+"-positional-perplexity.png")
	# TODO: plot all charts to one image

def graph_task_tokenization(items, task_name, bins, filename):
	bytes_per_token = []
	for item in items:
		bytes_per_token.append(max(item["byte_count"] / item["token_count"], 1))

	plt.figure()
	plt.suptitle(task_name+" bytes per token (n="+str(len(bytes_per_token))+")")
	plt.xlabel("UTF-8 Bytes / Token")
	plt.ylabel("Dataset Items")
	plt.hist(bytes_per_token, bins=bins)
	plt.xlim(xmin=1)
	plt.savefig(filename)
	plt.close()

def graph_task_perplexity(items, task_name, bins, filename):
	perplexities = []
	for item in items:
		perplexities.append(max(item["token_perplexity"], 1))

	plt.figure()
	plt.suptitle(task_name+" perplexity (n="+str(len(perplexities))+")")
	plt.xlabel("Token Perplexity")
	plt.ylabel("Dataset Items")
	plt.semilogx()
	hist, bins = np.histogram(perplexities, bins=bins)
	logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
	plt.hist(perplexities, bins=logbins)

	plt.xlim([1, 1000])
	plt.savefig(filename)
	plt.close()

def graph_task_length_perplexity(items, task_name, filename):
	lengths = []
	perplexities = []
	for item in items:
		lengths.append(item["token_count"])
		perplexities.append(max(item["token_perplexity"], 1))

	plt.figure()
	plt.suptitle(task_name+" perplexity by length (n="+str(len(perplexities))+")")
	plt.xlabel("Token Count")
	plt.ylabel("Token Perplexity")
	plt.loglog()
	plt.ylim([1, 1000])
	plt.scatter(lengths, perplexities, alpha=0.5)
	plt.savefig(filename)
	plt.close()

def graph_task_positional_perplexity(positional_probs, task_name, confidence_interval, filename):
	positional_probs=list(positional_probs.values())

	prob_median = []
	prob_lower_bound = []
	prob_upper_bound = []
	for prob_set in positional_probs:
		percentiles = np.percentile(prob_set, [(100.0-confidence_interval)/2, 50, confidence_interval+((100.0-confidence_interval)/2)])
		prob_lower_bound.append(max(percentiles[0], 1))
		prob_median.append(max(percentiles[1], 1))
		prob_upper_bound.append(max(percentiles[2], 1))

	items = len(positional_probs)

	plt.figure()
	plt.suptitle(task_name+" perplexity by position (n="+str(len(positional_probs[items-1]))+", "+str(confidence_interval)+"% CI)")
	plt.xlabel("Token Position")
	plt.ylabel("Token Perplexity")
	plt.semilogy()
	plt.xlim([0, items])
	plt.ylim([1, 1000])
	plt.plot(range(0, items), prob_median)
	plt.fill_between(range(0, items), prob_lower_bound, prob_upper_bound, alpha=0.2)
	plt.savefig(filename)
	plt.close()

def get_input_line_count(filename):
	with open(filename) as file:
		return sum(1 for line in file)

def process_input_data(filename):
	print("process_input_file", filename)
	line_count = get_input_line_count(filename)
	input_file = open(filename)

	metadata = {}
	task_metrics = {}
	tasks = {}
	task_positional_probs = {}

	for line in tqdm.tqdm(input_file, desc="analyze_file", total=line_count):
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
				graph_task(task_name, tasks[task_name], task_positional_probs[task_name])
				for key, value in calculate_task_data_metrics(tasks[task_name]).items():
					task_metrics[task_name][key] = value
				task_positional_probs[task_name] = {}
			elif task_name:
				task_metrics[task_name][key] = value
			elif isinstance(value, list):
				task_name = key
				if task_name not in tasks:
					tasks[task_name] = []
					task_positional_probs[task_name] = {}
				line_metrics = calculate_item_metrics(value)
				tasks[task_name].append(line_metrics[0])
				for i, prob in enumerate(line_metrics[1]):
					if i not in task_positional_probs[task_name]:
						task_positional_probs[task_name][i] = []
					task_positional_probs[task_name][i].append(prob)

	print("calculate_run_analysis")
	for key, value in task_metrics.items():
		task_name = key
		task_metrics[key]["started"] = task_metrics[key]["wallclock"]
		del task_metrics[key]["wallclock"]
		if "monotonic_ns" in task_metrics[key]:
			task_metrics[key]["duration"] = task_metrics[key]["monotonic_ns"] / 1000000000.0
			del task_metrics[key]["monotonic_ns"]
			for key, value in calculate_task_throughput_metrics(task_metrics[key]).items():
				task_metrics[task_name][key] = value
		elif not task_metrics[key]["completed"]:
			graph_task(task_name, tasks[task_name], task_positional_probs[task_name])
			for key, value in calculate_task_data_metrics(tasks[task_name]).items():
				task_metrics[task_name][key] = value
			task_positional_probs[task_name] = {}

	# TODO: Output and graphing

	print(json.dumps(task_metrics, indent="\t"))

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