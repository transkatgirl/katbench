import os
import json
import argparse
import tqdm
import time
import datetime
import math
import shutil
import nltk
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: Multithreading, CSV output, performance optimization
# TODO: Add flag to skip task-specific graphs

sns.set_theme()
mpl.rcParams['figure.dpi'] = 300
nltk_downloader = nltk.downloader.Downloader()

if not nltk_downloader.is_installed('punkt_tab'):
	nltk_downloader.download('punkt_tab')

parser = argparse.ArgumentParser()
parser.add_argument('input_files', nargs="+", type=str)
parser.add_argument('--output_dir', type=str, default="analysis-" + str(round(time.time())) + "/")

args = parser.parse_args()

if os.path.exists(args.output_dir):
	if os.path.isdir(args.output_dir):
		shutil.rmtree(args.output_dir)
	else:
		os.remove(args.output_dir)
os.makedirs(args.output_dir)

def calculate_item_metrics(token_logprobs):
	text = ""
	probs = []
	logprob_sum = 0.0
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
			"byte_perplexity": np.exp(-logprob_sum / byte_count),
			"word_perplexity": np.exp(-logprob_sum / word_count),
			"token_perplexity": np.exp(-logprob_sum / token_count),
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

	return (
		{
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
		},
		{
			"token_perplexity": token_perplexities,
			"bits_per_byte": bpbs,
			"byte_counts": byte_counts,
			"token_counts": token_counts,
			"bytes_per_token": bytes_per_token,
		}
	)

def calculate_task_element_metrics(items):
	percentiles = np.percentile(items, [5, 25, 50, 75, 95])

	return {
		"min": float(np.min(items)),
		"max": float(np.max(items)),
		"mean": float(np.mean(items)),
		"stdev": float(np.std(items)),
		"percentiles": {
			"5": percentiles[0],
			"25": percentiles[1],
			"50": percentiles[2],
			"75": percentiles[3],
			"95": percentiles[4],
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

def write_json(data, path):
	if os.path.exists(path):
		os.remove(path)
	output_file = open(path, "x")
	json.dump(data, output_file, indent="\t")
	output_file.close()

def graph_task(output_prefix, task_name, items, prob_items, incomplete):
	if incomplete:
		task_name += "~incomplete"
	output_prefix = os.path.join(output_prefix, "tasks/"+task_name)
	os.makedirs(output_prefix)
	graph_task_perplexity(items, task_name, os.path.join(output_prefix, "perplexity.png"))
	graph_task_bpb(items, task_name, os.path.join(output_prefix, "bits-per-byte.png"))
	graph_task_bpb_perplexity(items, task_name, os.path.join(output_prefix, "perplexity-bits-per-byte.png"))
	graph_task_length_perplexity(items, task_name, os.path.join(output_prefix, "perplexity-length.png"))
	graph_task_tokenization_perplexity(items, task_name, os.path.join(output_prefix, "perplexity-tokenization.png"))
	graph_task_positional_perplexity(prob_items, task_name, os.path.join(output_prefix, "perplexity-positional.png"))

def graph_tasks(output_prefix, comparative_data, model_name):
	graph_tasks_perplexity(comparative_data, model_name, os.path.join(output_prefix, "perplexity.png"))
	graph_tasks_tokenization(comparative_data, model_name, os.path.join(output_prefix, "tokenization.png"))
	graph_tasks_bpb(comparative_data, model_name, os.path.join(output_prefix, "bits-per-byte.png"))

def graph_model_comparison(output_prefix, comparative_data):
	graph_tasks_models_tokenization(comparative_data, os.path.join(output_prefix, "task-tokenization.png"))
	graph_tasks_models_bpb(comparative_data, os.path.join(output_prefix, "task-bits-per-byte.png"))

    # TODO: overall comparisons
    # see: https://seaborn.pydata.org/generated/seaborn.displot.html#seaborn.displot
    # need to ignore task data not shared by all models

def graph_tasks_models_tokenization(comparative_data, filename):
	task_name = []
	model_name = []
	bytes_per_token = []
	maximum_bytes_per_token = []
	items = 0

	for model, data in comparative_data.items():
		for key, value in data.items():
			items += 1
			for elem in value["bytes_per_token"]:
				task_name.append(key)
				model_name.append(model)
				bytes_per_token.append(elem)
			maximum_bytes_per_token.append(np.max(value["bytes_per_token"]))

	plt.figure(layout="constrained", figsize=[8.8, max(6.4, (2.4+items))])
	plt.suptitle("bytes per token by task + model")
	plt.xlabel("UTF-8 Bytes / Token")
	sns.violinplot(x=bytes_per_token, y=task_name, hue=model_name, density_norm="width")
	plt.xlim([1, math.ceil(np.percentile(maximum_bytes_per_token, 90))])
	plt.savefig(filename)
	plt.close()

def graph_tasks_models_bpb(comparative_data, filename):
	task_name = []
	model_name = []
	bits_per_byte = []
	items = 0

	for model, data in comparative_data.items():
		for key, value in data.items():
			items += 1
			for elem in value["bits_per_byte"]:
				task_name.append(key)
				model_name.append(model)
				bits_per_byte.append(elem)

	plt.figure(layout="constrained", figsize=[8.8, max(6.4, (2.4+items))])
	plt.suptitle("bits per byte by task + model")
	plt.xlabel("Bits / Byte")
	sns.violinplot(x=bits_per_byte, y=task_name, hue=model_name, density_norm="width")
	plt.xlim([0, 3])
	plt.savefig(filename)
	plt.close()

def graph_tasks_perplexity(comparative_data, model_name, filename):
	task_name = []
	perplexity = []

	for key, value in comparative_data.items():
		for elem in value["token_perplexity"]:
			task_name.append(key)
			perplexity.append(elem)

	plt.figure(layout="constrained", figsize=[8.8, max(6.4, (2.4+len(comparative_data.keys())))])
	plt.suptitle(model_name+" perplexity by task")
	plt.xlabel("Token Perplexity")
	sns.violinplot(x=perplexity, y=task_name, density_norm="width", log_scale=True)
	plt.xlim([1, 1000])
	plt.savefig(filename)
	plt.close()

def graph_tasks_tokenization(comparative_data, model_name, filename):
	task_name = []
	bytes_per_token = []
	maximum_bytes_per_token = []

	for key, value in comparative_data.items():
		for elem in value["bytes_per_token"]:
			task_name.append(key)
			bytes_per_token.append(elem)
		maximum_bytes_per_token.append(np.max(value["bytes_per_token"]))

	plt.figure(layout="constrained", figsize=[8.8, max(6.4, (2.4+len(comparative_data.keys())))])
	plt.suptitle(model_name+" bytes per token by task")
	plt.xlabel("UTF-8 Bytes / Token")
	sns.violinplot(x=bytes_per_token, y=task_name, density_norm="width")
	plt.xlim([1, math.ceil(np.percentile(maximum_bytes_per_token, 90))])
	plt.savefig(filename)
	plt.close()

def graph_tasks_bpb(comparative_data, model_name, filename):
	task_name = []
	bits_per_byte = []

	for key, value in comparative_data.items():
		for elem in value["bits_per_byte"]:
			task_name.append(key)
			bits_per_byte.append(elem)

	plt.figure(layout="constrained", figsize=[8.8, max(6.4, (2.4+len(comparative_data.keys())))])
	plt.suptitle(model_name+" bits per byte by task")
	plt.xlabel("Bits / Byte")
	sns.violinplot(x=bits_per_byte, y=task_name, density_norm="width")
	plt.xlim([0, 3])
	plt.savefig(filename)
	plt.close()

def graph_task_perplexity(items, task_name, filename):
	perplexities = []
	for item in items:
		perplexities.append(max(item["token_perplexity"], 1))

	plt.figure(layout="tight")
	plt.suptitle(task_name+" perplexity (n="+str(len(perplexities))+")")
	plt.xlabel("Token Perplexity")
	plt.ylabel("Dataset Items")
	sns.histplot(perplexities, kde=True, log_scale=True)
	plt.axvline(np.median(perplexities), color='.5', linestyle='--')
	plt.xlim([1, 1000])
	plt.savefig(filename)
	plt.close()

def graph_task_bpb(items, task_name, filename):
	bpbs = []
	for item in items:
		bpbs.append(item["bits_per_byte"])

	plt.figure(layout="tight")
	plt.suptitle(task_name+" bits per byte (n="+str(len(bpbs))+")")
	plt.xlabel("Bits / Byte")
	plt.ylabel("Dataset Items")
	sns.histplot(bpbs, kde=True)
	plt.axvline(np.median(bpbs), color='.5', linestyle='--')
	plt.xlim([0, 3])
	plt.savefig(filename)
	plt.close()

def graph_task_bpb_perplexity(items, task_name, filename):
	bpbs = []
	perplexities = []
	for item in items:
		bpbs.append(item["bits_per_byte"])
		perplexities.append(max(item["token_perplexity"], 1))

	g = sns.JointGrid(x=perplexities, y=bpbs, xlim=[1, 1000], ylim=[0, 3], height=9.6, ratio=3, marginal_ticks=True)
	g.figure.suptitle(task_name+" perplexity by bits per byte (n="+str(len(perplexities))+", 95% CI)")
	g.set_axis_labels("Token Perplexity", "Bits / Byte")
	g.ax_joint.set_xscale('log')
	if len(perplexities) > 1000:
		g.plot_joint(sns.regplot, scatter_kws={"alpha": 0.25}, logx=True, ci=95)
	elif len(perplexities) > 100:
		g.plot_joint(sns.regplot, scatter_kws={"alpha": 0.5}, logx=True, ci=95)
	else:
		g.plot_joint(sns.regplot, logx=True, ci=95)
	g.plot_marginals(sns.histplot, kde=True)
	g.refline(x=np.median(perplexities), y=np.median(bpbs))
	plt.savefig(filename)
	plt.close()

def graph_task_length_perplexity(items, task_name, filename):
	lengths = []
	perplexities = []
	for item in items:
		lengths.append(item["token_count"])
		perplexities.append(max(item["token_perplexity"], 1))

	g = sns.JointGrid(x=lengths, y=perplexities, ylim=[1, 1000], height=9.6, ratio=3, marginal_ticks=True)
	g.figure.suptitle(task_name+" perplexity by length (n="+str(len(perplexities))+")")
	g.set_axis_labels("Token Count", "Token Perplexity")
	g.ax_joint.set_xscale('log')
	g.ax_joint.set_yscale('log')
	if len(perplexities) > 1000:
		g.plot_joint(sns.scatterplot, alpha=0.25)
	elif len(perplexities) > 100:
		g.plot_joint(sns.scatterplot, alpha=0.5)
	else:
		g.plot_joint(sns.scatterplot, alpha=1)
	g.plot_marginals(sns.histplot, kde=True)
	g.refline(x=np.median(lengths), y=np.median(perplexities))
	g.savefig(filename)
	plt.close()

def graph_task_tokenization_perplexity(items, task_name, filename):
	bytes_per_token = []
	perplexities = []
	for item in items:
		bytes_per_token.append(max(item["byte_count"] / item["token_count"], 1))
		perplexities.append(max(item["token_perplexity"], 1))

	g = sns.JointGrid(x=bytes_per_token, y=perplexities, ylim=[1, 1000], height=9.6, ratio=3, marginal_ticks=True)
	g.figure.suptitle(task_name+" perplexity by bytes per token (n="+str(len(perplexities))+")")
	g.set_axis_labels("UTF-8 Bytes / Token", "Token Perplexity")
	g.ax_joint.set_yscale('log')
	if len(perplexities) > 1000:
		g.plot_joint(sns.scatterplot, alpha=0.25)
	elif len(perplexities) > 100:
		g.plot_joint(sns.scatterplot, alpha=0.5)
	else:
		g.plot_joint(sns.scatterplot, alpha=1)
	g.plot_marginals(sns.histplot, kde=True)
	g.refline(x=np.median(bytes_per_token), y=np.median(perplexities))
	g.savefig(filename)
	plt.close()

def graph_task_positional_perplexity(positional_probs, task_name, filename):
	positional_probs=list(positional_probs.values())

	prob_position = []
	prob_value = []
	for pos, prob_set in enumerate(positional_probs, start=1):
		for prob in prob_set:
			prob_position.append(pos)
			prob_value.append(prob)

	items = len(positional_probs)

	plt.figure(layout="tight", figsize=[11.2, 4.8])
	plt.suptitle(task_name+" perplexity by position ((i=1, n="+str(len(positional_probs[0]))+"), (i="+str(items)+", n="+str(len(positional_probs[items-1]))+"), 95% CI)")
	plt.xlabel("Token Position")
	plt.ylabel("Token Perplexity")
	sns.lineplot(x=prob_position, y=prob_value, errorbar=("ci", 95), estimator="median", seed=0)
	plt.loglog()
	plt.xlim([1, items])
	plt.ylim([1, 1000])
	plt.savefig(filename)
	plt.close()

def get_input_line_count(filename):
	with open(filename) as file:
		return sum(1 for line in file)

task_comparative_data = {}

def process_input_data(filename):
	line_count = get_input_line_count(filename)
	input_file = open(filename)

	metadata = {}
	task_metrics = {}
	tasks = {}
	task_positional_probs = {}
	output_prefix = ""
	model_name = ""

	for line in tqdm.tqdm(input_file, desc=filename, total=line_count):
		line_data = json.loads(line)
		if len(metadata) == 0:
			metadata = line_data
			model_name = metadata["endpoint_info"]["model_id"]
			task_comparative_data[model_name] = {}
			metadata["analyzed_from"] = filename
			output_prefix = os.path.join(args.output_dir, model_name)
			os.makedirs(output_prefix)
			write_json(metadata, os.path.join(output_prefix, "metadata.json"))
			continue

		task_name = None
		for key, value in line_data.items():
			if key == "start_task":
				task_name = value
				task_metrics[value] = {"completed": False}
			elif key == "completed_task":
				task_name = value
				task_metrics[value]["completed"] = True
				graph_task(output_prefix, task_name, tasks[task_name], task_positional_probs[task_name], False)
				task_calculated_outputs = calculate_task_data_metrics(tasks[task_name])
				for key, value in task_calculated_outputs[0].items():
					task_metrics[task_name][key] = value
				task_comparative_data[model_name][task_name] = task_calculated_outputs[1]
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

	input_file.close()

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
			graph_task(output_prefix, task_name, tasks[task_name], task_positional_probs[task_name], True)
			task_calculated_outputs = calculate_task_data_metrics(tasks[task_name])
			for key, value in task_calculated_outputs[0].items():
				task_metrics[task_name][key] = value
			task_comparative_data[model_name][task_name+"*"] = task_calculated_outputs[1]
			task_positional_probs[task_name] = {}

	graph_tasks(output_prefix, task_comparative_data[model_name], model_name)
	write_json(task_metrics, os.path.join(output_prefix, "metrics.json"))

for filename in args.input_files:
	process_input_data(filename)

graph_model_comparison(args.output_dir, task_comparative_data)

# measurements to include:
# - average bits per byte by model
# - bits per byte by task + model
# - average bytes/token by model
# - bytes/token by task + model

# create graphical plots, don't just output a handful of measurements
# - make data distributions visible
# - make statistical significance visible