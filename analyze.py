import os
import json
import argparse
import tqdm
import time
import datetime
import math
from tokenizers import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('bench_data', nargs="+", type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--tokenizer', type=str)

args = parser.parse_args()

outputfilename = args.output_file or "metrics-" + str(round(time.time())) + ".json"

tokenizer = Tokenizer.from_pretrained(args.tokenizer or "allenai/OLMo-2-0425-1B")

for filename in args.bench_data:
	file = open(filename)
	for line in file:
		linedata = json.loads(line)
		for key, value in linedata.items():
			if isinstance(value, list):
				task = key
				token_count = 0
				text = ""
				logprob_sum = 0
				for token in value:
					for key, value in token.items():
						if value:
							token_count += 1
							text += key
							logprob_sum += value
				byte_count = len(text.encode("utf-8"))
				llm_token_perplexity = math.exp(-logprob_sum / token_count)
				normalized_token_perplexity = math.exp(-logprob_sum / len(tokenizer.encode(text)))
				byte_perplexity = math.exp(-logprob_sum / byte_count)

				print(task, "byte_perplexity", byte_perplexity, "normalized_token_perplexity", normalized_token_perplexity)
	file.close()