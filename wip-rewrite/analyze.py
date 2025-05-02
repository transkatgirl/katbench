import os
import json
import argparse
import tqdm
import time
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('bench_data', nargs="+", type=str)
parser.add_argument('--output_file', type=str)

args = parser.parse_args()

outputfilename = args.output_file or "metrics-" + str(round(time.time())) + ".json"

for filename in args.bench_data:
	file = open(filename)
	for line in file:
		linedata = json.loads(line)
		print(linedata)
	file.close()