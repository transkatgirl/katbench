import os
import json
import argparse
import time
import datetime
import math

parser = argparse.ArgumentParser()
parser.add_argument('input_files', nargs="+", type=str)
parser.add_argument('output_file', type=str)

args = parser.parse_args()

output_file = open(args.output_file)

for filename in args.input_files:
	input_file = open(filename)

	input_file.close()

output_file.flush()
os.fsync(output_file)
output_file.close()