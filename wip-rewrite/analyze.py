import os
import json
import argparse
import tqdm
import time
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('bench_output', nargs="+", type=str)

args = parser.parse_args()
