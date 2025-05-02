import os
import json
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('input_files', nargs="+", type=str)
parser.add_argument('--output_file', type=str, default="output-collated-" + str(round(time.time())) + ".jsonl")

args = parser.parse_args()

print("open_output_file", args.output_file)
output_file = open(args.output_file, "x")

metadata = {"collated": args.input_files}
global_completed_tasks = set([])

for filename in args.input_files:
	print("process_input_file", filename)
	input_file = open(filename)

	completed_tasks = set([])

	is_start = True
	for line in input_file:
		if is_start:
			is_start = False
			line_data = json.loads(line)
			for key, value in line_data.items():
				if key not in metadata:
					metadata[key] = value
				elif key == "tasks":
					for key, value in value.items():
						if key not in metadata["tasks"]:
							metadata["tasks"][key] = value
						else:
							assert metadata["tasks"][key] == value
				elif key == "collated":
					metadata["collated"] = [*value, *metadata["collated"]]
				else:
					assert metadata[key] == value
		try:
			line_data = json.loads(line)
			for key, value in line_data.items():
				if key == "completed_task" and value not in global_completed_tasks:
					completed_tasks.add(value)
					global_completed_tasks.add(value)
		except:
			break

	input_file.seek(0)

	is_start = True
	for line in input_file:
		if is_start:
			is_start = False
			continue
		line_data = {}
		try:
			line_data = json.loads(line)
			for key, value in line_data.items():
				if key in completed_tasks:
					output_file.write(json.dumps(line_data, separators=(',', ':')))
					output_file.write("\n")
					break
				elif key == "start_task" and value in completed_tasks:
					output_file.write(json.dumps(line_data, separators=(',', ':')))
					output_file.write("\n")
					break
				elif key == "completed_task" and value in completed_tasks:
					output_file.write(json.dumps(line_data, separators=(',', ':')))
					output_file.write("\n")
					break
		except:
			continue

	input_file.close()

print("close_output_file", args.output_file)
output_file.seek(0)
output_file.write(json.dumps(metadata, separators=(',', ':')))
output_file.write("\n")

output_file.flush()
os.fsync(output_file)
output_file.close()