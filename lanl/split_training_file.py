#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Splits the original training file into several files, one for each recorded earthquake
Only use this approach when we are limited by memory...
"""

import os
import csv
import numpy as np
from dtsckit.utils import write_pickle


THRESHOLD = 0.01  # an arbitrary value signifying a new earthquake segment
root_folder = '/home/mchobanyan/data/kaggle/lanl_earthquake/'


def get_quake_indices(quake_times):
	"""Get the starting indices of each earthquake segment"""
	time_steps = quake_times.index[quake_times.diff() > 0.1].values
	return np.insert(time_steps, 0, 0)


def save_earthquakes_as_pickles(data):
	"""Save each distinct earthquake as a pickle

	Parameters
	----------
	data: pd.DataFrame
		A pandas data frame of the data with the first column as the acoustic value
		and the second column as the time_to_failure
	"""
	quake_starts = get_quake_indices(data.iloc[:, 1])
	segments = np.split(data.values, quake_starts[1:])
	for i, segment in enumerate(segments):
		write_pickle(segment, os.path.join(root_folder, f'training_earthquakes/segment{i}.pkl'))


def main():

	i = 0
	segment_id = 0
	prev_time = None
	header = None
	segment = []
	with open(os.path.join(root_folder, 'train.csv'), 'r') as main_file:

		file_reader = csv.reader(main_file)
		for line in file_reader:

			if i == 0:
				header = line
			else:

				time_left = float(line[1])

				if i == 1:
					prev_time = time_left

				elif time_left-prev_time > THRESHOLD:

					# write the segment to a new file
					filename = f'segment{segment_id}.csv'
					with open(os.path.join(root_folder, 'train', filename), 'w') as segment_file:
						file_writer = csv.writer(segment_file)
						for line_out in segment:
							file_writer.writerow(line_out)
					
					segment_id += 1
					segment = [header]
					print(f'Created {filename} after {i} lines...')

				prev_time = time_left

			segment.append(line)
			i += 1

	# output the remaining earthquake segment if needed
	if len(segment) > 1:
		filename = f'segment{segment_id}.csv'
		with open(os.path.join(root_folder, 'train', filename), 'w') as segment_file:
			file_writer = csv.writer(segment_file)
			for line_out in segment:
				file_writer.writerow(line_out)
		print(f'Created {filename} after {i} lines...')


if __name__ == '__main__':
	main()