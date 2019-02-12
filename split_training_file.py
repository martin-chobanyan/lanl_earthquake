#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Splits the original training file into several files, one for each recorded earthquake"""

import os
import csv


THRESHOLD = 0.01  # an arbitrary value signifying a new earthquake segment
root_folder = '/home/mchobanyan/data/kaggle/lanl_earthquake/'


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
