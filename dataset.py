#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset


def num_lines_in_file(filepath):
    with open(filepath, 'r') as file:
        return sum(1 for _ in file)


class EarthquakeSegmentDataset(Dataset):

    def __init__(self, folder, seglen=1000):
        super().__init__()
        self.folder = folder
        self.seglen = seglen

        # pattern = re.compile(r'segment(.*)\.csv')
        # self.file_names = sorted(os.listdir(self.folder), key=lambda x: int(pattern.search(x).group(1)))
        # self.earthquake_lengths = [num_lines_in_file(os.path.join(self.folder, file))-1 for file in self.file_names]

        self.file_names = ['segment0.csv', 'segment1.csv', 'segment2.csv', 'segment3.csv', 'segment4.csv',
                           'segment5.csv', 'segment6.csv', 'segment7.csv', 'segment8.csv', 'segment9.csv',
                           'segment10.csv', 'segment11.csv', 'segment12.csv', 'segment13.csv', 'segment14.csv',
                           'segment15.csv', 'segment16.csv']
        self.earthquake_lengths = [5656574, 44429304, 54591478, 34095097, 48869367, 31010810, 27176955, 62009332,
                                   30437370,
                                   37101561, 43991032, 42442743, 33988602, 32976890, 56791029, 36417529, 7159807]

    # NOTE: this approach maybe painfully slow because you will have to read the file in line by line...
    def __getitem__(self, item):
        # should be able to say 'I want the segment from i = 0 ... TOTAL_ROWS-1'
        # this will require a Dataset that can span across several earthquake segments
        # need a mapping of earthquake segment -> number of timesteps (file -> number of lines in file-1)
        return

    def __len__(self):
        return sum(self.earthquake_lengths)
