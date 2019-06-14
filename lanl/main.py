#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pandas import DataFrame, read_csv
import torch
from tqdm import tqdm
from dtsckit.metrics import AverageKeeper
from dtsckit.utils import read_pickle
from lanl.dataset import SegFeatureGen, TestSegFeatureGen, get_quake_indices
from lanl.model import FullModel, sort_pad_pack
from lanl.segmenter import SpikeSegmenter


def train_batch(model, data_gen, criterion, optimizer, device, num_features):
    batch_a, batch_s, batch_f, batch_t = next(data_gen)
    packed_batch_s, sort_idx = sort_pad_pack(batch_s, num_features)
    batch_a = batch_a[sort_idx]
    batch_f = batch_f[sort_idx]
    batch_t = batch_t[sort_idx]

    packed_batch_s = packed_batch_s.to(device)
    batch_a = batch_a.to(device)
    batch_f = batch_f.to(device)
    batch_t = batch_t.to(device)

    optimizer.zero_grad()
    out = model(batch_a, packed_batch_s, batch_f).squeeze()
    loss = criterion(out, batch_t)
    loss.backward()
    optimizer.step()

    return loss.item()


def validate_batch(model, data_gen, criterion, device, num_features):
    batch_a, batch_s, batch_f, batch_t = next(data_gen)
    packed_batch_s, sort_idx = sort_pad_pack(batch_s, num_features)
    batch_a = batch_a[sort_idx]
    batch_f = batch_f[sort_idx]
    batch_t = batch_t[sort_idx]

    packed_batch_s = packed_batch_s.to(device)
    batch_a = batch_a.to(device)
    batch_f = batch_f.to(device)
    batch_t = batch_t.to(device)

    out = model(batch_a, packed_batch_s, batch_f).squeeze()
    loss = criterion(out, batch_t)
    return loss.item()


def train_and_validate(model, train_gen, valid_gen, criterion, optimizer, device, model_dir,
                       num_epochs=100, num_train_batches=50, num_valid_batches=50, validation_rate=1, num_features=13):
    train_avg = AverageKeeper()
    valid_avg = AverageKeeper()
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model = model.train()
        for _ in range(num_train_batches):
            loss = train_batch(model, train_gen, criterion, optimizer, device, num_features)
            train_avg.add(loss)

        print(f'Epoch {epoch}, Training Loss = {train_avg.calculate()}\n')
        train_avg.reset()

        if epoch % validation_rate == 0:
            model = model.eval()
            with torch.no_grad():
                for _ in range(num_valid_batches):
                    loss = validate_batch(model, valid_gen, criterion, device, num_features)
                    valid_avg.add(loss)

            v_avg = valid_avg.calculate()
            if v_avg < best_val_loss:
                best_val_loss = v_avg
                torch.save(model.state_dict(), os.path.join(model_dir, f'model_{epoch}.pt'))

            print(f'Epoch {epoch}, Testing Loss = {v_avg}\n')
            valid_avg.reset()

    return model, train_avg.running_avg, valid_avg.running_avg


def train_on_all(model, data_gen, num_epochs=1000, num_train_batches=10, num_features=13, model_save_rate=5):
    avg_keeper = AverageKeeper()
    for epoch in range(num_epochs):
        model = model.train()
        for batch in range(num_train_batches):
            loss = train_batch(model, data_gen, l1_criterion, adam_optim, device, num_features)
            avg_keeper.add(loss)

        print(f'Epoch {epoch}, Training Loss = {avg_keeper.calculate()}\n')
        avg_keeper.reset()

        if epoch % model_save_rate == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'full_model_{epoch}.pt'))

    return model, avg_keeper.running_avg


def make_prediction(model, data_gen, num_features, test_data_dir, submission_filepath):
    segment_ids = []
    predictions = []
    for filename in tqdm(os.listdir(test_data_dir)):
        seg_id = filename[:-4]  # throw away the csv file extension
        test_segment = read_csv(os.path.join(test_data_dir, filename))
        test_segment = test_segment.values.flatten()
        x_a, x_s, x_f = data_gen(test_segment)

        packed_s, sort_idx = sort_pad_pack(x_s, num_features)
        x_a = x_a[sort_idx]
        x_f = x_f[sort_idx]

        packed_s = packed_s.to(device)
        x_a = x_a.to(device)
        x_f = x_f.to(device)

        with torch.no_grad():
            out = model(x_a, packed_s, x_f).squeeze()
            segment_ids.append(seg_id)
            predictions.append(out.item())

    submission = DataFrame({'seg_id': segment_ids, 'time_to_failure': predictions})
    submission.to_csv(submission_filepath, index=False)


if __name__ == '__main__':
    TRAIN_FILE = '/home/mchobanyan/data/kaggle/lanl_earthquake/train.pkl'
    MODEL_DIR = '/home/mchobanyan/data/kaggle/lanl_earthquake/models/final_model/'
    TEST_DIR = '/home/mchobanyan/data/kaggle/lanl_earthquake/test'
    SUBMISSION_FILEPATH = '/home/mchobanyan/data/kaggle/lanl_earthquake/submission.csv'

    df = read_pickle(TRAIN_FILE)
    data = df.values
    n = len(data)
    print(f'Data ready: {n} total timesteps')

    quake_starts = get_quake_indices(df.iloc[:, 1])

    batch_size = 64
    num_features = 13  # change this number if 'extract_segment_features' is modified
    segmenter = SpikeSegmenter(mu=3.5, sigma=4.5, window_size=300, only_spikes=True)

    train_data_gen = iter(SegFeatureGen(data, segmenter, batch_size=batch_size, min_index=quake_starts[2] + 1))
    valid_data_gen = iter(SegFeatureGen(data, segmenter, batch_size=batch_size, max_index=quake_starts[2]))
    full_data_gen = iter(SegFeatureGen(data, segmenter, batch_size=batch_size))

    device = torch.device('cuda')
    full_model = FullModel(num_seg_features=num_features)
    try:  # this avoids a "RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED" error...
        full_model = full_model.to(device)
    except:
        full_model = full_model.to(device)

    l1_criterion = torch.nn.L1Loss()
    adam_optim = torch.optim.Adam(full_model.parameters(), lr=0.0005)

    train_on_all(full_model, full_data_gen, num_epochs=3, num_train_batches=2)

    # pick the model to submit
    model_i = 20
    final_model_file = os.path.join(MODEL_DIR, f'full_model_{model_i}.pt')

    test_data_gen = TestSegFeatureGen(segmenter)
    final_model = full_model.load_state_dict(torch.load(final_model_file))
    final_model = final_model.eval()

    print('done training...')
    make_prediction(final_model, test_data_gen, num_features, TEST_DIR, SUBMISSION_FILEPATH)
    print('created submission file...')
