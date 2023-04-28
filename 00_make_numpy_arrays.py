"""
Make numpy arrays of shape image_condition x repetitions x eeg_channels x n_samples
from the preprocessed EEGLAB data (source) in THINGS EEG1 dataset.
"""

import os
import numpy as np
import pandas as pd
import mne
from tqdm.auto import tqdm

data_dir = "../data/source/"
np_dir = "../data/preprocessed_data/"
if not os.path.exists(np_dir):
    os.mkdir(np_dir)

for p_id in tqdm(range(1, 51)):
    # exclude subjects 1, 6, 18, 23, 49, 50
    if p_id in [1, 6, 18, 23, 49, 50]:
        continue

    sub_dir = np_dir + f"sub-{format(p_id, '02')}/"

    print("participant ", p_id)
    if os.path.exists((sub_dir)):
        print("exists...")
        continue

    # Load preprocessed participant data
    raw = mne.io.read_raw_eeglab(data_dir + 'sub-' + format(p_id, '02') + '_task-rsvp_continuous.set')

    # Select only occipital (O) and posterior (P) channels
    chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'],'^O *|^P *'))
    new_chans = [raw.info['ch_names'][c] for c in chan_idx]
    raw.pick_channels(new_chans)

    # get events and epoch based on stimulus onset
    events_from_annot, events_dict = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events_from_annot, event_id=1, tmin=-0.1, tmax=0.4)

    # resample to 200 Hz
    epochs.load_data()
    epochs.resample(200)
    data = epochs.get_data()

    # Load csv containing stimulus information
    df = pd.read_csv(data_dir + 'sub-' + format(p_id, '02') + '_task-rsvp_events.csv')

    # split the data into normal sequences and validation sequences
    train_data = data[:-2400, :, :]
    test_data = data[-2400:, :, :]

    # get stimulus for normal sequences and validation sequences
    train_stimulus = np.array(df['stim'])[:-2400]
    test_stimulus = np.array(df.iloc[-2400:]['stim'])

    # get unique
    train_images = sorted(list(set(train_stimulus)))
    test_images = sorted(list(set(test_stimulus)))

    # exclude test classes from training data
    def get_class(im):
        fo = im.find('\\')
        so = im[fo + 1:].find('\\')
        return im[fo + 1:fo + so + 1]

    test_classes = []
    for image in test_images:
        test_classes.append(get_class(image))

    exclude_idx = []
    for i, image in enumerate(train_stimulus):
        if get_class(image) in test_classes:
            exclude_idx.append(i)

    mask = np.ones(len(train_stimulus), dtype=bool)
    mask[exclude_idx] = False

    train_stimulus = train_stimulus[mask]
    train_data = train_data[mask]

    # sort train_data by image names to simplify many things
    sort_idx = np.argsort(train_stimulus)
    train_stimulus = train_stimulus[sort_idx]
    train_data = train_data[sort_idx]

    test_all_indices = []
    for image in test_images:
        indices = np.where(test_stimulus == image)[0]
        test_all_indices.append(indices)

    # reshape based on image repetitions
    tmp = []
    for i in range(200):
        tmp.append(test_data[test_all_indices[i]])

    # test_data now also sorted by test_image names
    test_data = np.array(tmp)
    train_data = np.expand_dims(train_data, 1)

    # rescale data in units of uV
    train_data = 10**6 * train_data
    test_data = 10**6 * test_data

    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

    # write excluded event list
    pd.Series(train_stimulus).to_csv(sub_dir + "train_stimulus.csv")
    pd.Series(test_images).to_csv(sub_dir + "test_stimulus.csv")

    # write data as dictionaries with numpy array inside
    train_dict = {
		'preprocessed_eeg_data': train_data,
		'ch_names': new_chans,
		'times': epochs.times
	}

    test_dict = {
		'preprocessed_eeg_data': test_data,
		'ch_names': new_chans,
		'times': epochs.times
	}

    np.save(sub_dir + "preprocessed_eeg_training.npy", train_dict)
    np.save(sub_dir + "preprocessed_eeg_test.npy", test_dict)

