import os
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams['font.size'] = 14
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

data_dir = '../data/'
plot_dir = '../data/plots/erps'

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# exclusion based on paper suggestion
excl = [1, 6, 18, 23, 49, 50]
p_ids = list(set(list(range(1, 51))) - set(excl))

erps = []
corrs = []

for s in tqdm(p_ids, total=len(p_ids)):

    eeg_parent_dir = os.path.join(data_dir, 'preprocessed_data', 'sub-'+format(s, '02'))
    eeg_data_train = np.load(os.path.join(eeg_parent_dir,
        'preprocessed_eeg_training.npy'), allow_pickle=True).item()
    eeg_data_test = np.load(os.path.join(eeg_parent_dir,
        'preprocessed_eeg_test.npy'), allow_pickle=True).item()

    times = eeg_data_train['times']
    ch_names = eeg_data_train['ch_names']

    erp_data_train = np.mean(eeg_data_train['preprocessed_eeg_data'], 1)
    erp_data_test = np.mean(eeg_data_test['preprocessed_eeg_data'], 1)
    erp_data_all = np.mean(np.append(erp_data_train, erp_data_test, 0), 0)

    erps.append(np.transpose(erp_data_all))

    V_limit = 15

    plt.figure(figsize=(12, 6))
    plt.plot([-.1, .4], [0, 0], 'k--', [0, 0], [-V_limit, V_limit], 'k--')
    plt.plot(times, np.transpose(erp_data_all))
    plt.xlabel('Time (s)')
    plt.xlim(left=-.1, right=.4)
    plt.ylabel('Voltage (uV)')
    plt.ylim(bottom=-V_limit, top=V_limit)
    plt.tight_layout()
    plt.legend(['_', '_'] + ch_names, loc='upper right', bbox_to_anchor=(1.14, 1.02))
    plt.savefig(os.path.join(plot_dir, 'ERP_' + format(s, '02') + '.png'), dpi=300, bbox_inches='tight')
    plt.show()

# new exclusions based on visual inspection of ERP plots
excl = [11, 12, 13, 15, 17, 19, 20, 25, 32, 36, 39, 48]
