"""Correlation of each synthetic test EEG data features (EEG_channels x
EEG_time_points) with the corresponding biological test EEG data features
(across the 200 test image conditions).

Parameters
----------
sub : int
	Used subject.
encoding_type : str
	Whether to analyze the 'linearizing' or 'end-to-end' encoding synthetic
	data.
dnn : str
	Used DNN network.
pretrained : bool
	If True, analyze the data synthesized through pretrained (linearizing or
	end-to-end) models. If False, analyze the data synthesized through randomly
	initialized (linearizing or end-to-end) models.
subjects : str
	If 'linearizing' encoding_type is chosen, whether to analyze the 'within' or
	'between' subjects linearizing encoding synthetic data.
layers : str
	If 'linearizing' encoding_type is chosen, whether to analyse the data
	synthesized using 'all', 'single' or 'appended' DNN layers feature maps.
n_components : int
	If 'linearizing' encoding_type is chosen, number of DNN feature maps PCA
	components retained for synthesizing the EEG data.
modeled_time_points : str
	If 'end_to_end' encoding_type is chosen, whether to analyze the synthetic
	data of end-to-end models trained to predict 'single' or 'all' time points.
lr : float
	If 'end_to_end' encoding_type is chosen, learning rate used to train the
	end-to-end encoding models.
weight_decay : float
	If 'end_to_end' encoding_type is chosen, weight decay coefficint used to
	train the end-to-end encoding models.
batch_size : int
	If 'end_to_end' encoding_type is chosen, batch size used to train the
	end-to-end encoding models.
n_iter : int
	Number of analysis iterations.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import pearsonr as corr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=2, type=int)
parser.add_argument('--encoding_type', default='end_to_end', type=str)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--pretrained', default='True', type=str)
parser.add_argument('--subjects', default='within', type=str)
parser.add_argument('--layers', default='all', type=str)
parser.add_argument('--n_components', default=1000, type=int)
parser.add_argument('--modeled_time_points', type=str, default='all')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_iter', default=100, type=int)
parser.add_argument('--project_dir', default='../data/', type=str)
parser.add_argument('--dry', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

print('>>> Correlation <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Load the biological EEG test data
# =============================================================================
data_dir = os.path.join('preprocessed_data', 'sub-'+
	format(args.sub,'02'), 'preprocessed_eeg_test.npy')
bio_test = np.load(os.path.join(args.project_dir, data_dir), allow_pickle=True).item()
times = bio_test['times']
ch_names = bio_test['ch_names']
bio_test = bio_test['preprocessed_eeg_data']

# =============================================================================
# Load the synthetic EEG test data
# =============================================================================
if args.encoding_type == 'linearizing':
	data_dir = os.path.join('results', 'sub-'+
		format(args.sub,'02'), 'synthetic_eeg_data', 'encoding-linearizing',
		'subjects-'+args.subjects, 'dnn-'+args.dnn, 'pretrained-'+
		str(args.pretrained), 'layers-'+args.layers, 'n_components-'+
		format(args.n_components,'05'), 'synthetic_eeg_test.npy')
elif args.encoding_type == 'end_to_end':
	data_dir = os.path.join('results', 'sub-'+
		format(args.sub,'02'), 'synthetic_eeg_data', 'encoding-end_to_end',
		'dnn-'+args.dnn, 'modeled_time_points-'+args.modeled_time_points,
		'pretrained-'+str(args.pretrained), 'lr-{:.0e}'.format(args.lr)+
		'__wd-{:.0e}'.format(args.weight_decay)+'__bs-'+
		format(args.batch_size,'03'), 'synthetic_eeg_test.npy')

synt_test = np.load(os.path.join(args.project_dir, data_dir),
	allow_pickle=True).item()
synt_test = synt_test['synthetic_data']


# =============================================================================
# Compute the correlation
# =============================================================================
# Results matrices of shape:
# (Iterations × EEG channels × EEG time points)

# Average across all the biological data repetitions for the noise ceiling
# upper bound calculation or plain correlation
bio_data_avg_all = np.mean(bio_test, 1)

correlation = {}
for layer in synt_test.keys():
	correlation[layer] = np.zeros((bio_test.shape[2],
		bio_test.shape[3]))
# Loop over EEG time points and channels
for t in range(bio_test.shape[3]):
	for c in range(bio_test.shape[2]):
		# Compute the correlation
		for layer in synt_test.keys():
			correlation[layer][c,t] = corr(synt_test[layer][:,c,t],
				bio_data_avg_all[:,c,t])[0]

# =============================================================================
# Save the results
# =============================================================================
# Store the results into a dictionary

results_dict = {
	'correlation' : correlation,
	'times': times,
	'ch_names': ch_names
}

# Saving directory
if args.encoding_type == 'linearizing':
	save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
		format(args.sub,'02'), 'correlation', 'encoding-linearizing',
		'subjects-'+args.subjects, 'dnn-'+args.dnn, 'pretrained-'+
		str(args.pretrained), 'layers-'+args.layers, 'n_components-'+
		format(args.n_components,'05'))
elif args.encoding_type == 'end_to_end':
	save_dir = os.path.join(args.project_dir, 'results', 'sub-'+
		format(args.sub,'02'), 'correlation', 'encoding-end_to_end',
		'dnn-'+args.dnn, 'modeled_time_points-'+args.modeled_time_points,
		'pretrained-'+str(args.pretrained), 'lr-{:.0e}'.format(args.lr)+
		'__wd-{:.0e}'.format(args.weight_decay)+'__bs-'+
		format(args.batch_size,'03'))
file_name = 'correlation.npy'

# Create the directory if not existing and save
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

if not args.dry:
	print("Saving the results...")
	np.save(os.path.join(save_dir, file_name), results_dict)

