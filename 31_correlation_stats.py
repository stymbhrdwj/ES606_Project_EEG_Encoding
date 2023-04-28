"""Calculate the confidence intervals (through bootstrap tests) and significance
(through one-sample t-tests) of the correlation analysis results, and of the
differences between the results and the noise ceiling.

Parameters
----------
used_subs : list
	List of subjects used for the stats.
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
	Number of iterations for the bootstrap test.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--used_subs', default=[2,3,4,5,7,8,9,10,14,16,21,22,24,26,27,28,29,30,31,33,34,35,37,38,40,41,42,43,44,45,46,47], type=list)
parser.add_argument('--encoding_type', default='end_to_end', type=str)
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--pretrained', default=False, type=bool)
parser.add_argument('--subjects', default='within', type=str)
parser.add_argument('--layers', default='all', type=str)
parser.add_argument('--n_components', default=1000, type=int)
parser.add_argument('--modeled_time_points', type=str, default='all')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_iter', default=10000, type=int)
parser.add_argument('--project_dir', default='../data/', type=str)
args = parser.parse_args()

print('>>> Correlation stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Load the correlation results
# =============================================================================
correlation = {}
for s, sub in enumerate(args.used_subs):
	if args.encoding_type == 'linearizing':
		data_dir = os.path.join('results', 'sub-'+
			format(sub,'02'), 'correlation', 'encoding-linearizing','subjects-'+
			args.subjects, 'dnn-'+args.dnn, 'pretrained-'+str(args.pretrained),
			'layers-'+args.layers, 'n_components-'+
			format(args.n_components,'05'), 'correlation.npy')
	elif args.encoding_type == 'end_to_end':
		data_dir = os.path.join('results', 'sub-'+
			format(sub,'02'), 'correlation', 'encoding-end_to_end', 'dnn-'+
			args.dnn, 'modeled_time_points-'+args.modeled_time_points,
			'pretrained-'+str(args.pretrained), 'lr-{:.0e}'.format(args.lr)+
			'__wd-{:.0e}'.format(args.weight_decay)+'__bs-'+
			format(args.batch_size,'03'), 'correlation.npy')
	results_dict = np.load(os.path.join(args.project_dir, data_dir),
		allow_pickle=True).item()
	for layer in results_dict['correlation'].keys():
		if s == 0:
			correlation[layer] = np.expand_dims(
				results_dict['correlation'][layer], 0)
		else:
			correlation[layer] = np.append(correlation[layer], np.expand_dims(
				results_dict['correlation'][layer], 0), 0)
	times = results_dict['times']
	ch_names = results_dict['ch_names']
del results_dict

# =============================================================================
# Bootstrap the confidence intervals (CIs)
# =============================================================================
ci_lower = {}
ci_upper = {}

# Calculate the CIs independently at each time point
for layer in correlation.keys():
	# CI matrices of shape: (Time)
	ci_lower[layer] = np.zeros((correlation[layer].shape[2]))
	ci_upper[layer] = np.zeros((correlation[layer].shape[2]))

	for t in tqdm(range(correlation[layer].shape[2])):
		sample_dist = np.zeros(args.n_iter)
		sample_dist_diff = np.zeros(args.n_iter)
		for i in range(args.n_iter):
			# Calculate the sample distribution of the correlation values
			# averaged across channels
			sample_dist[i] = np.mean(resample(np.mean(
				correlation[layer][:,:,t], 1)))
		# Calculate the 95% confidence intervals
		ci_lower[layer][t] = np.percentile(sample_dist, 2.5)
		ci_upper[layer][t] = np.percentile(sample_dist, 97.5)

# =============================================================================
# One-sample t-tests for significance & multiple comparisons correction
# =============================================================================
p_values = {}
for layer in correlation.keys():
	# p-values matrices of shape: (Time)
	p_values[layer] = np.ones((correlation[layer].shape[2]))
	for t in range(correlation[layer].shape[2]):
		# Fisher transform the correlation values and perform the t-tests
		fisher_vaules = np.arctanh(np.mean(correlation[layer][:,:,t], 1))
		p_values[layer][t] = ttest_1samp(fisher_vaules, 0,
			alternative='greater')[1]

# Correct for multiple comparisons
significance = {}
for layer in p_values.keys():
	significance[layer] = multipletests(p_values[layer], 0.05, 'bonferroni')[0]


# =============================================================================
# Save the results
# =============================================================================
# Store the results into a dictionary
stats_dict = {
	'correlation': correlation,
	'ci_lower': ci_lower,
	'ci_upper': ci_upper,
	'significance': significance,
	'times': times,
	'ch_names': ch_names
}

# Saving directory
if args.encoding_type == 'linearizing':
	save_dir = os.path.join(args.project_dir, 'results', 'stats', 'correlation',
		'encoding-linearizing', 'subjects-'+args.subjects, 'dnn-'+args.dnn,
		'pretrained-'+str(args.pretrained), 'layers-'+args.layers,
		'n_components-'+format(args.n_components,'05'))
elif args.encoding_type == 'end_to_end':
	save_dir = os.path.join(args.project_dir, 'results', 'stats', 'correlation',
		'encoding-end_to_end', 'dnn-'+args.dnn, 'modeled_time_points-'+
		args.modeled_time_points, 'pretrained-'+str(args.pretrained),
		'lr-{:.0e}'.format(args.lr)+'__wd-{:.0e}'.format(args.weight_decay)+
		'__bs-'+format(args.batch_size,'03'))
file_name = 'correlation_stats.npy'

# Create the directory if not existing and save
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), stats_dict)
