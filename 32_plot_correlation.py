"""Plot the correlation analysis results.

Parameters
----------
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

fig_dir = "../data/plots/correlation/"
if not os.path.exists(fig_dir):
	os.mkdir(fig_dir)

within = True
between = True
end_to_end = True

all_subs = [2,3,4,5,7,8,9,10,14,16,21,22,24,26,27,28,29,30,31,33,34,35,37,38,40,41,42,43,44,45,46,47]


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../data/', type=str)
args = parser.parse_args()


# =============================================================================
# Set plot parameters
# =============================================================================
# Setting the plot parameters
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
# matplotlib.rcParams['font.size'] = 16
# plt.rc('xtick', labelsize=16)
# plt.rc('ytick', labelsize=16)
# matplotlib.rcParams['axes.linewidth'] = 3
# matplotlib.rcParams['xtick.major.width'] = 3
# matplotlib.rcParams['xtick.major.size'] = 5
# matplotlib.rcParams['ytick.major.width'] = 3
# matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
color_noise_ceiling = (150/255, 150/255, 150/255)
colors = [(31/255, 119/255, 180/255), (255/255, 127/255, 14/255),
	(44/255, 160/255, 44/255), (214/255, 39/255, 40/255),
	(148/255, 103/255, 189/255), (140/255, 86/255, 75/255),
	(227/255, 119/255, 194/255), (127/255, 127/255, 127/255)]


# =============================================================================
# Plot the linearizing encoding correlation results
# =============================================================================
# Load the results

if within:
	subjects = 'within'
	pretrained = True
	layers = 'all'
	n_components = 1000
	dnns = ['alexnet']
	dnn_names = ['AlexNet']
	results = []
	for d in dnns:
		data_dir = os.path.join(args.project_dir, 'results', 'stats', 'correlation',
			'encoding-linearizing', 'subjects-'+subjects, 'dnn-'+d, 'pretrained-'+
			str(pretrained), 'layers-'+layers, 'n_components-'+
			format(n_components,'05'), 'correlation_stats.npy')
		results.append(np.load(data_dir, allow_pickle=True).item())
	times = results[0]['times']
	ch_names = results[0]['ch_names']

	# Organize the significance values for plotting
	sig = np.zeros((len(dnns),len(results[0]['significance']['all_layers'])))
	for d in range(sig.shape[0]):
		for t in range(sig.shape[1]):
			if results[d]['significance']['all_layers'][t] == False:
				sig[d,t] = -100
			else:
				sig[d,t] = -.085 + (abs(d+1-len(dnns)) / 100 * 1.75)
	results_baseline = results
	sig_baseline = sig

	# Plot the correlation results, averaged across subjects
	plt.figure(figsize=(4,3))
	# Plot the chance and stimulus onset dashed lines
	plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [10, -10], 'k--',
		label='_nolegend_')
	# Plot the correlation results
	plt.plot(times, np.mean(np.mean(results[0]['correlation']['all_layers'], 0),
		0), color=colors[0])
	# Plot the confidence intervals
	plt.fill_between(times, results[d]['ci_upper']['all_layers'],
		results[0]['ci_lower']['all_layers'], color=colors[0], alpha=.2)
	# Plot the significance markers
	plt.plot(times, sig[0], 'o', color=colors[0], markersize=2)
	# Plot parameters
	plt.xlabel('Time (s)')
	xticks = [-.1, 0, .1, .2, .3, max(times)]
	xlabels = [-0.1, 0, 0.1, 0.2, 0.3, 0.4]
	plt.xticks(ticks=xticks, labels=xlabels)
	plt.xlim(left=min(times), right=max(times))
	plt.ylabel('Pearson\'s $r$')
	yticks = np.arange(0, 1.01, 0.2)
	ylabels = [0, 0.2, 0.4, 0.6, 0.8, 1]
	plt.yticks(ticks=yticks, labels=ylabels)
	plt.ylim(bottom=-.116, top=1)
	#plt.legend(dnn_names, ncol=2, frameon=False)
	plt.savefig(fig_dir+"averaged_within.png", bbox_inches='tight', dpi=300)

	# Plot the single subjects correlation results
	fig, axs = plt.subplots(8, 4, figsize=(16, 32))
	axs = np.reshape(axs, (-1))
	for s in range(len(results[0]['correlation']['all_layers'])):
		# Plot the chance and stimulus onset dashed lines
		axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--')
		# Plot the correlation results
		axs[s].plot(times, np.mean(results[0]['correlation']['all_layers'][s],0), color=colors[0])
		# Plot parameters
		if s in [28, 29, 30, 31]:
			axs[s].set_xlabel('Time (s)')
		xticks = [-.1, 0, .1, .2, .3, max(times)]
		xlabels = [-0.1, 0, 0.1, 0.2, 0.3, 0.4]
		axs[s].set_xticks(ticks=xticks, labels=xlabels)
		if s in [0, 4, 8, 12, 16, 20, 24, 28]:
			axs[s].set_ylabel('Pearson\'s $r$')
		yticks = [0, 0.2, 0.4, 0.6]
		ylabels = [0, 0.2, 0.4, 0.6]
		axs[s].set_yticks(ticks=yticks, labels=ylabels)
		axs[s].set_xlim(left=min(times), right=max(times))
		axs[s].set_ylim(bottom=-.1, top=0.61)
		tit = 'Participant ' + str(all_subs[s])
		axs[s].set_title(tit)
	plt.savefig(fig_dir+"single_within.png", bbox_inches='tight', dpi=300)


	# Plot the single-channel correlation results, averaged across subjects
	fig, axs = plt.subplots(1, 1)
	axs = np.reshape(axs, (-1))
	d = 0
	img = axs[d].imshow(np.mean(results[d]['correlation']['all_layers'], 0),
		aspect='auto')
	# Plot parameters
	axs[d].set_xlabel('Time (s)')
	xticks = np.linspace(0, 100, 6)
	xlabels = [-0.1, 0, 0.1, 0.2, 0.3, 0.4]
	plt.xticks(ticks=xticks, labels=xlabels)
	axs[d].set_ylabel('Channels')
	yticks = np.arange(0, len(ch_names))
	plt.yticks(ticks=yticks, labels=ch_names)
	plt.colorbar(img, label='Pearson\'s $r$', fraction=0.2, ax=axs[d])
	plt.savefig(fig_dir+"ch_corr_avg_within.png", bbox_inches='tight', dpi=300)

# =============================================================================
# Plot the between subjects linearizing encoding correlation results
# =============================================================================
# Load the results

if between:
	subjects = 'between'
	pretrained = True
	layers = 'all'
	n_components = 1000
	results = []
	for d in dnns:
		data_dir = os.path.join(args.project_dir, 'results', 'stats', 'correlation',
			'encoding-linearizing', 'subjects-'+subjects, 'dnn-'+d, 'pretrained-'+
			str(pretrained), 'layers-'+layers, 'n_components-'+
			format(n_components,'05'), 'correlation_stats.npy')
		results.append(np.load(data_dir, allow_pickle=True).item())
	# Organize the significance values for plotting
	sig = np.zeros((len(dnns),len(results[0]['significance']['all_layers'])))
	for d in range(sig.shape[0]):
		for t in range(sig.shape[1]):
			if results[d]['significance']['all_layers'][t] == False:
				sig[d,t] = -100
			else:
				sig[d,t] = -.085 + (abs(d+1-len(dnns)) / 100 * 1.75)

	# Plot the correlation results, averaged across subjects
	plt.figure(figsize=(4,3))
	# Plot the chance and stimulus onset dashed lines
	plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [10, -10], 'k--',
		label='_nolegend_')
	# Plot the correlation results
	plt.plot(times, np.mean(np.mean(results[0]['correlation']['all_layers'], 0),
		0), color=colors[0])
	# Plot the confidence intervals
	plt.fill_between(times, results[d]['ci_upper']['all_layers'],
		results[0]['ci_lower']['all_layers'], color=colors[0], alpha=.2)
	# Plot the significance markers
	plt.plot(times, sig[0], 'o', color=colors[0], markersize=2)
	# Plot parameters
	plt.xlabel('Time (s)')
	xticks = [-.1, 0, .1, .2, .3, max(times)]
	xlabels = [-0.1, 0, 0.1, 0.2, 0.3, 0.4]
	plt.xticks(ticks=xticks, labels=xlabels)
	plt.xlim(left=min(times), right=max(times))
	plt.ylabel('Pearson\'s $r$')
	yticks = np.arange(0, 1.01, 0.2)
	ylabels = [0, 0.2, 0.4, 0.6, 0.8, 1]
	plt.yticks(ticks=yticks, labels=ylabels)
	plt.ylim(bottom=-.116, top=1)
	#plt.legend(dnn_names, ncol=2, frameon=False)
	plt.savefig(fig_dir+"averaged_between.png", bbox_inches='tight', dpi=300)

	# Plot the single subjects correlation results
	fig, axs = plt.subplots(8, 4, figsize=(16, 32))
	axs = np.reshape(axs, (-1))
	for s in range(len(results[0]['correlation']['all_layers'])):
		# Plot the chance and stimulus onset dashed lines
		axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--')
		# Plot the correlation results
		axs[s].plot(times, np.mean(results[0]['correlation']['all_layers'][s],0), color=colors[0])
		# Plot parameters
		if s in [28, 29, 30, 31]:
			axs[s].set_xlabel('Time (s)')
		xticks = [-.1, 0, .1, .2, .3, max(times)]
		xlabels = [-0.1, 0, 0.1, 0.2, 0.3, 0.4]
		axs[s].set_xticks(ticks=xticks, labels=xlabels)
		if s in [0, 4, 8, 12, 16, 20, 24, 28]:
			axs[s].set_ylabel('Pearson\'s $r$')
		yticks = [0, 0.2, 0.4, 0.6]
		ylabels = [0, 0.2, 0.4, 0.6]
		axs[s].set_yticks(ticks=yticks, labels=ylabels)
		axs[s].set_xlim(left=min(times), right=max(times))
		axs[s].set_ylim(bottom=-.1, top=0.61)
		tit = 'Participant ' + str(all_subs[s])
		axs[s].set_title(tit)
	plt.savefig(fig_dir+"single_between.png", bbox_inches='tight', dpi=300)

	# Plot the single-channel correlation results, averaged across subjects
	fig, axs = plt.subplots(1, 1)
	axs = np.reshape(axs, (-1))
	d = 0
	img = axs[d].imshow(np.mean(results[d]['correlation']['all_layers'], 0),
		aspect='auto')
	# Plot parameters
	axs[d].set_xlabel('Time (s)')
	xticks = np.linspace(0, 100, 6)
	xlabels = [-0.1, 0, 0.1, 0.2, 0.3, 0.4]
	plt.xticks(ticks=xticks, labels=xlabels)
	axs[d].set_ylabel('Channels')
	yticks = np.arange(0, len(ch_names))
	plt.yticks(ticks=yticks, labels=ch_names)
	plt.colorbar(img, label='Pearson\'s $r$', fraction=0.2, ax=axs[d])
	plt.savefig(fig_dir+"ch_corr_avg_between.png", bbox_inches='tight', dpi=300)

# =============================================================================
# Plot the end-to-end encoding correlation results
# =============================================================================
# Load the results
if end_to_end:
	modeled_time_points = ['all']
	pretrained = False
	lr = 1e-05
	weight_decay = 0.
	batch_size = 64
	results = []
	for m in modeled_time_points:
		data_dir = os.path.join(args.project_dir, 'results', 'stats',
			'correlation', 'encoding-end_to_end', 'dnn-alexnet',
			'modeled_time_points-'+m, 'pretrained-'+
			str(pretrained), 'lr-{:.0e}'.format(lr)+
			'__wd-{:.0e}'.format(weight_decay)+'__bs-'+
			format(batch_size,'03'), 'correlation_stats.npy')
		results.append(np.load(data_dir, allow_pickle=True).item())
	# Organize the significance values for plotting
	sig = np.zeros((len(modeled_time_points),
		len(results[0]['significance']['all_time_points'])))
	for m, model in enumerate(modeled_time_points):
		for t in range(sig.shape[1]):
			if results[m]['significance'][model+'_time_points'][t] == False:
				sig[m,t] = -100
			else:
				sig[m,t] = -.085 + (abs(m+4.25-len(modeled_time_points)) / 100 * 1.75)

	# Plot the correlation results, averaged across subjects
	plt.figure(figsize=(4,3))
	# Plot the chance and stimulus onset dashed lines
	plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [10, -10], 'k--',
		label='_nolegend_')
	for m, model in enumerate(modeled_time_points):
		# Plot the correlation results
		plt.plot(times, np.mean(np.mean(
			results[m]['correlation'][model+'_time_points'], 0), 0),
			color=colors[m])
	for m, model in enumerate(modeled_time_points):
		# Plot the confidence intervals
		plt.fill_between(times, results[m]['ci_upper'][model+'_time_points'],
			results[m]['ci_lower'][model+'_time_points'], color=colors[m],
			alpha=.2)
		# Plot the significance markers
		plt.plot(times, sig[m], 'o', color=colors[m], markersize=2)
	# Plot parameters
	plt.xlabel('Time (s)')
	xticks = [-.1, 0, .1, .2, .3, max(times)]
	xlabels = [-0.1, 0, 0.1, 0.2, 0.3, 0.4]
	plt.xticks(ticks=xticks, labels=xlabels)
	plt.xlim(left=min(times), right=max(times))
	plt.ylabel('Pearson\'s $r$')
	yticks = np.arange(0,1.01,0.2)
	ylabels = [0, 0.2, 0.4, 0.6, 0.8, 1]
	plt.yticks(ticks=yticks, labels=ylabels)
	plt.ylim(bottom=-.116, top=1)
	plt.savefig(fig_dir+"averaged_within_e2e.png", bbox_inches='tight', dpi=300)

	# Plot the single subjects correlation results
	fig, axs = plt.subplots(8, 4, figsize=(16, 32))
	axs = np.reshape(axs, (-1))
	for s in range(len(results[0]['correlation']['all_time_points'])):
		# Plot the correlation results
		for m, model in enumerate(modeled_time_points):
			axs[s].plot(times,
				np.mean(results[m]['correlation'][model+'_time_points'][s], 0), color=colors[m])
		# Plot the chance and stimulus onset dashed lines
		axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--')
		# Plot parameters
		if s in [28, 29, 30, 31]:
			axs[s].set_xlabel('Time (s)')
		xticks = [-.1, 0, .1, .2, .3, max(times)]
		xlabels = [-0.1, 0, 0.1, 0.2, 0.3, 0.4]
		axs[s].set_xticks(ticks=xticks, labels=xlabels)
		if s in [0, 4, 8, 12, 16, 20, 24, 28]:
			axs[s].set_ylabel('Pearson\'s $r$')
		yticks = [0, 0.2, 0.4, 0.6]
		ylabels = [0, 0.2, 0.4, 0.6]
		axs[s].set_yticks(ticks=yticks, labels=ylabels)
		axs[s].set_xlim(left=min(times), right=max(times))
		axs[s].set_ylim(bottom=-.1, top=0.61)
		tit = 'Participant ' + str(all_subs[s])
		axs[s].set_title(tit)
	plt.savefig(fig_dir+"single_within_e2e.png", bbox_inches='tight', dpi=300)


	# Plot the single-channel correlation results, averaged across subjects
	fig, axs = plt.subplots(1, 1)
	axs = np.reshape(axs, (-1))
	m = 0
	img = axs[m].imshow(np.mean(
		results[m]['correlation'][model+'_time_points'], 0), aspect='auto')
	# Plot parameters
	axs[m].set_xlabel('Time (s)')
	xticks = [0, 20, 40, 60, 80, 99]
	xlabels = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
	plt.xticks(ticks=xticks, labels=xlabels)
	axs[m].set_ylabel('Channels')
	yticks = np.arange(0, len(ch_names))
	plt.yticks(ticks=yticks, labels=ch_names)
	plt.colorbar(img, label='Pearson\'s $r$', fraction=0.2, ax=axs[m])
	plt.savefig(fig_dir+'ch_corr_avg_e2e.png', bbox_inches='tight', dpi=300)
