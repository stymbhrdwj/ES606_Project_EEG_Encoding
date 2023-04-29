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
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
colors = [(31/255, 119/255, 180/255), (255/255, 127/255, 14/255),
	(44/255, 160/255, 44/255), (214/255, 39/255, 40/255),
	(148/255, 103/255, 189/255), (140/255, 86/255, 75/255),
	(227/255, 119/255, 194/255), (127/255, 127/255, 127/255)]


# =============================================================================
# Plot the linearizing encoding correlation results
# =============================================================================

print("Plot the linearizing encoding correlation results for within subjects...")
# Load the results
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
within_corr = results[0]['correlation']['all_layers']
plt.plot(times, np.mean(np.mean(within_corr, 0), 0), color=colors[0])
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
for s in range(len(within_corr)):
	# Plot the chance and stimulus onset dashed lines
	axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--')
	# Plot the correlation results
	axs[s].plot(times, np.mean(within_corr[s],0), color=colors[0])
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

print("Plot the linearizing encoding correlation results for between subjects...")
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
between_corr = results[0]['correlation']['all_layers']
plt.plot(times, np.mean(np.mean(between_corr, 0), 0), color=colors[0])
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
for s in range(len(between_corr)):
	# Plot the chance and stimulus onset dashed lines
	axs[s].plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--')
	# Plot the correlation results
	axs[s].plot(times, np.mean(between_corr[s],0), color=colors[0])
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

print("Plot the end-to-end encoding correlation results for within subjects...")

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
plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [10, -10], 'k--', label='_nolegend_')
# Plot the correlation results
within_corr_e2e = results[0]['correlation']['all_time_points']
plt.plot(times, np.mean(np.mean(within_corr_e2e, 0), 0), color=colors[0])
# Plot the confidence intervals
plt.fill_between(times, results[0]['ci_upper']['all_time_points'],
				 results[0]['ci_lower']['all_time_points'], color=colors[0], alpha=.2)
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
for s in range(len(within_corr_e2e)):
	# Plot the correlation results
	axs[s].plot(times, np.mean(within_corr_e2e[s], 0), color=colors[0])
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

# =============================================================================
# Compare the correlation results
# =============================================================================

print("Plot the comparisons...")

plt.figure(figsize=(5,4))
# Plot the chance and stimulus onset dashed lines
plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [10, -10], 'k--', label='_nolegend_')
# Compare the correlation, within vs between for linearizing encoding
plt.plot(times, np.mean(np.mean(within_corr, 0), 0), color=colors[0], linewidth=1)
plt.plot(times, np.mean(np.mean(between_corr, 0), 0), color=colors[1], linewidth=1)
plt.plot(times, np.mean(np.mean(within_corr_e2e, 0), 0), color=colors[2], linewidth=1)
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
plt.legend(["linear (within)", "linear (between)", "end-to-end (within)"])
plt.savefig(fig_dir+"avg_comparison.png", bbox_inches='tight', dpi=300)

# Compare the single subjects correlation results
fig, axs = plt.subplots(8, 4, figsize=(16, 32))
axs = np.reshape(axs, (-1))
for s in range(len(within_corr_e2e)):
	# Plot the correlation results
	axs[s].plot(times, np.mean(within_corr[s], 0), color=colors[0])
	axs[s].plot(times, np.mean(between_corr[s], 0), color=colors[1])
	axs[s].plot(times, np.mean(within_corr_e2e[s], 0), color=colors[2])
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
plt.savefig(fig_dir+"single_comparison.png", bbox_inches='tight', dpi=300)

# =============================================================================
# Save the peak correlation data in a csv
# =============================================================================

import pandas as pd
peak = np.zeros((len(all_subs) + 1, 6))

peak[:-1, 0] = within_corr.mean(1).max(1)
peak[:-1, 1] = times[within_corr.mean(1).argmax(1)]
peak[:-1, 2] = between_corr.mean(1).max(1)
peak[:-1, 3] = times[between_corr.mean(1).argmax(1)]
peak[:-1, 4] = within_corr_e2e.mean(1).max(1)
peak[:-1, 5] = times[within_corr_e2e.mean(1).argmax(1)]

peak[-1, 0] = within_corr.mean(0).mean(0).max()
peak[-1, 1] = times[within_corr.mean(0).mean(0).argmax()]
peak[-1, 2] = between_corr.mean(0).mean(0).max()
peak[-1, 3] = times[between_corr.mean(0).mean(0).argmax()]
peak[-1, 4] = within_corr_e2e.mean(0).mean(0).max()
peak[-1, 5] = times[within_corr_e2e.mean(0).mean(0).argmax()]

df = pd.DataFrame({'lin_peak_corr (within)':peak[:,0],
				   'lin_peak_time (within)':peak[:,1],
				   'lin_peak_corr (between)':peak[:,2],
				   'lin_peak_time (between)':peak[:,3],
				   'e2e_peak_corr (within)':peak[:,4],
				   'e2e_peak_time (within)':peak[:,5]}, index=all_subs+['avg'])

df.to_csv('peaks.csv')
