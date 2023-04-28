"""PCA is performed on the DNN feature maps to reduce their dimensionality.
PCA is applied on either the feature maps of single DNN layers, or on the
appended feature maps of all layers.
Before applying PCA on the CORnet-S feature maps, run
'sort_feature_maps_cornet_s.py'.

Parameters
----------
dnn : str
	Used DNN among 'alexnet', 'resnet50', 'cornet_s', 'moco'.
pretrained : bool
	If True use the pretrained network feature maps, if False use the randomly
	initialized network feature maps.
layers : str
	Whether to use 'all' or 'single' layers.
n_components : int
	Number of DNN feature maps PCA components retained.
project_dir : str
	Directory of the project folder.

"""

import argparse
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from tqdm.auto import tqdm


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dnn', default='alexnet', type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layers', default='single', type=str)
parser.add_argument('--n_components', default=1000, type=int)
parser.add_argument('--project_dir', default='../../', type=str)
parser.add_argument('--imagenet', default=False, type=bool)
args = parser.parse_args()

print('>>> Apply PCA on the feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

# =============================================================================
# Apply PCA on the training images feature maps
# =============================================================================
# The standardization and PCA statistics computed on the training images feature
# maps are also applied to the test images feature maps and to the ILSVRC-2012
# images feature maps.

print("Apply PCA on the training images feature maps")

# Load the feature maps
print("Loading the feature maps...")
if args.layers == 'single':
	feats = []
feats_all = []
fmaps_train = {}
fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'full_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained),
	'training_images')
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for f, fmaps in tqdm(enumerate(fmaps_list), total=len(fmaps_list)):
	fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
		allow_pickle=True).item()
	all_layers = fmaps_data.keys()
	if args.layers == 'all':
		layer_names = ['all_layers']
	elif args.layers == 'single':
		layer_names = all_layers
	for l, dnn_layer in enumerate(all_layers):
		if args.layers == 'all':
			if l == 0:
				feats = fmaps_data[dnn_layer].flatten()
			else:
				feats = np.append(feats, fmaps_data[dnn_layer].flatten())
		elif args.layers == 'single':
			if f == 0:
				feats.append([[fmaps_data[dnn_layer].flatten()]])
			else:
				feats[l].append([fmaps_data[dnn_layer].flatten()])
	if args.layers == 'all':
		feats_all.append(feats)

if args.layers == 'all':
	fmaps_train[layer_names[0]] = np.asarray(feats_all)
elif args.layers == 'single':
	for l, dnn_layer in enumerate(layer_names):
		fmaps_train[dnn_layer] = np.squeeze(np.asarray(feats[l]))

# Standardize the data
print("Standarize the data...")
scaler = []
for l, dnn_layer in tqdm(enumerate(layer_names)):
	scaler.append(StandardScaler())
	scaler[l].fit(fmaps_train[dnn_layer])
	fmaps_train[dnn_layer] = scaler[l].transform(fmaps_train[dnn_layer])

# Apply PCA
print("Apply PCA...")
pca = []
for l, dnn_layer in tqdm(enumerate(list(layer_names)), total=len(list(layer_names))):
	pca.append(KernelPCA(n_components=args.n_components, kernel='poly',
		degree=4, random_state=seed, n_jobs=-1))
	pca[l].fit(fmaps_train[dnn_layer])
	fmaps_train[dnn_layer] = pca[l].transform(fmaps_train[dnn_layer])

# Save the downsampled feature maps
print('Save the downsampled feature maps')
save_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'pca_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained), 'layers-'+
	args.layers)
file_name = 'pca_feature_maps_training'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), fmaps_train)
del fmaps_train


# =============================================================================
# Apply PCA on the test images feature maps
# =============================================================================
# Load the feature maps

print("Apply PCA on the test images feature maps")
print("Load the feature maps")

feats = []
feats_all = []
fmaps_test = {}
fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'full_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained),
	'test_images')
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for f, fmaps in enumerate(fmaps_list):
	fmaps_data = np.load(os.path.join(fmaps_dir, fmaps), allow_pickle=True).item()
	all_layers = fmaps_data.keys()
	if args.layers == 'all':
		layer_names = ['all_layers']
	elif args.layers == 'single':
		layer_names = all_layers
	for l, dnn_layer in enumerate(all_layers):
		if args.layers == 'all':
			if l == 0:
				feats = np.reshape(fmaps_data[dnn_layer], -1)
			else:
				feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], -1))
		elif args.layers == 'single':
			if f == 0:
				feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
			else:
				feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
	if args.layers == 'all':
		feats_all.append(feats)
if args.layers == 'all':
	fmaps_test[layer_names[0]] = np.asarray(feats_all)
elif args.layers == 'single':
	for l, dnn_layer in enumerate(layer_names):
		fmaps_test[dnn_layer] = np.squeeze(np.asarray(feats[l]))

# Standardize the data
print("Standarize the data")
for l, dnn_layer in enumerate(layer_names):
	fmaps_test[dnn_layer] = scaler[l].transform(fmaps_test[dnn_layer])

# Apply PCA
print("Apply PCA")
for l, dnn_layer in enumerate(layer_names):
	fmaps_test[dnn_layer] = pca[l].transform(fmaps_test[dnn_layer])

# Save the downsampled feature maps
print("Save the downsampled feature maps")
file_name = 'pca_feature_maps_test'
np.save(os.path.join(save_dir, file_name), fmaps_test)
del fmaps_test

if args.imagenet:

	# =============================================================================
	# Apply PCA on the ILSVRC-2012 validation images feature maps
	# =============================================================================
	# PCA is applied to partitions of 10k images feature maps for memory efficiency.

	# Load the feature maps
	n_img_part = 10000
	fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
		'full_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained),
		'ILSVRC2012_img_val')
	fmaps_list = os.listdir(fmaps_dir)
	fmaps_list.sort()
	fmaps_ilsvrc2012_val = {}
	for p in range(0, len(fmaps_list), n_img_part):
		feats = []
		feats_all = []
		fmaps_part = {}
		for f, fmaps in enumerate(fmaps_list[p:p+n_img_part]):
			fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
				allow_pickle=True).item()
			for l, dnn_layer in enumerate(all_layers):
				if args.layers == 'all':
					if l == 0:
						feats = np.reshape(fmaps_data[dnn_layer], -1)
					else:
						feats = np.append(feats, np.reshape(fmaps_data[dnn_layer],
							-1))
				elif args.layers == 'single':
					if f == 0:
						feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
					else:
						feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
			if args.layers == 'all':
				feats_all.append(feats)
		if args.layers == 'all':
			fmaps_part[layer_names[0]] = np.asarray(feats_all)
		elif args.layers == 'single':
			for l, dnn_layer in enumerate(layer_names):
				fmaps_part[dnn_layer] = np.squeeze(np.asarray(feats[l]))

		# Standardize the data
		for l, dnn_layer in enumerate(layer_names):
			fmaps_part[dnn_layer] = scaler[l].transform(fmaps_part[dnn_layer])

		# Apply PCA
		for l, dnn_layer in enumerate(layer_names):
			if p == 0:
				fmaps_ilsvrc2012_val[dnn_layer] = pca[l].transform(
					fmaps_part[dnn_layer])
			else:
				fmaps_ilsvrc2012_val[dnn_layer] = np.append(
					fmaps_ilsvrc2012_val[dnn_layer], pca[l].transform(
					fmaps_part[dnn_layer]), 0)

	# Save the downsampled feature maps
	file_name = 'pca_feature_maps_ilsvrc2012_val'
	np.save(os.path.join(save_dir, file_name), fmaps_ilsvrc2012_val)
	del fmaps_ilsvrc2012_val


	# =============================================================================
	# Apply PCA on the ILSVRC-2012 test images feature maps
	# =============================================================================
	# PCA is applied to partitions of 10k images feature maps for memory efficiency.

	# Load the feature maps
	fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
		'full_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained),
		'ILSVRC2012_img_test_v10102019')
	fmaps_list = os.listdir(fmaps_dir)
	fmaps_list.sort()
	fmaps_ilsvrc2012_test = {}
	for p in range(0, len(fmaps_list), n_img_part):
		feats = []
		feats_all = []
		fmaps_part = {}
		for f, fmaps in enumerate(fmaps_list[p:p+n_img_part]):
			fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
				allow_pickle=True).item()
			for l, dnn_layer in enumerate(all_layers):
				if args.layers == 'all':
					if l == 0:
						feats = np.reshape(fmaps_data[dnn_layer], -1)
					else:
						feats = np.append(feats, np.reshape(fmaps_data[dnn_layer],
							-1))
				elif args.layers == 'single':
					if f == 0:
						feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
					else:
						feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
			if args.layers == 'all':
				feats_all.append(feats)
		if args.layers == 'all':
			fmaps_part[layer_names[0]] = np.asarray(feats_all)
		elif args.layers == 'single':
			for l, dnn_layer in enumerate(layer_names):
				fmaps_part[dnn_layer] = np.squeeze(np.asarray(feats[l]))

		# Standardize the data
		for l, dnn_layer in enumerate(layer_names):
			fmaps_part[dnn_layer] = scaler[l].transform(fmaps_part[dnn_layer])

		# Apply PCA
		for l, dnn_layer in enumerate(layer_names):
			if p == 0:
				fmaps_ilsvrc2012_test[dnn_layer] = pca[l].transform(
					fmaps_part[dnn_layer])
			else:
				fmaps_ilsvrc2012_test[dnn_layer] = np.append(
					fmaps_ilsvrc2012_test[dnn_layer], pca[l].transform(
					fmaps_part[dnn_layer]), 0)

	# Save the downsampled feature maps
	file_name = 'pca_feature_maps_ilsvrc2012_test'
	np.save(os.path.join(save_dir, file_name), fmaps_ilsvrc2012_test)
