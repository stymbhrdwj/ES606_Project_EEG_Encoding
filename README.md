# ES606_Project_EEG_Encoding

Applying the linearizing encoding model and end-to-end AlexNet on the THINGS EEG1 dataset. 

### Dataset reference: https://osf.io/hd6zk/

Grootswagers, T., Zhou, I., Robinson, A.K. et al. Human EEG recordings for 1,854 concepts presented in rapid serial visual presentation streams. Sci Data 9, 3 (2022). https://doi.org/10.1038/s41597-021-01102-7

### Code reference: https://github.com/gifale95/eeg_encoding/

Gifford AT, Dwivedi K, Roig G, Cichy RM. 2022. A large and rich EEG dataset for modeling human visual object recognition. NeuroImage, 264:119754. DOI: https://doi.org/10.1016/j.neuroimage.2022.119754

The code has been adjusted as per this dataset, and assumes the following directory structure:

```
├── code
│   ├── 00_make_numpy_arrays.py
│   ├── 01_plot_ERPs.py
│   ├── 10_extract_feature_maps_alexnet.py
│   ├── 11_feature_maps_pca.py
│   ├── 20_linearizing_encoding.py
│   ├── 21_end_to_end_encoding.py
│   ├── 30_correlation.py
│   ├── 31_correlation_stats.py
│   ├── 32_plot_correlation.py
│   ├── LICENSE
│   ├── README.md
│   └── utils
│       ├── end_to_end_encoding_utils.py
│       ├── linearizing_encoding_utils.py
│       └── ols.py
└── data
    ├── dnn_feature_maps
    │   ├── full_feature_maps
    │   └── pca_feature_maps
    ├── plots
    │   ├── correlation
    │   └── erps
    ├── preprocessed_data
    │   ├── sub-02
    │   ├── sub-03
    │   ├── sub-04
    │   ├── sub-05
    │   ├── sub-07
    │   ├── sub-08
    │   ├── sub-09
    │   ├── sub-10
    │   ├── sub-11
    │   ├── sub-12
    │   ├── sub-13
    │   ├── sub-14
    │   ├── sub-15
    │   ├── sub-16
    │   ├── sub-17
    │   ├── sub-19
    │   ├── sub-20
    │   ├── sub-21
    │   ├── sub-22
    │   ├── sub-24
    │   ├── sub-25
    │   ├── sub-26
    │   ├── sub-27
    │   ├── sub-28
    │   ├── sub-29
    │   ├── sub-30
    │   ├── sub-31
    │   ├── sub-32
    │   ├── sub-33
    │   ├── sub-34
    │   ├── sub-35
    │   ├── sub-36
    │   ├── sub-37
    │   ├── sub-38
    │   ├── sub-39
    │   ├── sub-40
    │   ├── sub-41
    │   ├── sub-42
    │   ├── sub-43
    │   ├── sub-44
    │   ├── sub-45
    │   ├── sub-46
    │   ├── sub-47
    │   └── sub-48
    ├── results
    │   ├── stats
    │   ├── sub-02
    │   ├── sub-03
    │   ├── sub-04
    │   ├── sub-05
    │   ├── sub-07
    │   ├── sub-08
    │   ├── sub-09
    │   ├── sub-10
    │   ├── sub-14
    │   ├── sub-16
    │   ├── sub-21
    │   ├── sub-22
    │   ├── sub-24
    │   ├── sub-26
    │   ├── sub-27
    │   ├── sub-28
    │   ├── sub-29
    │   ├── sub-30
    │   ├── sub-31
    │   ├── sub-33
    │   ├── sub-34
    │   ├── sub-35
    │   ├── sub-37
    │   ├── sub-38
    │   ├── sub-40
    │   ├── sub-41
    │   ├── sub-42
    │   ├── sub-43
    │   ├── sub-44
    │   ├── sub-45
    │   ├── sub-46
    │   └── sub-47
    ├── source
    │   ├── sub-01_task-rsvp_continuous.fdt
    │   ├── sub-01_task-rsvp_continuous.set
    │   ├── sub-01_task-rsvp_events.csv
    │   ├── sub-02_task-rsvp_continuous.fdt
    │   ├── sub-02_task-rsvp_continuous.set
    │   ├── sub-02_task-rsvp_events.csv
    │   ├── sub-03_task-rsvp_continuous.fdt
    │   ├── sub-03_task-rsvp_continuous.set
    │   ├── sub-03_task-rsvp_events.csv
    │   ├── sub-04_task-rsvp_continuous.fdt
    │   ├── sub-04_task-rsvp_continuous.set
    │   ├── sub-04_task-rsvp_events.csv
    │   ├── sub-05_task-rsvp_continuous.fdt
    │   ├── sub-05_task-rsvp_continuous.set
    │   ├── sub-05_task-rsvp_events.csv
    │   ├── sub-06_task-rsvp_continuous.fdt
    │   ├── sub-06_task-rsvp_continuous.set
    │   ├── sub-06_task-rsvp_events.csv
    │   ├── sub-07_task-rsvp_continuous.fdt
    │   ├── sub-07_task-rsvp_continuous.set
    │   ├── sub-07_task-rsvp_events.csv
    │   ├── sub-08_task-rsvp_continuous.fdt
    │   ├── sub-08_task-rsvp_continuous.set
    │   ├── sub-08_task-rsvp_events.csv
    │   ├── sub-09_task-rsvp_continuous.fdt
    │   ├── sub-09_task-rsvp_continuous.set
    │   ├── sub-09_task-rsvp_events.csv
    │   ├── sub-10_task-rsvp_continuous.fdt
    │   ├── sub-10_task-rsvp_continuous.set
    │   ├── sub-10_task-rsvp_events.csv
    │   ├── sub-11_task-rsvp_continuous.fdt
    │   ├── sub-11_task-rsvp_continuous.set
    │   ├── sub-11_task-rsvp_events.csv
    │   ├── sub-12_task-rsvp_continuous.fdt
    │   ├── sub-12_task-rsvp_continuous.set
    │   ├── sub-12_task-rsvp_events.csv
    │   ├── sub-13_task-rsvp_continuous.fdt
    │   ├── sub-13_task-rsvp_continuous.set
    │   ├── sub-13_task-rsvp_events.csv
    │   ├── sub-14_task-rsvp_continuous.fdt
    │   ├── sub-14_task-rsvp_continuous.set
    │   ├── sub-14_task-rsvp_events.csv
    │   ├── sub-15_task-rsvp_continuous.fdt
    │   ├── sub-15_task-rsvp_continuous.set
    │   ├── sub-15_task-rsvp_events.csv
    │   ├── sub-16_task-rsvp_continuous.fdt
    │   ├── sub-16_task-rsvp_continuous.set
    │   ├── sub-16_task-rsvp_events.csv
    │   ├── sub-17_task-rsvp_continuous.fdt
    │   ├── sub-17_task-rsvp_continuous.set
    │   ├── sub-17_task-rsvp_events.csv
    │   ├── sub-18_task-rsvp_continuous.fdt
    │   ├── sub-18_task-rsvp_continuous.set
    │   ├── sub-18_task-rsvp_events.csv
    │   ├── sub-19_task-rsvp_continuous.fdt
    │   ├── sub-19_task-rsvp_continuous.set
    │   ├── sub-19_task-rsvp_events.csv
    │   ├── sub-20_task-rsvp_continuous.fdt
    │   ├── sub-20_task-rsvp_continuous.set
    │   ├── sub-20_task-rsvp_events.csv
    │   ├── sub-21_task-rsvp_continuous.fdt
    │   ├── sub-21_task-rsvp_continuous.set
    │   ├── sub-21_task-rsvp_events.csv
    │   ├── sub-22_task-rsvp_continuous.fdt
    │   ├── sub-22_task-rsvp_continuous.set
    │   ├── sub-22_task-rsvp_events.csv
    │   ├── sub-23_task-rsvp_continuous.fdt
    │   ├── sub-23_task-rsvp_continuous.set
    │   ├── sub-23_task-rsvp_events.csv
    │   ├── sub-24_task-rsvp_continuous.fdt
    │   ├── sub-24_task-rsvp_continuous.set
    │   ├── sub-24_task-rsvp_events.csv
    │   ├── sub-25_task-rsvp_continuous.fdt
    │   ├── sub-25_task-rsvp_continuous.set
    │   ├── sub-25_task-rsvp_events.csv
    │   ├── sub-26_task-rsvp_continuous.fdt
    │   ├── sub-26_task-rsvp_continuous.set
    │   ├── sub-26_task-rsvp_events.csv
    │   ├── sub-27_task-rsvp_continuous.fdt
    │   ├── sub-27_task-rsvp_continuous.set
    │   ├── sub-27_task-rsvp_events.csv
    │   ├── sub-28_task-rsvp_continuous.fdt
    │   ├── sub-28_task-rsvp_continuous.set
    │   ├── sub-28_task-rsvp_events.csv
    │   ├── sub-29_task-rsvp_continuous.fdt
    │   ├── sub-29_task-rsvp_continuous.set
    │   ├── sub-29_task-rsvp_events.csv
    │   ├── sub-30_task-rsvp_continuous.fdt
    │   ├── sub-30_task-rsvp_continuous.set
    │   ├── sub-30_task-rsvp_events.csv
    │   ├── sub-31_task-rsvp_continuous.fdt
    │   ├── sub-31_task-rsvp_continuous.set
    │   ├── sub-31_task-rsvp_events.csv
    │   ├── sub-32_task-rsvp_continuous.fdt
    │   ├── sub-32_task-rsvp_continuous.set
    │   ├── sub-32_task-rsvp_events.csv
    │   ├── sub-33_task-rsvp_continuous.fdt
    │   ├── sub-33_task-rsvp_continuous.set
    │   ├── sub-33_task-rsvp_events.csv
    │   ├── sub-34_task-rsvp_continuous.fdt
    │   ├── sub-34_task-rsvp_continuous.set
    │   ├── sub-34_task-rsvp_events.csv
    │   ├── sub-35_task-rsvp_continuous.fdt
    │   ├── sub-35_task-rsvp_continuous.set
    │   ├── sub-35_task-rsvp_events.csv
    │   ├── sub-36_task-rsvp_continuous.fdt
    │   ├── sub-36_task-rsvp_continuous.set
    │   ├── sub-36_task-rsvp_events.csv
    │   ├── sub-37_task-rsvp_continuous.fdt
    │   ├── sub-37_task-rsvp_continuous.set
    │   ├── sub-37_task-rsvp_events.csv
    │   ├── sub-38_task-rsvp_continuous.fdt
    │   ├── sub-38_task-rsvp_continuous.set
    │   ├── sub-38_task-rsvp_events.csv
    │   ├── sub-39_task-rsvp_continuous.fdt
    │   ├── sub-39_task-rsvp_continuous.set
    │   ├── sub-39_task-rsvp_events.csv
    │   ├── sub-40_task-rsvp_continuous.fdt
    │   ├── sub-40_task-rsvp_continuous.set
    │   ├── sub-40_task-rsvp_events.csv
    │   ├── sub-41_task-rsvp_continuous.fdt
    │   ├── sub-41_task-rsvp_continuous.set
    │   ├── sub-41_task-rsvp_events.csv
    │   ├── sub-42_task-rsvp_continuous.fdt
    │   ├── sub-42_task-rsvp_continuous.set
    │   ├── sub-42_task-rsvp_events.csv
    │   ├── sub-43_task-rsvp_continuous.fdt
    │   ├── sub-43_task-rsvp_continuous.set
    │   ├── sub-43_task-rsvp_events.csv
    │   ├── sub-44_task-rsvp_continuous.fdt
    │   ├── sub-44_task-rsvp_continuous.set
    │   ├── sub-44_task-rsvp_events.csv
    │   ├── sub-45_task-rsvp_continuous.fdt
    │   ├── sub-45_task-rsvp_continuous.set
    │   ├── sub-45_task-rsvp_events.csv
    │   ├── sub-46_task-rsvp_continuous.fdt
    │   ├── sub-46_task-rsvp_continuous.set
    │   ├── sub-46_task-rsvp_events.csv
    │   ├── sub-47_task-rsvp_continuous.fdt
    │   ├── sub-47_task-rsvp_continuous.set
    │   ├── sub-47_task-rsvp_events.csv
    │   ├── sub-48_task-rsvp_continuous.fdt
    │   ├── sub-48_task-rsvp_continuous.set
    │   ├── sub-48_task-rsvp_events.csv
    │   ├── sub-49_task-rsvp_continuous.fdt
    │   ├── sub-49_task-rsvp_continuous.set
    │   ├── sub-49_task-rsvp_events.csv
    │   ├── sub-50_task-rsvp_continuous.fdt
    │   ├── sub-50_task-rsvp_continuous.set
    │   └── sub-50_task-rsvp_events.csv
    └── stimuli
        ├── test_images
        └── training_images
```

## 0. Prepare the dataset

The first step is to transform the given EEGLAB format data into numpy arrays. Then visualize the ERPs and reject the participants whose ERPs show large spikes and artifacts. See `plots/erps/reject` for the rejected participants.

## 1. Extract feature maps 

We use a pretrained AlexNet to extract feature maps from the images in the THINGS image dataset. Then we perform a kernel PCA with a polynomial kernel to extract the 1000 most important components of our data.

## 2. Training DNNs

As a first step, we train a linearizing encoding model, essentially a linear regression between the KPCA'd feature maps and the EEG data for each channel and time-point. We also try training a randomly initalized AlexNet in an end-to-end fashion to synthesize the EEG visual response when fed in the corresponding image.

## 3. Correlation analysis

As a measure of how well our models synthesize the EEG data, we perform a correlation analysis using Pearson's-r. We observe significant correlation  above chance level for both within and between subjects model training. Curiously, the end-to-end encoding model shows a poorer correlation compared to the linearizing encoding model, which is contrary to what Gifford et al. report. This could be explained by the fact that their dataset (THINGS EEG2) contains many more trial repetitions (4 for train, 80 for test) compared to THINGS EEG1 (1 for train, 12 for test). As a result of the fewer test repetitions, we skip the noise ceiling analysis.

### Average correlation for linearizing encoding model (within subjects)

<img src="plots/correlation/averaged_within.png" width=400>

### Average correlation for linearizing encoding model (between subjects)

<img src="plots/correlation/averaged_between.png" width=400>

### Average correlation for end-to-end AlexNet encoding model (within subjects)

<img src="plots/correlation/averaged_within_e2e.png" width=400>

### Single participant correlation for linearizing encoding model (within subjects)

<img src="plots/correlation/single_within.png" width=1000>

### Single participant correlation for linearizing encoding model (between subjects)

<img src="plots/correlation/single_between.png" width=1000>

### Single participant correlation for end-to-end AlexNet encoding model (within subjects)

<img src="plots/correlation/single_within_e2e.png" width=1000>

### Single channel correlation for linearizing encoding model (within subjects)

<img src="plots/correlation/ch_corr_avg_within.png" width=600>

### Single channel correlation for linearizing encoding model (between subjects)

<img src="plots/correlation/ch_corr_avg_between.png" width=600>

### Single channel correlation for end-to-end AlexNet encoding model (within subjects)

<img src="plots/correlation/ch_corr_avg_e2e.png" width=600>
