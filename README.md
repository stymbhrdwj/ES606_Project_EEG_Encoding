# ES606_Project_EEG_Encoding

Applying the linearizing encoding model and end-to-end AlexNet by Gifford et al. on the THINGS EEG1 dataset by Grootswagers et al.

### Dataset reference: https://osf.io/hd6zk/

Grootswagers, T., Zhou, I., Robinson, A.K. et al. Human EEG recordings for 1,854 concepts presented in rapid serial visual presentation streams. Sci Data 9, 3 (2022). https://doi.org/10.1038/s41597-021-01102-7

### Code reference: https://github.com/gifale95/eeg_encoding/

Gifford AT, Dwivedi K, Roig G, Cichy RM. 2022. A large and rich EEG dataset for modeling human visual object recognition. NeuroImage, 264:119754. DOI: https://doi.org/10.1016/j.neuroimage.2022.119754

The code has been modified extensively for this dataset, and assumes the directory structure given in `tree.txt`.

## 0. Prepare the dataset

The first step is to transform the given EEGLAB format data into numpy arrays. Then visualize the ERPs and reject the participants whose ERPs show large spikes and artifacts. See `plots/erps/reject` for the rejected participants.

## 1. Extract DNN feature maps 

We use a pretrained AlexNet to extract feature maps from the images in the THINGS image dataset. Then we perform a kernel PCA with a polynomial kernel to extract the 1000 most important components of our data.

## 2. Training

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
