## Segmentation of cells based on label-free microscopy
 
## Description
This repository contains a TensorFlow implementation of a segmentation algorithm, a modified U-Net achieving accurate segmentation of various cells in a label-free living tumor microenvironment. 

## About
Many advances have been made in the area of microscopy thanks to the rapid development of markers and labels. However, marker-based analyses have fundamental limitations due to its perturbance to the living system. Label-free nonlinear optical microscopy, which produces high-resolution images with rich functional and structural information based on intrinsic molecular contrast, has demonstrated strong potential to overcome these problems by generating a broader array of volumetric signals from tissue structures and molecular composition. Here, we shared a multiclass pixel-level neural network program that segments the major components of the tumor microenvironment, including tumor cells, stromal components (fibroblasts, endothelial cells, lymphocytes, red blood cells, adipocytes), and EVs. 

## Installation
### Requirement:
*   TensorFlow >=1.12.0
*   Skimage

## Usage
### Set directories
*   homedir: home directory
*   logdir: log directory
*   traindir: training directory where the mask files for the training data are saved
*   validationdir: validation directory where files for the validation data were saved
*   testdir: testing direcotry where raw files for testing were saved
*   resultModeldir: save models to this directory
*   resultsImagedir: save validation images to this directory
*   resultsApplydir: save results from applying the model to test images to this directory

### Set parameters:
*   ous: network output size
*   ins: network input size (random cropping size during the training)
*   interv: difference of gray value in mask labeling
*   batch_size: batch size for training
*   iterModel: whether to load previous model or to train from scratch
*   iterMax: maximum iterations to run
*   parameters: saved name including all the parameters
*   imageType: how many modalities
*   nc: how many channels

### Example
#### Training
python3 segmentation.py
#### Testing
python3 segmentation.py -model homedir/resultsModeldir/15000.ckpt -test_dir homedir/testdir\ png/

## Citation
If you find this useful or use this method in your work, please cite: Sixian You, Eric J. Chaney, Haohua Tu, Yi Sun, Saurabh Sinha, Stephen A. Boppart. Label-free deep profiling of the tumor microenvironment. 
