# Back to Basics: Unsupervised Learning of Optical Flow via Brightness Constancy and Motion Smoothness
This repository contains a TensorFlow implementation from the original work: [Back to Basics: Unsupervised Learning of Optical Flow via Brightness Constancy and Motion Smoothness](https://arxiv.org/abs/1608.05842) (ECCV 2016 Workshops, Part 3).
Authors: [Jason J. Yu](http://scs.ryerson.ca/~jjyu/), [Adam W. Harley](https://www.cs.cmu.edu/~aharley/), [Konstantinos G. Derpanis](http://www.scs.ryerson.ca/~kosta/)

### [Project Page](http://scs.ryerson.ca/~jjyu/projects/unsupervised/optical/flow/machine/learning/2016/08/30/Unsup-flow.html)
### [Slides](https://drive.google.com/file/d/0Bz1dfcnrpXM-T1BjU1dhV29wQXM/view)
### Citation
	@inproceedings{
	    jjyu2016unsupflow,
	    author = {Jason J. Yu and Adam W. Harley and Konstantinos G. Derpanis},
	    title = {Back to Basics: Unsupervised Learning of Optical Flow via Brightness Constancy and Motion Smoothness},
	    booktitle = {Computer Vision - ECCV 2016 Workshops, Part 3},
	    year = {2016}
	}

## Description
The code provided has been improved beyond what was described in the original paper. The original work focused on only a loss based on photometric constancy and motion smoothness. This can be achieved by disabling additional loss terms in the hyper-parameter file. This code has only been tested on python 2.7 and TensorFlow 1.8.0.
## Usage
### Hardware
It is recommended to train using an NVIDIA GPU with minimum 8GB of VRAM.
### Software and Libraries
- Python 2.7
- TensorFlow 1.8.0 (And its requirements)
- Pillow
- NumPy
- SciPy

## Scripts and Modules
[train.py](src/train.py): Script to train network  
[test.py](src/test.py): Script for tested trained network  
[components](src/components): Helpers to build graph for network and losses  
[util](src/util): Larger components such as the network, losses, etc.  
[hyperparams.json](src/hyperparams.json): Contains all hyper-parameters, flags for different loss terms, and settings for training  
