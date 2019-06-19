# Estimation of Acoustic Impedance from Seismic Data using Temporal Convolutional Network
Ahmad Mustafa, [Motaz Alfarraj](http://www.motaz.me), and [Ghasssan AlRegib](http://www.ghassanalregib.com) 

This repository includes the codes for the paper:

'**Estimation of Acoustic Impedance from Seismic Data using Temporal Convolutional Network**' that was recently 
accepted to SEG Technical Program Expanded Abstracts 2019.

[preprint](https://arxiv.org/abs/1906.02684)

The code has been built in Python in the popular deep learning framework, [Pytorch](https://pytorch.org/).

## Abstract

In exploration seismology, seismic inversion refers to the process of inferring physical properties of the subsurface 
from seismic data. Knowledge of physical properties can prove helpful in identifying key structures in the subsurface 
for hydrocarbon exploration. In this work, we propose a workflow for predicting acoustic impedance (AI) from seismic 
data using a network architecture based on Temporal Convolutional Network by posing the problem as that of sequence 
modeling. The proposed workflow overcomes some of the problems that other network architectures usually face, like 
gradient vanishing in Recurrent Neural Networks, or overfitting in Convolutional Neural Networks. The proposed workflow
was used to predict AI on Marmousi 2 dataset with an average r2 coefficient of 91% on a hold-out validation set. 
 
## Dataset
Create a directory called `data`. Download the data from this 
[link](https://www.dropbox.com/s/jly7m44r84ecw0c/data.zip?dl=0) and unzip the contents of the file in the `data` folder.

## Running the code



## Citation 
If you have found our code and data useful, we humbly request you to cite our work. You can cite the arXiv preprint:
```tex
@incollection{amustafa2019AI,
title=Estimation of Acoustic Impedance from Seismic Data using Temporal Convolutional Network,
author=Mustafa, Ahmad and AlRegib, Ghassan,
booktitle=arXiv:1906.02684,
year=2019,
publisher=Society of Exploration Geophysicists}
```
The arXiv preprint is available at: [https://arxiv.org/abs/1906.02684](https://arxiv.org/abs/1906.02684)

## Questions?
The code and the data are provided as is with no guarantees. If you have any questions, regarding the dataset or the 
code, you can contact me at (amustafa9@gatech.edu), or even better, open an issue in this repo and we will do our best 
to help. 