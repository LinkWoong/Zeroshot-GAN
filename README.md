# Zeroshot-GAN

## Overview

Cases of large computation memory consumptions are often encountered during the training process of the model, which requiring decent and expensive GPUs. Facing this regular issue, our group was wondering creating a model that does not take numerious of data during the training. Instead, we only use very small portion of data as training, such as 10% 

## Results
### 40 Epochs
<img src="https://raw.githubusercontent.com/LinkWoong/Zeroshot-GAN/master/images/sample_26_1500.png" width="100px" />
It is just CRAP, only 40-epoch training is used. 

### 120 Epochs
<img src="https://raw.githubusercontent.com/LinkWoong/Zeroshot-GAN/master/images/sample_113_6350.png" width="100px" />
This seems better, 300-epoch training is on the way.
 
## Logs (Update on Oct.26th.2017)

> Epoch: [ 0], [  15/  56] time: 61.0244, d_loss: 24.58650, g_loss: 138.14224, category_loss: 0.00000, cheat_loss: 104.68754, const_loss: 2.15147, l1_loss: 31.30323, tv_loss: 0.00000<br>
Epoch: [ 0], [  16/  56] time: 64.4936, d_loss: 95.96950, g_loss: 38.68196, category_loss: 0.00000, cheat_loss: 4.63433, const_loss: 0.74106, l1_loss: 33.30657, tv_loss: 0.00000<br>
Epoch: [ 0], [  17/  56] time: 67.9703, d_loss: 36.20229, g_loss: 92.88159, category_loss: 0.00000, cheat_loss: 63.53211, const_loss: 0.33416, l1_loss: 29.01531, tv_loss: 0.00000<br>
Epoch: [ 0], [  18/  56] time: 71.4461, d_loss: 35.16706, g_loss: 34.27352, category_loss: 0.00000, cheat_loss: 0.00000, const_loss: 0.12292, l1_loss: 34.15059, tv_loss: 0.00000<br>
Epoch: [ 0], [  19/  56] time: 74.9350, d_loss: 64.25539, g_loss: 126.85447, category_loss: 0.00000, cheat_loss: 92.53378, const_loss: 0.06005, l1_loss: 34.26064, tv_loss: 0.00000<br>
Epoch: [ 0], [  20/  56] time: 78.4193, d_loss: 6.55383, g_loss: 29.47481, category_loss: 0.00000, cheat_loss: 0.00000, const_loss: 0.15530, l1_loss: 29.31951, tv_loss: 0.00000<br>
Epoch: [ 0], [  21/  56] time: 81.9037, d_loss: 58.20816, g_loss: 87.30659, category_loss: 0.00000, cheat_loss: 54.24187, const_loss: 0.30953, l1_loss: 32.75519, tv_loss: 0.00000<br>
Epoch: [ 0], [  22/  56] time: 85.4986, d_loss: 51.28491, g_loss: 42.22787, category_loss: 0.00000, cheat_loss: 2.64490, const_loss: 0.47864, l1_loss: 39.10434, tv_loss: 0.00000<br>
Epoch: [ 0], [  23/  56] time: 89.2431, d_loss: 24.99184, g_loss: 60.38929, category_loss: 0.00000, cheat_loss: 28.54649, const_loss: 0.17622, l1_loss: 31.66659, tv_loss: 0.00000<br>
Epoch: [ 0], [  24/  56] time: 92.9491, d_loss: 7.62794, g_loss: 76.92113, category_loss: 0.00000, cheat_loss: 42.01003, const_loss: 0.39682, l1_loss: 34.51428, tv_loss: 0.00000<br>
Epoch: [ 0], [  25/  56] time: 96.6540, d_loss: 11.37754, g_loss: 32.95026, category_loss: 0.00000, cheat_loss: 0.00000, const_loss: 0.48561, l1_loss: 32.46465, tv_loss: 0.00000<br>
Epoch: [ 0], [  26/  56] time: 100.3963, d_loss: 83.95575, g_loss: 125.36658, category_loss: 0.00000, cheat_loss: 93.82712, const_loss: 0.27872, l1_loss: 31.26073, tv_loss: 0.00000<br>
Epoch: [ 0], [  27/  56] time: 104.2243, d_loss: 41.33235, g_loss: 29.70360, category_loss: 0.00000, cheat_loss: 0.00093, const_loss: 0.28001, l1_loss: 29.42266, tv_loss: 0.00000<br>
Epoch: [ 0], [  28/  56] time: 108.0938, d_loss: 36.62217, g_loss: 105.36942, category_loss: 0.00000, cheat_loss: 76.07848, const_loss: 0.40882, l1_loss: 28.88213, tv_loss: 0.00000<br>
Epoch: [ 0], [  29/  56] time: 111.9638, d_loss: 68.66853, g_loss: 35.65288, category_loss: 0.00000, cheat_loss: 0.90976, const_loss: 0.66164, l1_loss: 34.08148, tv_loss: 0.00000<br>
Epoch: [ 0], [  30/  56] time: 115.8335, d_loss: 43.61385, g_loss: 72.99104, category_loss: 0.00000, cheat_loss: 40.52444, const_loss: 0.50118, l1_loss: 31.96540, tv_loss: 0.00000<br>

I've grabbed some of the logs at the very beginning of training. **category_loss** and **tv_loss** are zero ...I don't think it is a good start. 120-epoch training is on the way.

## People

This is my final year project in [Pattern Recognition & Machine Intelligence Laboratory](http://www.premilab.com/). PremiLab is a new lab set up by the [dean](https://scholar.google.com.hk/citations?user=3l5B0joAAAAJ&hl=en) of EEE department, XJTLU.
