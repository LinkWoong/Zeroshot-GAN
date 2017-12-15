# Zeroshot-GAN

## Intro

GAN requires lots of training data, so we want to design a novel architecture that could help training data generation. Specifically, Chinese characters are used in this investigation. That is, we try to use limited training data to capture the writing style of user. The model could learn right now, however, requires 20% of target characters. For instance, if 10000 samples are desired so 2000 training set should be fed into the nets.

## Architecture
The whole architecture looks like 
<img src="https://raw.githubusercontent.com/LinkWoong/Zeroshot-GAN/master/images/model.png" width="900px"/> 
and for the encoder part
<img src="https://raw.githubusercontent.com/LinkWoong/Zeroshot-GAN/master/images/generator.png" width="900px"/> 

## Results
<img src="https://raw.githubusercontent.com/LinkWoong/Zeroshot-GAN/master/images/combine006.png" width="900px"/> 
The left one is generated characters, and the right one is handwriting.

## People

This is my final year project in [Pattern Recognition & Machine Intelligence Laboratory](http://www.premilab.com/). PremiLab is a new lab set up by the [dean](https://scholar.google.com.hk/citations?user=3l5B0joAAAAJ&hl=en) of EEE department, XJTLU.
