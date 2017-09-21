# Zeroshot-GAN

## Overview

Cases of large computation memory consumptions are often encountered during the training process of the model, which requiring decent and expensive GPUs. Facing this regular issue, our group was wondering creating a model that does not take numerious of data during the training. Instead, we only use very small portion of data as training, say 10%. 

## Related work

Indeed there are some researchers have implemented the idea of **One-Shot** or **Zero-Shot**, however, the results are limited to the training set and they are not generative. For instance, the [cGAN-based Manga Colorization Using a Single Training Image](https://arxiv.org/pdf/1706.06918.pdf) is quite an amazing idea. Their result is obtained from using one single image as training set, which is of great simplicity. Here is their result
<img src="https://raw.githubusercontent.com/LinkWoong/Zeroshot-GAN/master/image.png" width="900px"/>

--------------------------------------------------

In our project, we will generate chinese handwriting characters instead of colorization of images. It is more difficult to generate Chinese characters due to its complexity and abundance. In fact, the first level of Chinese characters including 3755 and second level contains 3008 characters. 

## Initial ideas

The very first thing that comes to my mind is using GAN and Convnets, their performance on different kinds of task is successful. Then why uses of small amount of data could learn the entire style of dataset? How this prediction is implemented? Well, this is a new area of reaserch and our group are dealing with it. 

## People

This is my final year project in [Pattern Recognition & Machine Intelligence Laboratory](http://www.premilab.com/). PremiLab is a new lab set up by the [dean](https://scholar.google.com.hk/citations?user=3l5B0joAAAAJ&hl=en) of EEE department, XJTLU.