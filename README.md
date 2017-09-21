# Zeroshot-GAN

## Overview

Cases of large computation memory consumptions are often encountered during the training process of the model, which requiring decent and expensive GPUs. Facing this regular issue, our group was wondering creating a model that does not take numerious of data during the training. Instead, we only use very small portion of data as training, say 10%. 

## Related work

Indeed there are some researchers have implemented the idea of **One-Shot** or **Zero-Shot**, however, the results are limited to the training set and they are not generative. For instance, the ()[https://arxiv.org/pdf/1706.06918.pdf]