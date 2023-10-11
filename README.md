# SSMU-Net: A Style Separation and Mode Unification Network for Multimodal Remote Sensing Image Classification
Code for " A Style Separation and Mode Unification Network for Multimodal Remote Sensing Image Classification".

Yi Han, Hao Zhu, Licheng Jiao, Xiaoyu Yi, Xiaotong Li, Biao Hou, Wenping Ma, Shuang Wang

Xidian University

>The rapid progress in remote sensing technology has made it convenient for satellites to capture both multispectral (MS) and panchromatic (PAN) images. MS has more spectral information, and PAN has higher spatial resolution. How to exploit the complementarity between MS and PAN images, and effectively combine their respective advantageous features while alleviating mode differences, has become a crucial research task. This paper designs a Style Separation and Mode Unification network (SSMU-Net) for MS and PAN image classification from a novel and effective perspective. The network can be divided into two stages: style separation and mode unification. In the style separation stage, we use wavelet decomposition and techniques similar to generative adversarial networks to preliminarily separate the information of MS and PAN into different components. These components better preserve complete information from the original data and have their own advantages in style and content. Then we propose a Symmetrical Triplet Traction module to perform style traction on different components, making style features more unique and content features more unified, achieving feature separation and purification. In the mode unification stage, we design an encoder-decoder model to reduce the impact of mode differences. The experimental results from multiple datasets validate the effectiveness of our proposed method. Our overall accuracy improved by approximately 4% on the Shanghai and Beijing datasets, and it has exceeded 99.28% on the Hohhot and Vancouver datasets.

![readme_overall](https://github.com/proudpie/SSMU-Net/assets/134203137/df354067-187d-4afc-bf8c-1ac8a0df8981)

## Data
PAN and MS images

the ground truth file

## Preprocessing
We provide four codes for preprocessing different datasets. 

We only need to import the preprocessing code instead of running them.

## Models
For different datasets, what needs to be modified is the number of categories in the classfication network.

## Train
The selection of hyper-parameters is based on the original text of our paper. (The overall impact is not significant.)

## Classification maps
two types: the image with ground truth and the overall image

Please modify the dataloader for different types of classification results' visualizations.
