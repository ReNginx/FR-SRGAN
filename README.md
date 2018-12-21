# FR-SRGAN

Final Project for MIT 6.819 Advances in Computer Vision

## Required Package

```
pytorch
skimage
numpy
cv2
tqdm
```

## Usage

TODO

## Overview

This project is based on the following two papers.

> Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network https://arxiv.org/abs/1609.04802
>
> Frame-Recurrent Video Super-Resolution https://arxiv.org/abs/1801.04590

We are trying to combine these two models together in our project (this is the where the name come from "Frame Recurrent Super Resolution GAN").

## Dataset

We used the dataset from [toflow-dataset](http://data.csail.mit.edu/tofu/testset/vimeo_test_clean.zip), which is about 15GB, containing 7.8k video clips with 7 frames per clip.

## Results

We trained our model on the dataset for 9 epochs. and compared with default SRGAN model in the repo. **We did not retrain SRGAN in our dataset**, so the result is for reference only.

The following results are produced by 7-epoch model. 

### Temporal Profile

#### SRGAN

![image-20181220171220312](https://i.postimg.cc/PpgSyQgj/image-20181220171220312.png)

#### FRVSR

![image-20181220171308894](https://i.postimg.cc/tnF2MVbV/image-20181220171308894.png)

#### FR-SRGAN

![image-20181220171408311](https://i.postimg.cc/cggDfb5h/image-20181220171408311.png)

#### Ground Truth

![image-20181220171412148](https://i.postimg.cc/TyH7L6Qh/image-20181220171412148.png)



### Comparison
[![FrjF9s.md.gif](https://s1.ax1x.com/2018/12/21/FrjF9s.md.gif)](https://imgchr.com/i/FrjF9s)

## Acknowledgement

We referred to [SRGAN Implementation](https://github.com/leftthomas/SRGAN) by LeftThomas in our project. We would like to thank the author for his fabulous work.
