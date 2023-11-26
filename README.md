#  Few-shot Gaussian Splatting
This is my **unofficial** implementation of: Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images (Jaeyoung Chung, Jeongtaek Oh, Kyoung Mu Lee) [Paper link](https://arxiv.org/abs/2311.13398)

**TLDR:** Gaussian Splatting requires a large number of training images, or it tends to overfit the training data, because of the lack of geometry cues. They employ some tricks to mitigate this issue and train with sparse (2-5) views.

### Comparison (5 views)
<img src="https://github.com/pablodawson/few-shot-gaussian-splatting/assets/74978011/b0c8e37b-cd4f-4f4c-b62c-15590a924bac" width="300" > <img src="https://github.com/pablodawson/few-shot-gaussian-splatting/assets/74978011/bc627a44-b1d7-4acd-b666-c2f851c56b8a" width="300" >

*They solve this problem by:*

### Depth regularization
 
- Create a dense depth map for each image (monocular depth estimation), then align it with the points obtained with COLMAP.

- Obtain the l1 loss between the aligned depth map and the rendered depth.

- Stop training when the depth loss starts to rise. I found it more effective to do this based on the test-set loss as it's more stable, could be wrong though.

 ### Smoothness regularization

 - They create a smoothness loss, making sure depth values are close on similar positions.

### Hyperparameters

- 3rd degree spherical harmonics are too complex for sparse views. Limit them to 1. (This seems to be the biggest factor in my tests)
- Opacity reset does not help either. Remove it.

## Train
Same as the original. Add input images to your `dataset path`, inside a folder called `input`.

Then convert:
```
python convert.py -s dataset_path
```
A COLMAP dataset will be created, with the corresponding aligned depth maps in the `depth_aligned` folder.

Then, as usual:
```
python train.py -s dataset_path --eval
```
