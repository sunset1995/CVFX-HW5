# CVFX Homework5 Team11

This is NTHU CVFX course project 5 of `team 11`. Here we use saliency mask to help crating many types of *multi-view 3D visual effects* including:
- motion parallax
- stop motion
- live photo
Also with the help of saliency mask, we can post-processing the image for to fix the color of certain pixels and thus enhance the final effect (detail describe in below).


## Saliency mask
Saliency is defined by the most noticeable part in the image for a human. The use of saliency mask in this multi-view visual effects project is obivious. We use the mask the help deciding which features to track or filter, instead of defining rule (e.g. the translation of the features) to classify the detected features into foreground and background.

We use a state-of-the-art saliency prediction model, PiCANet ([paper](https://arxiv.org/abs/1708.06433), [github](https://github.com/Ugness/PiCANet-Implementation), Liu et al., CVPR'18), to yield saliency mask of each image for later 3D visual effects.


## Multi-view 3D visual effects

In below, we will present the results of each effect and briefly describe them, showing the results of different setting.

### Motion Parallax
In this effect, we want to align the two images such that the main role, the saliency, move as little as possible. To to this, we use the saliency mask the filter all non foreground sift features and align the two images based on the features inside the saliency mask.

The first example is a `city caffe`, if we simply stack the two image without doing anything we will get:

| Img1 | Img2 | Do nothing |
| :--: | :--: | :--: |
| ![](imgs/motion_parallax/city_caffe/img0.jpg) | ![](imgs/motion_parallax/city_caffe/img1.jpg) | ![](imgs/motion_parallax/city_caffe/out_raw.gif) |

We use saliency mask to keep sift features of the `city caffe`. Note that the saliency masks are actually a grey images, we blend them with the original color image for better visualization:

| Saliency of Img1 | Saliency of Img2 | Keeped SIFT features |
| :--------------: | :--------------: | :------------------: |
| ![](imgs/motion_parallax/city_caffe/img0_saliency.jpg) | ![](imgs/motion_parallax/city_caffe/img1_saliency.jpg) | ![](imgs/motion_parallax/city_caffe/out_match.jpg) |

Finally, we align Img1 to Img2 by homography and also showing the result of fixing pixels in saliency region:

| Do nothing | Align | Fix saliency pixels |
| :--------: | :---: | :-----------------: |
| ![](imgs/motion_parallax/city_caffe/out_raw.gif) | ![](imgs/motion_parallax/city_caffe/out.gif) | ![](imgs/motion_parallax/city_caffe/out_fix_saliency.gif) |


Yet another example of motion parallax:

| Img1 | Img2 | Do nothing |
| :--: | :--: | :--: |
| ![](imgs/motion_parallax/cats/img0.jpg) | ![](imgs/motion_parallax/cats/img1.jpg) | ![](imgs/motion_parallax/cats/out_raw.gif) |

| Saliency of Img1 | Saliency of Img2 | Keeped SIFT features |
| :--------------: | :--------------: | :------------------: |
| ![](imgs/motion_parallax/cats/img0_saliency.jpg) | ![](imgs/motion_parallax/cats/img1_saliency.jpg) | ![](imgs/motion_parallax/cats/out_match.jpg) |

We show the result by different alignment algorithm. The translation model have 2 degree of freedom and can only left/right/top/down shift the image. The affine model have dof=5 and can translate, scale and inplane rotate the image. The homography model is the strongest and can align features on different 3D planes. Please *Stop Motion* effect for better understanding the different of the three alignment models.

| Align by translation (DoF=2) | Align by affine (DoF=5) | Align by homography (DoF=8) |
| :--------: | :---: | :-----------------: |
| ![](imgs/motion_parallax/cats/out_translation.gif) | ![](imgs/motion_parallax/cats/out_affine.gif) | ![](imgs/motion_parallax/cats/out.gif) |


### Stop Motion
The implementation of stop motion effect is very similar to motion parallax. The only different is that we apply the same process like motion parallax


### Live Photo
- Column 1 shows the saliency maps for all frames.
- Column 2 shows the raw video frames without alignment.
- For column 3,4,5, we use saliency maps to match only background (non-saliency) features, and align all frames with homography model.
- For Column 4 & 5, we create a fixed background (BG) image by taking median along all frames, then fill the BG pixels (based on the ```BG mask```) in each frame with the corresponding pixels of the median BG image.
- The ```BG mask``` for column 4 is defined as the sum of the standard deviations of R/G/B channels along all frames, and the ```BG mask``` for column 5 is defined as the union of salency maps of all frames.

#### Example: mediatek
|          | saliency |     raw        |   only align   |  fix BG w/ RGB variance | fix BG w/ union of saliency |
| :------: | :------: | :------------: | :--------------: | :------------------: | :------: |
|  Result  | ![](imgs/live_photo/mediatek/saliency.gif) | ![](imgs/live_photo/mediatek/out_raw.gif) | ![](imgs/live_photo/mediatek/out.gif) | ![](imgs/live_photo/mediatek/out_rgb.gif) | ![](imgs/live_photo/mediatek/out_saliency.gif) |
|  BG mask | | | |  ![](imgs/live_photo/mediatek/out_rgb.png) | ![](imgs/live_photo/mediatek/out_saliency.png) |

#### Example: standing human
|          | saliency |     raw        |   only align   |  fix BG w/ RGB variance | fix BG w/ union of saliency |
| :------: | :------: | :------------: | :--------------: | :------------------: | :------: |
|  Result  | ![](imgs/live_photo/stand/saliency.gif) | ![](imgs/live_photo/stand/out_raw.gif) | ![](imgs/live_photo/stand/out.gif) | ![](imgs/live_photo/stand/out_rgb.gif) | ![](imgs/live_photo/stand/out_saliency.gif) |
|  BG mask | | | |  ![](imgs/live_photo/stand/out_rgb.png) | ![](imgs/live_photo/stand/out_saliency.png) |

#### Example: tissue
- In this example, objects on different planes of different depths are predicted as saliency, so the alignment is a little bit worse than the previous example.
- The ```fix BG w/ union of saliency``` fails as the saliency model only recognize part of tissue.

|          | saliency |    raw         |   only align   |  fix BG w/ RGB variance | fix BG w/ union of saliency |
| :------: | :------: | :------------: | :--------------: | :------------------: | :------: |
|  Result  | ![](imgs/live_photo/tissue/saliency.gif) | ![](imgs/live_photo/tissue/out_raw.gif) | ![](imgs/live_photo/tissue/out.gif) | ![](imgs/live_photo/tissue/out_rgb.gif) | ![](imgs/live_photo/tissue/out_saliency.gif) |
|  BG mask | | | |  ![](imgs/live_photo/tissue/out_rgb.png) | ![](imgs/live_photo/tissue/out_saliency.png) |
