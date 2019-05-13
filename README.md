# Sailency Based Multi-View 3D Visual Effect

This is NTHU CVFX course project 5 of `team 11`. Here we present a ...

## Conventional method

## Saliency Mask

## Application
### Motion Parallax

### Stop Motion

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


## Limitation & Disccusion
