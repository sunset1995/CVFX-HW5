import cv2
import imageio
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage

import feature_matcher
from saliency import SliencyModel


def raw_live_photo(img_paths):
    return [imageio.imread(path) for path in img_paths]


def wrap_hue(arr, upbound):
    arr[np.where(arr > (upbound//2))] -= upbound
    return arr

def take_median_background(image_list, bgmask):
    bg_image = np.median(image_list, axis=0)
    for i in range(len(image_list)):
        image_list[i][bgmask] = bg_image[bgmask]
    return image_list


def saliency_sift_live_photo(img_paths, model='homography', avg='no', thres=155,
                              xmin=0.1, xmax=0.9, ymin=0.1, ymax=0.9):
    images = [imageio.imread(path) for path in img_paths]
    saliencies = [
        imageio.imread(path[:-4] + '_saliency.png') for path in img_paths]
    method = feature_matcher.CreateMethod.sift_bf_ratiotest()

    Ms = []
    match_imgs = []
    for i in range(1, len(images)):
        img1 = images[i-1]
        img2 = images[i]
        saliency1 = saliencies[i-1]
        saliency2 = saliencies[i]

        # Detect features
        kp1, des1, kp2, des2, matches = method(img1, img2)

        # Filter by saliency
        p1 = [[int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])] for m in matches]
        p2 = [[int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1])] for m in matches]
        matches = [
            matches[i] for i in range(len(matches))
            if saliency1[p1[i][1], p1[i][0]] < thres and saliency2[p2[i][1], p2[i][0]] < thres
        ]

        # Drawing matched result
        match_imgs.append(cv2.drawMatches(
            img1, kp1, img2, kp2, matches,
            None, flags=2))

        # Model image plane movement
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        if model == 'homography':
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10)
        elif model == 'translation':
            src_m = np.median(src_pts, 0)[0]
            dst_m = np.median(dst_pts, 0)[0]
            M = np.eye(3).astype(np.float32)
            M[0, 2] = dst_m[0] - src_m[0]
            M[1, 2] = dst_m[1] - src_m[1]
        elif model == 'affine':
            M = cv2.estimateRigidTransform(src_pts, dst_pts, fullAffine=False)
            M = np.array([M[0], M[1], [0, 0, 1]])
        else:
            raise Exception()

        Ms.append(M)

    # Align all images with middle frame
    height, width = images[-1].shape[:2]
    M = [
        np.eye(3).astype(np.float32)
        for _ in range(len(images))
    ]
    target_i = len(images) // 2
    for i in range(0, target_i):
        for j in range(i, target_i):
            M[i] = Ms[j] @ M[i]
    for i in range(target_i+1, len(M)):
        for j in range(i, target_i, -1):
            M[i] = np.linalg.inv(Ms[j-1]) @ M[i]

    # Warping all images
    warp_imgs = [
        cv2.warpPerspective(images[i], M[i], (width, height))
        for i in range(len(M))
    ]

    # Crop all black region
    xmin = int(xmin if xmin > 1 else xmin * images[0].shape[1])
    xmax = int(xmax if xmax > 1 else xmax * images[0].shape[1])
    ymin = int(ymin if ymin > 1 else ymin * images[0].shape[0])
    ymax = int(ymax if ymax > 1 else ymax * images[0].shape[0])
    for i in range(len(warp_imgs)):
        warp_imgs[i] = warp_imgs[i][ymin:ymax, xmin:xmax]
        saliencies[i] = saliencies[i][ymin:ymax, xmin:xmax]

    # Average all background images
    #all_frames = np.stack(warp_imgs, axis=0)
    if avg != 'no':
        if avg == 'rgb_stddev':
            stddev = np.std(warp_imgs, axis=0)
            stddev = np.mean(stddev, axis=-1)

            stddev = ndimage.gaussian_filter(stddev, sigma=(5, 5), order=0)
            rgb_thres = np.mean(stddev) * 2
            bgmask = stddev < rgb_thres
        
        elif avg == 'saliency':  # union of salienct maps of all frames
            union = saliencies[0].copy() >= thres
            for s in saliencies[1:]:
                union += (s>=thres)
            bgmask = ~union        

        # elif avg == 'hsv_stddev':
        #     ''' hue: [0,180], saturation: [0, 255], Value: [0, 255]'''
        #     all_frames = np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in all_frames], axis=0)
        #     h = all_frames[..., 0] 
        #     h = wrap_hue(h, 180)
        #     s = all_frames[..., 1]
        #     v = all_frames[..., 2]
        #     stddev = np.std(h, axis=0)  # (h, w)
        #     stddev = ndimage.gaussian_filter(stddev, sigma=(2, 2), order=0)
        #     stddev = stddev.astype(np.uint8)
        
        warp_imgs = take_median_background(warp_imgs, bgmask)
        
        imageio.imsave(args.out.replace('gif', 'png'), (255*bgmask).astype(np.uint8))  # visualize bg mask
        outdir = os.path.dirname(args.out)
        imageio.mimsave(os.path.join(outdir, 'saliency.gif'), saliencies, 'GIF-FI', palettesize=256, fps=args.fps)

    return warp_imgs, match_imgs


if __name__ == '__main__':

    import os
    import sys
    import glob
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_format', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--out_match_format')
    parser.add_argument('--mode', choices=['raw', 'saliency_sift'], default='saliency_sift')
    parser.add_argument('--model', choices=['homography', 'translation', 'affine'], default='homography')
    parser.add_argument('--fps', default=17, type=int)
    parser.add_argument('--fixbg', choices=['no', 'saliency', 'rgb_stddev', 'hsv_stddev'], default='no')
    args = parser.parse_args()

    # Read all images
    img_paths = []
    for i in range(100000):
        path = args.img_format % i
        if os.path.isfile(path):
            img_paths.append(path)
        else:
            break
    if len(img_paths) <= 1:
        print('No enough images found !?')
        sys.exit()

    # Generate motion parallex
    if args.mode == 'raw':
        images = raw_live_photo(img_paths)
    elif args.mode == 'saliency_sift':
        img_paths = img_paths[:]
        images, match_imgs = saliency_sift_live_photo(img_paths, args.model, args.fixbg)
    else:
        raise NotImplementedError()

    # Saved result as gif
    imageio.mimsave(args.out, images, 'GIF-FI', palettesize=256, fps=args.fps)
    if args.out_match_format:
        for i in range(len(match_imgs)):
            imageio.imsave(args.out_match_format % i, match_imgs[i])
