import cv2
import numpy as np


class CreateMethod():
    def sift_bf_crosscheck(**kwargs):
        detector = cv2.xfeatures2d.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        def run(img1, img2):
            kp1, des1 = detector.detectAndCompute(img1, None)
            kp2, des2 = detector.detectAndCompute(img2, None)
            matches = matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            return kp1, des1, kp2, des2, matches

        return run

    def sift_bf_ratiotest(**kwargs):
        kwargs['ratio'] = kwargs.get('ratio', 0.7)
        detector = cv2.xfeatures2d.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        def run(img1, img2):
            kp1, des1 = detector.detectAndCompute(img1, None)
            kp2, des2 = detector.detectAndCompute(img2, None)
            matches = matcher.knnMatch(des1, des2, k=2)
            matches = [m for m, n in matches if m.distance < kwargs['ratio'] * n.distance]
            matches = sorted(matches, key=lambda x: x.distance)
            return kp1, des1, kp2, des2, matches

        return run

    def surf_bf_crosscheck(**kwargs):
        kwargs['extended'] = kwargs.get('extended', False)
        detector = cv2.xfeatures2d.SURF_create(extended=True)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        def run(img1, img2):
            kp1, des1 = detector.detectAndCompute(img1, None)
            kp2, des2 = detector.detectAndCompute(img2, None)
            matches = matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            return kp1, des1, kp2, des2, matches

        return run

    def surf_bf_ratiotest(**kwargs):
        kwargs['extended'] = kwargs.get('extended', False)
        kwargs['ratio'] = kwargs.get('ratio', 0.7)
        detector = cv2.xfeatures2d.SURF_create(extended=True)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        def run(img1, img2):
            kp1, des1 = detector.detectAndCompute(img1, None)
            kp2, des2 = detector.detectAndCompute(img2, None)
            matches = matcher.knnMatch(des1, des2, k=2)
            matches = [m for m, n in matches if m.distance < kwargs['ratio'] * n.distance]
            matches = sorted(matches, key=lambda x: x.distance)
            return kp1, des1, kp2, des2, matches

        return run

    def surfext_bf_crosscheck(**kwargs):
        kwargs['extended'] = True
        return CreateMethod.surf_bf_crosscheck(**kwargs)

    def surfext_bf_ratiotest(**kwargs):
        kwargs['extended'] = True
        return CreateMethod.surf_bf_ratiotest(**kwargs)

    def orb_bf_crosscheck(**kwargs):
        detector = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        def run(img1, img2):
            kp1, des1 = detector.detectAndCompute(img1, None)
            kp2, des2 = detector.detectAndCompute(img2, None)
            matches = matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            return kp1, des1, kp2, des2, matches

        return run

    def orb_bf_ratiotest(**kwargs):
        kwargs['ratio'] = kwargs.get('ratio', 0.7)
        detector = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        def run(img1, img2):
            kp1, des1 = detector.detectAndCompute(img1, None)
            kp2, des2 = detector.detectAndCompute(img2, None)
            matches = matcher.knnMatch(des1, des2, k=2)
            matches = [m for m, n in matches if m.distance < kwargs['ratio'] * n.distance]
            matches = sorted(matches, key=lambda x: x.distance)
            return kp1, des1, kp2, des2, matches

        return run


def drawMatches(method_name, img1, img2, top=50, distfilter=False, homocheck=False, **kwargs):
    method = CreateMethod.__dict__[method_name](**kwargs)
    kp1, des1, kp2, des2, matches = method(img1, img2)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    dist = ((src_pts - dst_pts)**2).sum(1)
    idx = dist.argsort()
    matches = [matches[i] for i in idx]

    if homocheck:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches = [m for m, v in zip(matches, mask.ravel()) if v]

    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:top],
        None, flags=2)
    return match_img


if __name__ == '__main__':

    import os
    import sys
    import glob
    import argparse

    METHODS = [k for k in CreateMethod.__dict__.keys() if k[:2] != '__']

    # Arguments parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path1', required=True)
    parser.add_argument('--path2', required=True)
    parser.add_argument('--patho', required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--homocheck', action='store_true')
    parser.add_argument('--top', default=1000, type=int)
    parser.add_argument('--oscale', default=1, type=float)
    args = parser.parse_args()

    # Read two images
    img1 = cv2.imread(args.path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(args.path2, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        print('Image not found')
        sys.exit()

    # Draw matches
    match_img = drawMatches(args.method, img1, img2,
                            top=args.top,
                            homocheck=args.homocheck)

    # Resize and save result
    if args.oscale != 1:
        match_img = cv2.resize(match_img, None, fx=args.oscale, fy=args.oscale)
    cv2.imwrite(args.patho, match_img)
