import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_format', required=True)
parser.add_argument('--h', default=1024, type=int)
parser.add_argument('--w', default=768, type=int)
parser.add_argument('--crop', action='store_true')

args = parser.parse_args()

# Read all images
img_paths = []
for i in range(100000):
    path = args.img_format % i
    if os.path.isfile(path):
        if not args.crop:
            print('Processing', path)
            Image.open(path).resize((args.w, args.h), Image.LANCZOS).save(path)
        else:
            im = Image.open(path)
            width, height = im.size
            left = (width - args.w)/2
            top = (height - args.h)/2
            right = (width + args.w)/2
            bottom = (height + args.h)/2
            box = (left, top, right, bottom)
            im.crop(box).save(path)
    else:
        break
