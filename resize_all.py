import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_format', required=True)
parser.add_argument('--h', default=1024, type=int)
parser.add_argument('--w', default=768, type=int)
args = parser.parse_args()

# Read all images
img_paths = []
for i in range(100000):
    path = args.img_format % i
    if os.path.isfile(path):
        print('Processing', path)
        Image.open(path).resize((args.w, args.h), Image.LANCZOS).save(path)
    else:
        break
