'''
Please refer to https://github.com/Ugness/PiCANet-Implementation for more detail
'''

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from saliency_network import Unet


class SliencyModel():
    def __init__(self, pth='saliency.pth', device='cpu'):
        self.device = torch.device(device)
        state_dict = torch.load(pth, map_location=self.device)
        model = Unet().to(self.device)
        model.load_state_dict(state_dict, strict=False)
        self.model = model.eval()

        self.pre_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

    def compute_saliency(self, pilimg, ret_feature=False):
        pilimg = pilimg.convert('RGB')
        post_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((pilimg.size[1], pilimg.size[0]))])

        with torch.no_grad():
            x = self.pre_transform(pilimg).unsqueeze(0).to(self.device)
            pred, features = self.model(x)
            saliency = post_transform(pred.squeeze(0))
            features = features.squeeze(0).cpu().numpy()

        if ret_feature:
            return saliency, features
        return saliency


if __name__ == '__main__':

    import os
    import glob
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', default='saliency.pth',
                        help="Directory of pre-trained model, you can download at \n"
                             "https://drive.google.com/drive/folders/1s4M-_SnCPMj_2rsMkSy3pLnLQcgRakAe?usp=sharing")
    parser.add_argument('--device', default='cpu',
                        help="Device to run the model")
    parser.add_argument('--imgs', required=True,
                        help="Input image")
    args = parser.parse_args()

    print('Preparing saliency model')
    model = SliencyModel(args.pth, args.device)

    for path in glob.glob(args.imgs):
        print('Processing', path)
        out_saliency = path[:-4] + '_saliency.png'
        out_features = path[:-4] + '_features.npy'
        out_blend = path[:-4] + '_saliency.jpg'
        img = Image.open(path).convert('RGB')
        saliency, features = model.compute_saliency(img, ret_feature=True)
        saliency.save(out_saliency)
        Image.blend(img, saliency.convert('RGB'), alpha=0.8).save(out_blend)
        np.save(out_features, features)
        print(features.min(), features.max(), features.shape)
