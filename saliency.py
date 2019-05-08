'''
Please refer to https://github.com/Ugness/PiCANet-Implementation for more detail
'''

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from saliency_network import Unet


class SliencyModel():
    def __init__(self, pth, device):
        self.device = torch.device(device)
        state_dict = torch.load(pth, map_location=self.device)
        model = Unet().to(self.device)
        model.load_state_dict(state_dict, strict=False)
        self.model = model.eval()

        self.pre_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

    def compute_saliency(self, pilimg):
        pilimg = pilimg.convert('RGB')
        post_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((pilimg.size[1], pilimg.size[0]))])

        with torch.no_grad():
            x = self.pre_transform(pilimg).unsqueeze(0).to(self.device)
            pred, loss = self.model(x)
            saliency = post_transform(pred[5].squeeze(0))

        return saliency


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', default='saliency.pth',
                        help="Directory of pre-trained model, you can download at \n"
                             "https://drive.google.com/drive/folders/1s4M-_SnCPMj_2rsMkSy3pLnLQcgRakAe?usp=sharing")
    parser.add_argument('--device', default='cpu',
                        help="Device to run the model")
    parser.add_argument('--img', required=True,
                        help="Input image")
    parser.add_argument('--out', required=True,
                        help="Output saliency map")
    args = parser.parse_args()

    img = Image.open(args.img).convert('RGB')
    model = SliencyModel(args.pth, args.device)
    saliency = model.compute_saliency(img)
    Image.blend(img, saliency.convert('RGB'), alpha=0.8).save(args.out)
