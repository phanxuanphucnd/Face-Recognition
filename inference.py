import argparse

import cv2
import torch
import numpy as np

from backbones.utils import get_model
from model_irse import IR_50

INPUT_SIZE = [112, 112]
RGB_MEAN = [0.5, 0.5, 0.5] 
RGB_STD = [0.5, 0.5, 0.5]


@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    # net = get_model(name, fp16=False)
    net = IR_50(INPUT_SIZE)
    
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(weight))
    else:
        net.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))

    net.eval()
    print(img.size())
    feat = net(img).numpy()
    
    return feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='models/ir50/backbone_ir50_asia.pth')
    
    img1 = './data/phucpx1.jpg'
    img2 = './data/phucpx2.jpg'
    
    args = parser.parse_args()
    
    feat1 = inference(args.weight, args.network, img1)
    feat2 = inference(args.weight, args.network, img2)

    cosin = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cosin(torch.Tensor(feat1), torch.Tensor(feat2))
    print(output)