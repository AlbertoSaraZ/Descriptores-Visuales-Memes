import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from DescriptorComputer import DescriptorComputer

class ResNet152Computer(DescriptorComputer):
	
    def __init__(self):
        resnet152 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True)
        modules=list(resnet152.children())[:-1]
        resnet152=nn.Sequential(*modules)
        for p in resnet152.parameters():
            p.requires_grad = False
        self.resnet152 = resnet152
        self.resnet152.to('cuda')
        
    def compute(self, img):
        #torch.cuda.empty_cache()
        img = cv2.resize(img, (224,224))
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img)
        tens = input_tensor.unsqueeze(0)
        tens = tens.to('cuda')
        return np.resize(self.resnet152(tens).cpu().numpy(), (2048))