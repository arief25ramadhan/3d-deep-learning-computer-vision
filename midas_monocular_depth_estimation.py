import cv2
import torch
import time
import numpy as np


# Load a MiDaS model for depth estimation
model_type = 'MiDaS_small'
midas = torch.hub.load('intel-isl/MiDaS', model_type)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

if model_type == 'DPT_Large' or model_type == 'DPT_Hybrid':
    transform = midas_transforms.dpt_transform

else:
    transform = midas_transforms.small_transform