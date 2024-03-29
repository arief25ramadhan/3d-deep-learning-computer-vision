import cv2
import torch
import time
import numpy as np


# Load a MiDaS model for depth estimation
model_type = 'DPT_Large'
# model_type = 'DPT_Hybrid'
# model_type = 'MiDaS_small'

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

# Open up the video capture webcam
def webcam():

    cap = cv2.VideoCapture(2)

    while cap.isOpened():

        succes, img = cap.read()

        start = time.time()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            # Resize to original input size. 
            # Because after transform, the image size is changed
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, 
        norm_type = cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        end = time.time()
        totalTime = end-start

        fps = 1 / totalTime

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        depth_map = (depth_map*255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Image', img)
        cv2.imshow('Depth Map', depth_map) 

        if cv2.waitKey(5) & 0xFF==27:
            break
    
    cap.release()
        

def inference(image_path):
      
    start = time.time()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        # Resize to original input size. 
        # Because after transform, the image size is changed
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, 
    norm_type = cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    end = time.time()
    totalTime = end-start

    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    cv2.imwrite('depth_map.jpg', depth_map)

    print('totalTime: ', totalTime)
    

image_path = 'car.png'
inference(image_path)