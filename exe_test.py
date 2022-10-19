import json
from train_decoder import get_frames

s = './models/ae_model_arch.json'
tmp = open(s)
result = json.load(tmp)

for item in result.keys():
    print(item, result[item])


import torch
print(torch.cuda.is_available())


import numpy as np
import cv2
import matplotlib.pyplot as plt
tmp1 = np.load('./data/ae_data.npy').T
plt.plot(tmp1[0])
plt.show(block=True)

video_path = './data/Reaglined MRI video.avi'
cap = cv2.VideoCapture(video_path)
frame = get_frames(cap)

tensor_path = r'./alter10/events.out.tfevents.1664260149.DESKTOP-GBQJ4PV.17216.0'
