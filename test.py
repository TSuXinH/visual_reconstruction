from os import read
from cv2 import transform
import h5py
import numpy as np
from numpy.lib.polynomial import polysub
import matplotlib.pyplot as plt
import json 
import pickle
from scipy.io import savemat

# def read_numpy(file):
#     f = h5py.File(file,'r') 
#     return np.asarray(f[list(f.keys())[0]][()]).astype(np.float32)

with open('./data/valid_WFZ200.json', 'rb') as f:
    pls2 = pickle.load(f)

rotation = pls2.x_rotations_

importance = np.sum(abs(rotation),axis=1)
savemat('./data/valid_WFZ200_importance.mat',{'importance':importance})
