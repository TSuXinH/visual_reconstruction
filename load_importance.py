import numpy as np
import pickle
from scipy.io import savemat
from torch.serialization import save
transform_file = './data/valid_WFZ1000.json'

with open(transform_file, 'rb') as f:
    pls2 = pickle.load(f)
    weights = pls2.x_weights_
    savemat('./data/valid_WFZ1000.mat',{'x_weights':weights})