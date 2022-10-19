import torch
import os
import json
from PIL import Image
import numpy as np
from models.aes import AE
from easydict import EasyDict
from x_train_autoencoder import arch_params, test_show_contrast


args = EasyDict()
args.device = 'cuda'
path = './ckpt/ae_adam_lr0.005_ep500_bs256_hid512_wsig.pth'
frame_path = './x_data/frames_cropped'
arch_params['n_ae_latents'] = 512
arch_params['with_sigmoid'] = True
model = AE(arch_params)
state = torch.load(path)
model.load_state_dict(state['model'])

dirs = os.listdir(frame_path)
x = np.random.randint(0, len(dirs))
img = Image.open(os.path.join(frame_path, dirs[x])).convert('L')
test_show_contrast(model, img, 'cuda')

