import numpy as np
import torch
import sys
import time
import argparse
from torchvision import transforms as T
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim, nn
from models.aes import AE
from x_train_autoencoder import Train, FrameDataset, arch_params, tensor2visual


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='train autoencoder')
    parser.add_argument('--with_sigmoid', type=bool)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--n_ae_latents', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    neu_path = r'./x_data/neurons.mat'
    video_path = r'./x_data/frames_cropped'
    period_len = 1004
    tensorboard_path = './tensorboard'
    ckpt_path = './ckpt/ae_adam_lr{}_ep{}_bs256_hid{}_{}.pth'.format(args.lr, args.max_epoch, args.n_ae_latents, 'wsig' if args.with_sigmoid else 'wosig')

    args.with_label = False
    args.device = 'cuda'
    args.alter_train_method = False
    args.tensorboard = SummaryWriter(log_dir=tensorboard_path)
    args.ckpt_path = ckpt_path
    args.transform = T.Compose([
        T.RandomHorizontalFlip(p=.5),
        T.ToTensor(),
        T.RandomApply(
            nn.ModuleList([
                T.RandomRotation(45),
            ]),
            p=.3
        ),
        # T.Pad(16),
        # T.TenCrop((256, 256)),
        # T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
        # T.Lambda(lambda crops: torch.stack([T.RandomAffine(degrees=45, translate=(0, .2), scale=(.8, 1.2))(crop) for crop in crops])),
        # T.Lambda(lambda crops: torch.stack([T.RandomPerspective(distortion_scale=.2)(crop) for crop in crops])),
    ])
    args.ten_crop = False
    args.lr = 5e-4
    args.batch_size = 256

    hparams = arch_params
    hparams['with_sigmoid'] = args.with_sigmoid
    hparams['n_ae_latents'] = args.n_ae_latents
    model = AE(hparams)

    # print(model)
    # sys.exit()

    dataset = FrameDataset(video_path, args.transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

    # x = np.random.randint(0, len(dataset))
    # z = dataset.__getitem__(x)
    # print(x)
    # tensor2visual(z)
    # sys.exit()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    train_setting = Train(model, loader, criterion, optimizer, args)
    print('start training')
    train_setting.train()

    end = time.time()
    print('total time: {}'.format(end - start))
